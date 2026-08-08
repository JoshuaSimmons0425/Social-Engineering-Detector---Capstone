import os
import json
import time
import random
import pandas as pd
from google import genai
from google.genai import types
from typing import Any
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError

class TechniqueResult(BaseModel):
    label: bool
    confidence: float = Field(ge=0.0, le=1.0)

class TechniqueSchema(BaseModel):

    urgency: TechniqueResult
    fear: TechniqueResult
    authority: TechniqueResult
    reciprocity: TechniqueResult
    curiosity: TechniqueResult
    pretexting: TechniqueResult
    promotional: TechniqueResult
    transactional: TechniqueResult
    reminder: TechniqueResult
    personal: TechniqueResult

class BatchAnnotationItem(BaseModel):
    message_id: str
    techniques: TechniqueSchema

class BatchTechniqueSchema(BaseModel):
    annotations: list[BatchAnnotationItem]

def _build_batch_prompt(
    prompt_template: str,
    messages: list[dict[str, str]],
) -> str:
    """
    Insert a JSON-like collection of messages into the prompt template.

    The prompt template must contain the placeholder:
        {{MESSAGES}}
    """
    formatted_messages = "\n\n".join(
        [
            (
                f'<message id="{message["message_id"]}">\n'
                f'{message["text"]}\n'
                f"</message>"
            )
            for message in messages
        ]
    )

    return prompt_template.replace("{{MESSAGES}}", formatted_messages).replace("{{MESSAGE_COUNT}}", str(len(messages)))


def _create_annotation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create output columns if they do not already exist."""
    technique_names = list(TechniqueSchema.model_fields.keys())

    for technique in technique_names:
        label_column = f"{technique}_label"
        confidence_column = f"{technique}_confidence"

        if label_column not in df.columns:
            df[label_column] = pd.Series(
                pd.NA,
                index=df.index,
                dtype="boolean",
            )

        if confidence_column not in df.columns:
            df[confidence_column] = float("nan")

    metadata_defaults: dict[str, Any] = {
        "annotation_status": "pending",
        "annotation_error": pd.NA,
        "annotation_model": pd.NA,
    }

    for column, default_value in metadata_defaults.items():
        if column not in df.columns:
            df[column] = default_value

    return df


def _write_annotation_to_row(
    df: pd.DataFrame,
    row_index: Any,
    annotation: TechniqueSchema,
    model_name: str,
) -> None:
    """Flatten one structured annotation into DataFrame columns."""
    technique_results = annotation.model_dump()

    for technique, result in technique_results.items():
        df.at[row_index, f"{technique}_label"] = result["label"]
        df.at[row_index, f"{technique}_confidence"] = result["confidence"]

    df.at[row_index, "annotation_status"] = "success"
    df.at[row_index, "annotation_error"] = pd.NA
    df.at[row_index, "annotation_model"] = model_name


def _save_checkpoint(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Persist current progress to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def enrich_with_techniques(
    df: pd.DataFrame,
    client,
    prompt_path: str | Path,
    system_prompt_path: str | Path,
    output_file: str | Path = "enriched_example.csv",
    text_column: str = "text",
    id_column: str | None = None,
    model_name: str = "gemini-2.5-flash-lite",
    batch_size: int = 10,
    checkpoint_interval: int = 5,
    request_delay: float = 1.0,
    max_retries: int = 5,
) -> pd.DataFrame:
    """
    Enrich message rows with LLM-generated technique labels.

    Each Gemini request processes up to `batch_size` messages. The returned
    message IDs are used to map annotations back to their original rows.

    The prompt file must contain:
        {{MESSAGES}}

    Args:
        df:
            Source DataFrame containing one message per row.
        client:
            Initialised google.genai Client.
        prompt_path:
            Path to the normal annotation prompt.
        system_prompt_path:
            Path to the taxonomy/system instruction.
        output_file:
            CSV used for checkpoints and final output.
        text_column:
            Name of the column containing the message body.
        id_column:
            Optional stable identifier column. When omitted, DataFrame index
            values are used as message IDs.
        model_name:
            Gemini model used for annotation.
        batch_size:
            Maximum number of rows sent in one API request.
        checkpoint_interval:
            Save after this many successfully completed API batches.
        request_delay:
            Delay in seconds after each API request.
        max_retries:
            Maximum number of API attempts for each batch.

    Returns:
        A copy of the DataFrame containing flattened labels, confidences,
        and annotation metadata.
    """
    if text_column not in df.columns:
        raise KeyError(
            f"Text column {text_column!r} was not found. "
            f"Available columns: {list(df.columns)}"
        )

    if id_column is not None and id_column not in df.columns:
        raise KeyError(
            f"ID column {id_column!r} was not found. "
            f"Available columns: {list(df.columns)}"
        )

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be at least 1.")

    if max_retries < 1:
        raise ValueError("max_retries must be at least 1.")

    prompt_path = Path(prompt_path)
    system_prompt_path = Path(system_prompt_path)
    output_path = Path(output_file)

    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    if not system_prompt_path.is_file():
        raise FileNotFoundError(
            f"System prompt file not found: {system_prompt_path}"
        )

    prompt_template = prompt_path.read_text(encoding="utf-8")
    system_prompt = system_prompt_path.read_text(encoding="utf-8")

    if "{{MESSAGES}}" not in prompt_template:
        raise ValueError(
            "The annotation prompt must contain the placeholder "
            "'{{MESSAGES}}'."
        )

    enriched_df = _create_annotation_columns(df.copy())

    pending_indices: list[Any] = []

    for row_index, row in enriched_df.iterrows():
        if row.get("annotation_status") == "success":
            continue

        message = row[text_column]

        if pd.isna(message) or not str(message).strip():
            enriched_df.at[row_index, "annotation_status"] = "skipped"
            enriched_df.at[row_index, "annotation_error"] = (
                "Message text was empty."
            )
            continue

        pending_indices.append(row_index)

    successful_batches_since_checkpoint = 0

    for batch_start in range(0, len(pending_indices), batch_size):
        batch_indices = pending_indices[
            batch_start : batch_start + batch_size
        ]

        batch_messages: list[dict[str, str]] = []
        id_to_index: dict[str, Any] = {}

        for row_index in batch_indices:
            row = enriched_df.loc[row_index]

            if id_column is not None:
                message_id = str(row[id_column])
            else:
                message_id = str(row_index)

            if message_id in id_to_index:
                raise ValueError(
                    f"Duplicate message ID encountered: {message_id!r}. "
                    "Use a column containing unique identifiers."
                )

            id_to_index[message_id] = row_index

            batch_messages.append(
                {
                    "message_id": message_id,
                    "text": str(row[text_column]).strip(),
                }
            )

        batch_prompt = _build_batch_prompt(
            prompt_template=prompt_template,
            messages=batch_messages,
        )

        last_error: Exception | None = None
        batch_succeeded = False

        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=batch_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=BatchTechniqueSchema,
                        temperature=0.0,
                    ),
                )

                if not response.text:
                    raise ValueError("Gemini returned an empty response.")

                parsed_response = BatchTechniqueSchema.model_validate_json(
                    response.text
                )

                returned_annotations = parsed_response.annotations

                if len(returned_annotations) != len(batch_messages):
                    raise ValueError(
                        "Gemini returned an incorrect number of annotations. "
                        f"Expected {len(batch_messages)}, "
                        f"received {len(returned_annotations)}."
                    )

                returned_ids = {
                    item.message_id for item in returned_annotations
                }
                expected_ids = set(id_to_index.keys())

                missing_ids = expected_ids - returned_ids
                unexpected_ids = returned_ids - expected_ids

                if missing_ids or unexpected_ids:
                    raise ValueError(
                        "Returned message IDs did not match the submitted "
                        f"batch. Missing IDs: {sorted(missing_ids)}; "
                        f"unexpected IDs: {sorted(unexpected_ids)}."
                    )

                if len(returned_ids) != len(returned_annotations):
                    raise ValueError(
                        "Gemini returned duplicate message IDs."
                    )

                for item in returned_annotations:
                    row_index = id_to_index[item.message_id]

                    _write_annotation_to_row(
                        df=enriched_df,
                        row_index=row_index,
                        annotation=item.techniques,
                        model_name=model_name,
                    )

                batch_succeeded = True
                successful_batches_since_checkpoint += 1
                break

            except (ValidationError, ValueError) as error:
                # These errors may be caused by malformed model output, so
                # retrying the request can still succeed.
                last_error = error

            except Exception as error:
                # Includes API, network, quota, and transient SDK errors.
                last_error = error

            if attempt < max_retries:
                retry_delay = min(60, 5 * (2 ** (attempt - 1)))
                retry_delay += random.uniform(0, 3)

                print(
                    f"Batch beginning at position {batch_start} failed on "
                    f"attempt {attempt}/{max_retries}: {last_error}. "
                    f"Retrying in {retry_delay} seconds."
                )

                time.sleep(retry_delay)

        if not batch_succeeded:
            error_message = (
                str(last_error)
                if last_error is not None
                else "Unknown batch annotation error."
            )

            for row_index in batch_indices:
                enriched_df.at[
                    row_index,
                    "annotation_status",
                ] = "failed"

                enriched_df.at[
                    row_index,
                    "annotation_error",
                ] = error_message

                enriched_df.at[
                    row_index,
                    "annotation_model",
                ] = model_name

            print(
                f"Annotation failed for batch beginning at position "
                f"{batch_start}: {error_message}"
            )

        if (
            successful_batches_since_checkpoint >= checkpoint_interval
            or batch_start + batch_size >= len(pending_indices)
        ):
            _save_checkpoint(
                df=enriched_df,
                output_path=output_path,
            )
            successful_batches_since_checkpoint = 0

        if request_delay > 0:
            time.sleep(request_delay)

    _save_checkpoint(
        df=enriched_df,
        output_path=output_path,
    )

    return enriched_df
    