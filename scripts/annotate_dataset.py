import os
import pandas as pd
from google import genai
import src.annotation.ai_annotation as ai_annotator

client = genai.Client(api_key=os.environ.get("GEMINI_ANNOTATOR_API_KEY"))
dataset = pd.read_csv("data/silver/unified_dataset.csv")

enriched_dataset = ai_annotator.enrich_with_techniques(
                        df=dataset,
                        text_column="Full_Text",
                        client=client,
                        prompt_path="../prompts/annotator/annotation_prompt.md",
                        system_prompt_path="../prompts/annotator/system_prompt.md",
                        output_file="../data/gold/enriched_dataset.csv",
                        model_name="gemini-2.5-flash",
                        batch_size=50,
                        checkpoint_interval=10,
                        request_delay=30.0,
                        max_retries=5,
                        id_column="Message_ID",
                        )

print(enriched_dataset.head())

os._exit(0)

