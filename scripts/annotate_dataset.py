import os
import pandas as pd
from dotenv import load_dotenv
from google import genai
import src.annotation.ai_annotation as ai_annotator

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_ANNOTATOR_API_KEY"))

with open("data/silver/remaining_annotations.csv", 'r', encoding='utf-8', errors='replace') as f:
    dataset = pd.read_csv(f)

enriched_dataset = ai_annotator.enrich_with_techniques(
                        df=dataset,
                        text_column="Full_Text",
                        client=client,
                        prompt_path="prompts/annotator/annotation_prompt.md",
                        system_prompt_path="prompts/annotator/system_prompt.md",
                        output_file="data/gold/remaining_enriched_dataset.csv",
                        model_name="gemini-2.5-flash",
                        batch_size=5,
                        checkpoint_interval=25,
                        request_delay=30.0,
                        max_retries=5
                        )

print(enriched_dataset.head())

os._exit(0)

