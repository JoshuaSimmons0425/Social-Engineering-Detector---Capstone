import os
import json
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# -------------------------------------------------------------
# Define Pydantic response contracts to enforce clean JSON 
# -------------------------------------------------------------
class DirectEvaluationResponse(BaseModel):
    evaluation_steps_analysis: str = Field(
        description="Detailed step-by-step G-Eval audit reasoning and text-based critique."
    )
    score: int = Field(
        description="The final calculated metric score strictly from 1 (poor) to 5 (excellent)."
    )

class PairwiseEvaluationResponse(BaseModel):
    evaluation_steps_analysis: str = Field(
        description="Analytical breakdown detailing why one response outperformed the other."
    )
    winner: str = Field(
        description="The definitive choice of matching winner. Must be exactly 'A', 'B', or 'Tie'."
    )


class GeminiEvalJudge:
    def __init__(self, model_id: str = "gemini-2.5-flash"):
        """Initializes Gemini Flash client utilizing system environment variables."""
        # Automatically retrieves GEMINI_API_KEY from environment vars
        self.client = genai.Client()
        self.model_id = model_id

        # Bias mitigation anchor rules
        self.bias_system_instruction = (
            "You are an elite, impartial Cyber Security Evaluation Judge specializing in Social Engineering. "
            "CRITICAL SYSTEMIC BIAS INSTRUCTIONS TO FOLLOW:\n"
            "1. Verbosity Bias: Ignore output length. A short, highly precise, direct answer is strictly superior "
            "to a long, wordy risk assessment filled with generic security filler text.\n"
            "2. Style Bias: Ignore markdown formatting, bold headers, or bullet lists. Focus entirely on the accuracy, "
            "soundness, and context awareness of the underlying technical security evaluation logic."
        )

        # G-Eval Inspired Reference-Free Direct Template
        self.absolute_template = (
            "###Task Description:\n{task_description}\n\n"
            "###Criterion:\n{criterion}\n\n"
            "###Context Message Instructions:\n{instruction}\n\n"
            "###Candidate SLM Response to Evaluate:\n{response}\n\n"
            "###Score Rubric Matrix Guidance:\n{rubric}\n\n"
            "Execute step-by-step logical reasoning to grade the Candidate SLM Response based on the instructions."
        )

        # G-Eval Inspired Reference-Free Pairwise Template
        self.relative_template = (
            "###Task Description:\n{task_description}\n\n"
            "###Context Message Instructions:\n{instruction}\n\n"
            "###Response A:\n{response_a}\n\n"
            "###Response B:\n{response_b}\n\n"
            "###Score Rubric Matrix Guidance:\n{rubric}\n\n"
            "Compare Response A and Response B side-by-side. Decide which response displays higher analytical value."
        )

    def evaluate_direct(self, instruction: str, response: str, rubrics: dict) -> dict:
        """Performs absolute grading (Scores 1-5) returning validated structured data."""
        prompt = self.absolute_template.format(
            task_description=rubrics["task_description"],
            criterion=rubrics["criterion"],
            instruction=instruction,
            response=response,
            rubric=rubrics["direct_rubric"]
        )

        # Request native structured JSON matching the Pydantic schema
        res = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.bias_system_instruction,
                response_mime_type="application/json",
                response_schema=DirectEvaluationResponse,
                temperature=0.1  # Set low temperature for evaluation reproducibility
            ),
        )

        # Parse data safely
        data = json.loads(res.text)
        return {
            "feedback": data["evaluation_steps_analysis"],
            "parsed_score": data["score"]
        }

    def evaluate_pairwise(self, instruction: str, response_a: str, response_b: str, rubrics: dict) -> dict:
        """Performs pairwise grading utilizing structural tracking schemas."""
        prompt = self.relative_template.format(
            task_description=rubrics["task_description"],
            instruction=instruction,
            response_a=response_a,
            response_b=response_b,
            rubric=rubrics["pairwise_rubric"]
        )

        res = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.bias_system_instruction,
                response_mime_type="application/json",
                response_schema=PairwiseEvaluationResponse,
                temperature=0.1
            ),
        )

        data = json.loads(res.text)
        return {
            "feedback": data["evaluation_steps_analysis"],
            "parsed_decision": data["winner"]
        }