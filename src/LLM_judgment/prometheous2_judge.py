import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class SocialEngineeringEvalJudge:
    def __init__(self, model_id: str = "prometheus-eval/prometheus-7b-v2.0", device: str = "cuda"):
        """Initializes Prometheus 2 7B evaluator in Reference-Free Mode."""
        print(f"Loading Evaluator Model: {model_id} on {device}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Safe-guard for tokenizer padding configuration
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map=device
        )
        
        # Official Prometheus 2 Reference-Free Direct Template
        self.absolute_template = (
            "###Task Description:\n{task_description}\n\n"
            "###Criterion:\n{criterion}\n\n"
            "###Instruction:\n{instruction}\n\n"
            "###Response:\n{response}\n\n"
            "###Score Rubric:\n{rubric}\n\n"
            "###Feedback:"
        )

        # Official Prometheus 2 Reference-Free Pairwise Template
        self.relative_template = (
            "###Task Description:\n{task_description}\n\n"
            "###Instruction:\n{instruction}\n\n"
            "###Response A:\n{response_a}\n\n"
            "###Response B:\n{response_b}\n\n"
            "###Score Rubric:\n{rubric}\n\n"
            "###Feedback:"
        )

    def _generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generates text from the evaluator LLM under tight parameters."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # FIXED: Removed temperature=0.0 because do_sample=False already enforces deterministic greedy decoding.
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        generated_tokens = outputs[inputs['input_ids'].shape[-1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def _extract_score(self, text: str) -> int:
        """Parses the text to extract Prometheus 2 score (1-5)."""
        # Improved regex to handle variations like "score is 4" or "[RESULT] 4"
        match = re.search(r"(?:score is|\[result\])\s*([1-5])", text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _extract_winner(self, text: str) -> str:
        """Parses the text to extract the relative evaluation winner."""
        if re.search(r"Response A is much better|Response A is slightly better", text, re.IGNORECASE):
            return "A"
        elif re.search(r"Response B is much better|Response B is slightly better", text, re.IGNORECASE):
            return "B"
        elif re.search(r"equal in quality", text, re.IGNORECASE):
            return "Tie"
        
        # Fallback filter for aggressive choice text variations (e.g. "correct choice is Response A")
        fallback_match = re.search(r"choice is\s*Response\s*([A-B])", text, re.IGNORECASE)
        if fallback_match:
            return fallback_match.group(1).upper()
            
        return "Unknown"

    def evaluate_direct(self, instruction: str, response: str, rubrics: dict) -> dict:
        """Performs absolute grading (Scores 1-5) without a reference answer."""
        prompt = self.absolute_template.format(
            task_description=rubrics["task_description"],
            criterion=rubrics["criterion"],
            instruction=instruction,
            response=response,
            rubric=rubrics["direct_rubric"]
        )
        raw_feedback = self._generate(prompt)
        return {
            "feedback": raw_feedback,
            "parsed_score": self._extract_score(raw_feedback)
        }

    def evaluate_pairwise(self, instruction: str, response_a: str, response_b: str, rubrics: dict) -> dict:
        """Performs pairwise grading without a reference answer."""
        prompt = self.relative_template.format(
            task_description=rubrics["task_description"],
            instruction=instruction,
            response_a=response_a,
            response_b=response_b,
            rubric=rubrics["pairwise_rubric"]
        )
        raw_feedback = self._generate(prompt)
        return {
            "feedback": raw_feedback,
            "parsed_decision": self._extract_winner(raw_feedback)
        }
