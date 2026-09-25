import os
import torch
import json
import numpy as np
import scipy as sp
import pandas as pd
import shap
from lime.lime_text import LimeTextExplainer
from transformers_interpret import SequenceClassificationExplainer
from src.datasets.dataengine import DataEngine

class ExplainableBert:
    def __init__(self, binary_model, multilabel_model, binary_calibrators, multilabel_calibrators, tokenizer, input_text, multi_label_names, device='cpu'):
        self.binary_model = binary_model
        self.multilabel_model = multilabel_model
        self.binary_calibrators = binary_calibrators
        self.multilabel_calibrators = multilabel_calibrators
        self.tokenizer = tokenizer
        self.input_text = input_text
        self.device = device

        self.binary_model.to(self.device).eval()
        self.multilabel_model.to(self.device).eval()

        self.binary_class_names = ['malicious', 'benign']
        self.multilabel_label_names = multi_label_names

        self.binary_information = {}
        self.multilabel_information = {}
        self.overall_information = {}

        self.clean_text = ""
        self.processed_tensors = None

    def preprocess_input(self):

        engine = DataEngine()
        temp_df = pd.DataFrame({'text': [self.input_text]})

        temp_df = engine.mask_money(temp_df, 'text')
        temp_df = engine.anonymize_data(temp_df, 'text')

        self.clean_text = temp_df['text'].iloc[0]

        self.processed_tensors = self.tokenizer.encode_plus(
            self.clean_text, 
            return_tensors='pt',
            add_special_tokens=True,
            max_length=1024,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            )

    def _get_binary_platt_params(self):
        """Helper to resolve calibration parameters from dictionary or PlattScaler PyTorch parameters safely."""
        # 1. Handle if it's your PlattScaler object containing nn.Parameter tensors
        if hasattr(self.binary_calibrators, 'A') and hasattr(self.binary_calibrators, 'B'):
            # Use .item() to safely pull out raw float parameters, bypassing PyTorch tensor wrapper errors
            a_val = self.binary_calibrators.A.item() if hasattr(self.binary_calibrators.A, 'item') else float(self.binary_calibrators.A)
            b_val = self.binary_calibrators.B.item() if hasattr(self.binary_calibrators.B, 'item') else float(self.binary_calibrators.B)
            return a_val, b_val
            
        # 2. Handle if binary_calibrators is passed as a raw dictionary
        if isinstance(self.binary_calibrators, dict):
            return self.binary_calibrators.get('A', 1.0), self.binary_calibrators.get('B', 0.0)
        
        # 3. Handle if it is an object with generic attributes (like .slope / .intercept)
        for attr_a, attr_b in [('slope', 'intercept'), ('a', 'b')]:
            if hasattr(self.binary_calibrators, attr_a):
                val_a = getattr(self.binary_calibrators, attr_a)
                val_b = getattr(self.binary_calibrators, attr_b)
                a_val = val_a.item() if hasattr(val_a, 'item') else float(val_a)
                b_val = val_b.item() if hasattr(val_b, 'item') else float(val_b)
                return a_val, b_val
                
        return 1.0, 0.0

    def predict_binary(self):
        assert self.processed_tensors is not None, "Input has not been preprocessed. Call preprocess_input() first."

        tensors = {k: v.to(self.device) for k, v in self.processed_tensors.items()}
        with torch.no_grad():
            output = self.binary_model(**tensors)
            # Safe extraction regardless of direct tensor vs structured model output dict object
            logits = output.logits if hasattr(output, 'logits') else output
            raw_logits = logits.cpu().numpy()

        # Apply Platt Scaling calibration parameters dynamically
        A, B = self._get_binary_platt_params()
        calibrated_logits = raw_logits * A + B
        
        # Softmax over the calibrated pair
        exp_logits = np.exp(calibrated_logits - np.max(calibrated_logits, axis=-1, keepdims=True))
        probabilities = (exp_logits / exp_logits.sum(axis=-1, keepdims=True)).flatten()
        
        pred_idx = int(np.argmax(probabilities))
        self.binary_information['prediction'] = {
            'predicted_class': self.binary_class_names[pred_idx],
            'confidence': float(probabilities[pred_idx]),
            'probabilities': {self.binary_class_names[i]: float(p) for i, p in enumerate(probabilities)}
        }
        return self.binary_information['prediction']

    def predict_multilabel(self):
        assert self.processed_tensors is not None, "Input has not been preprocessed. Call preprocess_input() first."

        tensors = {k: v.to(self.device) for k, v in self.processed_tensors.items()}
        with torch.no_grad():
            outputs = self.multilabel_model(**tensors)
            raw_logits = outputs.logits.cpu().numpy()[0] # array of shape (10,)
            
        calibrated_probs = {}
        active_techniques = []
        
        # Iterate over all 10 sigmoid classification heads
        for idx, label in enumerate(self.multilabel_label_names):
            # Fetch the specific PlattScaler object instance for this index or label name
            cal_scaler = self.multilabel_calibrators.get(idx, self.multilabel_calibrators.get(label))
            
            # Resolve slope (A) and intercept (B) from the PlattScaler PyTorch parameters
            if cal_scaler is not None and hasattr(cal_scaler, 'A') and hasattr(cal_scaler, 'B'):
                A = cal_scaler.A.item() if hasattr(cal_scaler.A, 'item') else float(cal_scaler.A)
                B = cal_scaler.B.item() if hasattr(cal_scaler.B, 'item') else float(cal_scaler.B)
            elif isinstance(cal_scaler, dict):
                A = cal_scaler.get('A', cal_scaler.get('slope', 1.0))
                B = cal_scaler.get('B', cal_scaler.get('intercept', 0.0))
            else:
                A, B = 1.0, 0.0
            
            cal_logit = raw_logits[idx] * A + B
            # Apply Sigmoid for independent multilabel probabilities
            prob = 1 / (1 + np.exp(-cal_logit))
            calibrated_probs[label] = float(prob)
            
            if prob >= 0.1:
                active_techniques.append(label)
                
        self.multilabel_information['prediction'] = {
            'active_techniques': active_techniques,
            'probabilities': calibrated_probs
        }
        return self.multilabel_information['prediction']

    def explain_binary(self):
        A, B = self._get_binary_platt_params()

        BATCH_SIZE = 32  # Define a batch size for processing texts in batches

        def calibrated_binary_probs(texts):
            # Convert NumPy string types or generic arrays explicitly to native Python string lists
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            elif not isinstance(texts, list):
                texts = [str(texts)]
            else:
                texts = [str(t) for t in texts]

            all_probs = []
            
            # Process texts in small batches to preserve CUDA VRAM
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                
                enc = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=1024, 
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    out = self.binary_model(**enc)
                    logits = out.logits if hasattr(out, 'logits') else out
                    logits = logits.cpu().numpy()
                
                cal_logits = logits * A + B
                
                if cal_logits.shape[-1] == 1:
                    malicious_probs = 1 / (1 + np.exp(-cal_logits))
                    batch_probs = np.hstack([1.0 - malicious_probs, malicious_probs])
                else:
                    exp_l = np.exp(cal_logits - np.max(cal_logits, axis=-1, keepdims=True))
                    batch_probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
                
                all_probs.append(batch_probs)
                
                # Clear structural residual memory pools explicitly
                del enc, out, logits, cal_logits
                torch.cuda.empty_cache()

            return np.vstack(all_probs)

        def calibrated_binary_logits(texts):
            # Convert NumPy string types or generic arrays explicitly to native Python string lists
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            elif not isinstance(texts, list):
                texts = [str(texts)]
            else:
                texts = [str(t) for t in texts]

            all_logits = []
            
            # Process texts in small batches to prevent SHAP from overwhelming the memory stack
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                
                enc = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=1024, 
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    out = self.binary_model(**enc)
                    logits = out.logits if hasattr(out, 'logits') else out
                    logits = logits.cpu().numpy()
                
                batch_cal_logits = logits * A + B
                all_logits.append(batch_cal_logits)
                
                del enc, out, logits
                torch.cuda.empty_cache()

            return np.vstack(all_logits)

        torch.cuda.empty_cache()

        # Execute LIME
        lime_explainer = LimeTextExplainer(class_names=self.binary_class_names)
        lime_exp = lime_explainer.explain_instance(self.clean_text, calibrated_binary_probs, num_features=10)
        
        # Execute SHAP
        shap_explainer = shap.Explainer(calibrated_binary_logits, self.tokenizer)
        shap_values = shap_explainer([self.clean_text])
        
        # Safely index target dim whether binary calibration outputs 1D or 2D array
        target_idx = 1 if len(shap_values.shape) > 2 and shap_values.shape[-1] > 1 else 0
        shap_slice = shap_values[0, :, target_idx]

        # Execute Transformers-Interpret wrapped in an architectural safety block
        ti_attributions = "Skipped: Architecture hook mismatch"
        try:
            # Re-map target to your wrapper's inner huggingface module (adjust attribute name like .model or .bert if different)
            hf_binary_module = self.binary_model.bert if hasattr(self.binary_model, 'bert') else self.binary_model.model
            
            orig_forward = hf_binary_module.forward
            def patched_forward(*args, **kwargs):
                out = orig_forward(*args, **kwargs)
                if hasattr(out, 'logits'):
                    out.logits = (out.logits * A) + B
                    return out
                return (out * A) + B
            
            hf_binary_module.forward = patched_forward
            
            # Explicitly instruct Captum how to find ModernBERT's unique embedding footprint
            ti_explainer = SequenceClassificationExplainer(
                model=hf_binary_module, 
                tokenizer=self.tokenizer,
                custom_labels=self.binary_class_names
            )
            
            # Use the explicit layer tracking parameter mapping for ModernBERT
            ti_attributions = ti_explainer(
                self.clean_text, 
                embedding_name="model.embeddings.tok_embeddings"
            )[:15]
            
            hf_binary_module.forward = orig_forward
        except Exception as e:
            ti_attributions = f"Skipped: Gradient graph tracing error ({str(e)})"

        self.binary_information['explanations'] = {
            'lime': lime_exp.as_list(),
            'shap': [{'token': t, 'val': float(v)} for t, v in zip(shap_slice.data, shap_slice.values) if t.strip()],
            'transformers_interpret': ti_attributions
        }

    def explain_multilabel(self):
        # Ensure we have predictions to pull probabilities from
        probs_dict = self.multilabel_information.get('prediction', {}).get('probabilities', {})
        if not probs_dict:
            self.predict_multilabel()
            probs_dict = self.multilabel_information['prediction']['probabilities']
            
        # Segregate labels based on your 10% threshold rule
        active_labels = [label for label, prob in probs_dict.items() if prob >= 0.10]
        below_threshold_labels = [label for label, prob in probs_dict.items() if prob < 0.10]

        # Set a conservative sub-batch size to protect the 8GB GPU memory allocation profile
        BATCH_SIZE = 32

        # Black-box pipeline for SHAP that outputs all 10 calibrated logits safely inside memory boundaries
        def calibrated_multilabel_logits(texts):
            # Ensure type safety conversion for NumPy arrays from SHAP
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            elif not isinstance(texts, list):
                texts = [str(texts)]
            else:
                texts = [str(t) for t in texts]

            all_calibrated_logits = []

            # Chunk incoming string evaluations to protect VRAM allocations
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]

                enc = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=1024, 
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.multilabel_model(**enc)
                    batch_logits = outputs.logits.cpu().numpy() # Shape: (batch_len, 10)
                    
                # Apply Platt Calibration to each index lane inside the slice
                for idx, label in enumerate(self.multilabel_label_names):
                    cal_scaler = self.multilabel_calibrators.get(idx, self.multilabel_calibrators.get(label))
                    
                    if cal_scaler is not None and hasattr(cal_scaler, 'A') and hasattr(cal_scaler, 'B'):
                        A = cal_scaler.A.item() if hasattr(cal_scaler.A, 'item') else float(cal_scaler.A)
                        B = cal_scaler.B.item() if hasattr(cal_scaler.B, 'item') else float(cal_scaler.B)
                    elif isinstance(cal_scaler, dict):
                        A = cal_scaler.get('A', cal_scaler.get('slope', 1.0))
                        B = cal_scaler.get('B', cal_scaler.get('intercept', 0.0))
                    else:
                        A, B = 1.0, 0.0
                        
                    batch_logits[:, idx] = batch_logits[:, idx] * A + B

                all_calibrated_logits.append(batch_logits)

                # Explicitly tear down step memory references and purge cache references
                del enc, outputs
                torch.cuda.empty_cache()

            return np.vstack(all_calibrated_logits)

        # Clear active memory pools prior to initiating graph structures
        torch.cuda.empty_cache()

        # Initialize the SHAP explainer once to save compute cycles
        shap_explainer = shap.Explainer(calibrated_multilabel_logits, self.tokenizer)
        shap_values = shap_explainer([self.clean_text])

        # Dictionary to store the multi-label attributions
        explanations_per_active_label = {}

        # Loop through each active technique above 10% to compute individual token attributions
        for target_label in active_labels:
            target_idx = self.multilabel_label_names.index(target_label)
            
            # Extract target label's SHAP attributions from the pre-computed array
            label_shap = [{
                'token': t, 
                'val': float(v)
            } for t, v in zip(shap_values[0, :, target_idx].data, shap_values[0, :, target_idx].values) if t.strip()]

            # Compute Transformers-Interpret with native structural safety fallbacks
            # Compute Transformers-Interpret with native structural safety fallbacks
            ti_attributions = "Skipped: ModernBERT internal layer tracing unmapped"
            try:
                orig_forward = self.multilabel_model.forward
                def patched_forward(*args, **kwargs):
                    out = orig_forward(*args, **kwargs)
                    for idx, label in enumerate(self.multilabel_label_names):
                        cal_scaler = self.multilabel_calibrators.get(idx, self.multilabel_calibrators.get(label))
                        
                        if cal_scaler is not None and hasattr(cal_scaler, 'A') and hasattr(cal_scaler, 'B'):
                            A = cal_scaler.A.item() if hasattr(cal_scaler.A, 'item') else float(cal_scaler.A)
                            B = cal_scaler.B.item() if hasattr(cal_scaler.B, 'item') else float(cal_scaler.B)
                        elif isinstance(cal_scaler, dict):
                            A = cal_scaler.get('A', cal_scaler.get('slope', 1.0))
                            B = cal_scaler.get('B', cal_scaler.get('intercept', 0.0))
                        else:
                            A, B = 1.0, 0.0
                            
                        out.logits[:, idx] = out.logits[:, idx] * A + B
                    return out
                
                self.multilabel_model.forward = patched_forward
                
                # Instantiating the explainer targeting the multi-label head
                ti_explainer = SequenceClassificationExplainer(
                    model=self.multilabel_model, 
                    tokenizer=self.tokenizer,
                    custom_labels=self.multilabel_label_names
                )
                
                # Explicit layer definition enables Integrated Gradients execution over ModernBERT
                ti_attributions = ti_explainer(
                    self.clean_text, 
                    class_index=target_idx,
                    embedding_name="model.embeddings.tok_embeddings"
                )[:15]
                
                self.multilabel_model.forward = orig_forward
            except Exception as e:
                ti_attributions = f"Skipped: Architecture gradient hook error ({str(e)})"

            # Save the pair of local feature explanations for this label
            explanations_per_active_label[target_label] = {
                'shap': label_shap,
                'transformers_interpret': ti_attributions
            }

        # Build the structured metadata summary for your output logs
        self.multilabel_information['explanations'] = {
            'active_labels_explained': active_labels,
            'other_labels_below_10_percent': below_threshold_labels,
            'attributions': explanations_per_active_label
        }

    def run_explanations(self):
        self.preprocess_input()
        self.predict_binary()
        self.predict_multilabel()
        self.explain_binary()
        self.explain_multilabel()

    def _extract_clean_words(self, scored_items, top_k=5):
        """
        Takes a list of dicts/tuples containing raw tokens and attribution scores,
        removes stop words, reconstructs subtokens and structured mask tags using the input string, 
        and returns the top-k highest scoring meaningful words.
        """
        STOP_WORDS = {
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'of', 'at', 'by', 
            'from', 'for', 'in', 'on', 'to', 'with', 'is', 'was', 'were', 'be', 
            'been', 'this', 'that', 'these', 'those', 'it', 'its', 'you', 'your', 'i'
        }
        
        if len(scored_items) == 0:
            return []
            
        if isinstance(scored_items, list) and len(scored_items) > 0 and isinstance(scored_items[0], dict):
            sorted_items = sorted(scored_items, key=lambda x: x.get('val', 0.0), reverse=True)
            raw_tokens = [item['token'] for item in sorted_items if item.get('val', 0.0) > 0.001]
        else:
            sorted_items = sorted(scored_items, key=lambda x: x, reverse=True)
            raw_tokens = [item for item in sorted_items if x > 0.0]

        valid_words = []
        reference_text_lower = self.clean_text.lower()
        
        # 1. EXTRACT STRUCTURED MASK TAGS AND NATURAL WORDS
        import re
        # This regex explicitly matches structural mask fields like <url> or <email> 
        # as complete single units before fallback matching natural words (\b\w+\b)
        source_tokens = re.findall(r'<\w+>|\b\w+\b', reference_text_lower)

        for token in raw_tokens:
            # Strip typical transformer subtoken framing markers
            clean_token = token.replace('##', '').replace('Ġ', '').replace(' ', '').strip().lower()
            
            # Filter out floating punctuation fragments left behind by the tokenizer (like standalone '<', '>', or '[pad]')
            if not clean_token or clean_token in ['[cls]', '[sep]', '[pad]', '<', '>', '[', ']'] or clean_token in STOP_WORDS:
                continue
                
            matched_full_word = None
            
            # 2. MATCH SUBTOKENS AGAINST THE MASKED/NATURAL SOURCE TOKENS
            for source_tok in source_tokens:
                # If the subtoken is part of a mask tag (e.g. 'url' matches '<url>') or a natural word (e.g. 'compl' matches 'compliance')
                if clean_token in source_tok:
                    matched_full_word = source_tok
                    break
            
            target_word = matched_full_word if matched_full_word else clean_token
            
            # Enforce upper case format specifically for our structural tag outputs for better visibility (e.g. <URL>)
            if target_word.startswith('<') and target_word.endswith('>'):
                target_word = target_word.upper()
            
            if target_word not in valid_words and target_word.lower() not in STOP_WORDS:
                valid_words.append(target_word)
                if len(valid_words) >= top_k:
                    break

        return valid_words

    def save_explanations(self, file_path):
        """Compiles explanation artifacts into a cleaned, readable plaintext file (.txt)."""
        binary_probs = self.binary_information.get('prediction', {}).get('probabilities', {})
        malicious_probability = float(binary_probs.get('malicious', 0.0))
        
        # Keep your custom "Critical" risk level assignment
        if malicious_probability >= 0.90:
            risk_level = "Critical"
        elif malicious_probability >= 0.75:
            risk_level = "High"
        elif malicious_probability >= 0.35:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # --- 2. CLEAN CLASSIFICATION EVIDENCE ---
        # Collect and feed raw SHAP tokens straight into our type-safe de-noiser
        shap_binary = self.binary_information.get('explanations', {}).get('shap', [])
        classification_evidence_list = self._extract_clean_words(shap_binary, top_k=5)
        classification_evidence_str = ", ".join(classification_evidence_list) if classification_evidence_list else "no prominent evidence"

        lines = [
            "Classification:",
            f"Malicious probability: {malicious_probability:.2f}",
            f"Risk level: {risk_level}",
            f"Evidence: {classification_evidence_str}",
            "",
            "Detected techniques:"
        ]

        # --- 3. CLEAN MULTI-LABEL TECHNIQUE EVIDENCE ---
        tech_probs = self.multilabel_information.get('prediction', {}).get('probabilities', {})
        tech_explanations = self.multilabel_information.get('explanations', {}).get('attributions', {})
        active_techniques = self.multilabel_information.get('explanations', {}).get('active_labels_explained', [])

        for label in self.multilabel_label_names:
            prob = float(tech_probs.get(label, 0.0))
            prob_percent = prob * 100
            clean_tech_name = label.replace('_label', '').replace('_', ' ').capitalize()
            
            if label in active_techniques or prob >= 0.10:
                attributions = tech_explanations.get(label, {})
                shap_tech = attributions.get('shap', [])
                
                # Clean and stitch tokens for this individual vector channel
                evidence_keywords = self._extract_clean_words(shap_tech, top_k=5)
                evidence_str = ", ".join(evidence_keywords) if evidence_keywords else "no prominent evidence"
                
                lines.append(f"- {clean_tech_name} ({prob_percent:.1f}%): {evidence_str}")
            else:
                lines.append(f"- {clean_tech_name} ({prob_percent:.1f}%): no prominent evidence")

        # Save logic stays identical
        if file_path.endswith('.json'):
            file_path = file_path.rsplit('.json', 1)[0] + '.txt'
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
            
        print(f"Successfully compiled clean text explanation summary directly to: {file_path}")
    
        


