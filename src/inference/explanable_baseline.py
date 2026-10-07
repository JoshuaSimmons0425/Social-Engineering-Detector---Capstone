import os
import re
import numpy as np
import scipy as sp
import pandas as pd
import shap
from lime.lime_text import LimeTextExplainer
from src.datasets.dataengine import DataEngine

class ExplainableBaseline:
    def __init__(self, binary_model, multi_label_model, estimators, binary_calibrators, multilabel_calibrators, vectorizer, decision_threshold, input_text, multi_label_names):

        self.binary_model = binary_model
        self.multi_label_model = multi_label_model
        self.estimators = estimators
        self.binary_calibrators = binary_calibrators
        self.multilabel_calibrators = multilabel_calibrators
        self.vectorizer = vectorizer
        self.decision_threshold = decision_threshold
        self.multi_label_names = multi_label_names

        self.binary_class_names = ['benign', 'malicious']
        self.multi_label_class_names = multi_label_names

        self.binary_information = {}
        self.multilabel_information = {}
        self.overall_information = {}

        self.clean_text = input_text

    # def clean_input_text(self):

    #     engine = DataEngine()
    #     temp_df = pd.DataFrame([self.input_text], columns=['text'])

    #     temp_df = engine.mask_money(temp_df, 'text')
    #     temp_df = engine.anonymize_data(temp_df, 'text')

    #     self.clean_text = temp_df['text'].iloc[0]

    def _get_binary_platt_params(self):
        """Helper to safely resolve calibration parameters from objects or dictionaries."""
        if hasattr(self.binary_calibrators, 'A') and hasattr(self.binary_calibrators, 'B'):
            a_val = self.binary_calibrators.A.item() if hasattr(self.binary_calibrators.A, 'item') else float(self.binary_calibrators.A)
            b_val = self.binary_calibrators.B.item() if hasattr(self.binary_calibrators.B, 'item') else float(self.binary_calibrators.B)
            return a_val, b_val
        if isinstance(self.binary_calibrators, dict):
            return self.binary_calibrators.get('A', 1.0), self.binary_calibrators.get('B', 0.0)
        return 1.0, 0.0

    def _get_binary_logits(self, X):
        # Handles models that implement decision_function vs predict_proba log-odds
        if hasattr(self.binary_model, 'decision_function'):
            return self.binary_model.decision_function(X).flatten()
        else:
            probs = self.binary_model.predict_proba(X)[:, 1]
            return np.log(probs / (1 - probs + 1e-15))

    def predict_binary(self):
        if not self.clean_text:
            raise ValueError("Input text has not been cleaned. Call clean_input_text() first.")

        # Transform raw text string context to sparse vector array matrix representation
        X = self.vectorizer.transform([self.clean_text])
        raw_logits = self._get_binary_logits(X)
        
        A, B = self._get_binary_platt_params()
        scaled_logits = raw_logits * A + B
        
        malicious_prob = float(1 / (1 + np.exp(-scaled_logits))[0])
        benign_prob = 1.0 - malicious_prob
        
        pred_idx = 1 if malicious_prob >= self.decision_threshold else 0

        self.binary_information['prediction'] = {
            'predicted_class': self.binary_class_names[pred_idx],
            'confidence': float(malicious_prob if pred_idx == 1 else benign_prob),
            'probabilities': {
                'benign': benign_prob,
                'malicious': malicious_prob
            }
        }

    def predict_multilabel(self):
        if not self.clean_text:
            raise ValueError("Input text has not been cleaned. Call clean_input_text() first.")

        # Transform raw text string to sparse feature vectors
        X = self.vectorizer.transform([self.clean_text])
        
        calibrated_probs = {}
        active_techniques = []

        # Iterate over each independent binary label estimator
        for idx, label in enumerate(self.multi_label_class_names):
            # Safe extraction of the specific classifier corresponding to this index lane
            estimator = self.estimators[idx]
            
            # Pull uncalibrated decision function values out of our linear base classifiers
            if hasattr(estimator, 'decision_function'):
                raw_logit = float(estimator.decision_function(X).flatten()[0])
            else:
                prob_pair = estimator.predict_proba(X).flatten()
                pos_idx = 1 if (True in estimator.classes_ or 1 in estimator.classes_ or 'True' in estimator.classes_) else 0
                true_prob = prob_pair[pos_idx]
                raw_logit = float(np.log(true_prob / (1 - true_prob + 1e-15)))

            # Extract your saved scikit-learn calibrator metrics dictionary
            cal_params = self.multilabel_calibrators.get(idx, self.multilabel_calibrators.get(label, {"A": 1.0, "B": 0.0}))
            
            A = cal_params.get("A", 1.0)
            B = cal_params.get("B", 0.0)

            # Note the positive sign inside exp() to align with CalibratedClassifierCV's parameter definition
            prob = float(1 / (1 + np.exp(A * raw_logit + B)))
            calibrated_probs[label] = prob

            if prob >= 0.10:
                active_techniques.append(label)

        self.multilabel_information['prediction'] = {
            'active_techniques': active_techniques,
            'probabilities': calibrated_probs
        }
    
    def explain_binary(self):
        A, B = self._get_binary_platt_params()

        # Text wrappers to feed on-the-fly vector matrices into LIME and SHAP
        def calibrated_binary_probs(texts):
            X_batch = self.vectorizer.transform(texts)
            logits = self._get_binary_logits(X_batch)
            cal_logits = logits * A + B
            malicious_probs = 1 / (1 + np.exp(-cal_logits))
            return np.vstack([1.0 - malicious_probs, malicious_probs]).T

        def calibrated_binary_logits(texts):
            X_batch = self.vectorizer.transform(texts)
            return (self._get_binary_logits(X_batch) * A + B).reshape(-1, 1)

        # LIME Execution
        lime_explainer = LimeTextExplainer(class_names=self.binary_class_names)
        lime_exp = lime_explainer.explain_instance(self.clean_text, calibrated_binary_probs, num_features=10, labels=[1])

        text_masker = shap.maskers.Text(tokenizer=r"\W+")
        # SHAP Linear/Explainer Integration via Background Text Kernel
        # Since TF-IDF is treated as a text model pipe wrapper, we can pass text inputs into SHAP's Explicit Text Kernel
        shap_explainer = shap.Explainer(calibrated_binary_logits, masker=text_masker) 
        shap_values = shap_explainer([self.clean_text])
        
        # Flatten and isolate output array records safely
        if len(shap_values.shape) > 1 and shap_values.shape[-1] > 1:
            shap_slice = shap_values[0, :, 1]
        else:
            shap_slice = shap_values[0, :]

        self.binary_information['explanations'] = {
            'lime': lime_exp.as_list(label=1),
            'shap': [{'token': t, 'val': float(v.item() if hasattr(v, 'item') else v)} for t, v in zip(shap_slice.data, shap_slice.values) if t.strip()]
        }

    
    def explain_multilabel(self):
        probs_dict = self.multilabel_information.get('prediction', {}).get('probabilities', {})
        if not probs_dict:
            self.predict_multilabel()
            probs_dict = self.multilabel_information['prediction']['probabilities']

        active_labels = [label for label, prob in probs_dict.items() if prob >= 0.10]
        below_threshold_labels = [label for label, prob in probs_dict.items() if prob < 0.10]

        # Black-box pipeline mapping sub-batches straight to multi-class target indexes
        def calibrated_multilabel_logits(texts):
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            elif not isinstance(texts, list):
                texts = [str(texts)]
            else:
                texts = [str(t) for t in texts]

            X_batch = self.vectorizer.transform(texts)
            # Pre-allocate array matrix structure matching your shapes: (batch_size, 10)
            logits = np.zeros((len(texts), len(self.multi_label_class_names)))
                
            for idx, label in enumerate(self.multi_label_class_names):
                estimator = self.estimators[idx]
                
                if hasattr(estimator, 'decision_function'):
                    label_logits = estimator.decision_function(X_batch).flatten()
                else:
                    prob_pair = estimator.predict_proba(X_batch)[:, 1]
                    label_logits = np.log(prob_pair / (1 - prob_pair + 1e-15))
                
                cal_params = self.multilabel_calibrators.get(idx, self.multilabel_calibrators.get(label, {"A": 1.0, "B": 0.0}))
                A = cal_params.get("A", 1.0)
                B = cal_params.get("B", 0.0)
                
                # We calculate log-odds that match SHAP's directional visualization mapping.
                # To match 1 / (1 + exp(A*x + B)) natively as an increasing probability function,
                # the directional logit model passed to SHAP is evaluated as: -(A * x + B)
                logits[:, idx] = -(A * label_logits + B)
            return logits

        text_masker = shap.maskers.Text(tokenizer=r"\W+")
        # Compute combined structural SHAP valuations once across feature subsets
        shap_explainer = shap.Explainer(calibrated_multilabel_logits, masker=text_masker)
        shap_values = shap_explainer([self.clean_text])

        explanations_per_active_label = {}
        for target_label in active_labels:
            target_idx = self.multi_label_class_names.index(target_label)
            
            # Safely unpack 3D/2D array slices generated by SHAP's text mapping kernel
            if len(shap_values.shape) == 3:
                shap_slice_data = shap_values[0, :, target_idx].data
                shap_slice_vals = shap_values[0, :, target_idx].values
            else:
                shap_slice_data = shap_values[0].data
                shap_slice_vals = shap_values[0].values[:, target_idx] if len(shap_values[0].values.shape) > 1 else shap_values[0].values

            label_shap = [{
                'token': t, 
                'val': float(v.item() if hasattr(v, 'item') else v)
            } for t, v in zip(shap_slice_data, shap_slice_vals) if t.strip()]

            explanations_per_active_label[target_label] = {
                'shap': label_shap
            }

        self.multilabel_information['explanations'] = {
            'active_labels_explained': active_labels,
            'other_labels_below_10_percent': below_threshold_labels,
            'attributions': explanations_per_active_label
        }

    def run_explanations(self):
        """Pipeline orchestration runner."""
        # self.clean_input_text()
        self.predict_binary()
        self.predict_multilabel()
        self.explain_binary()
        self.explain_multilabel()

    def _extract_clean_words(self, scored_items, top_k=5):
        """Standardised text denoising method to match the BERT text formatting outputs."""
        STOP_WORDS = {
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'of', 'at', 'by', 
            'from', 'for', 'in', 'on', 'to', 'with', 'is', 'was', 'were', 'be', 
            'been', 'this', 'that', 'these', 'those', 'it', 'its', 'you', 'your', 'i'
        }
        if len(scored_items) == 0: 
            return []
        
        sorted_items = sorted(scored_items, key=lambda x: x.get('val', 0.0), reverse=True)
        raw_tokens = [item['token'] for item in sorted_items if item.get('val', 0.0) > 0.001]

        valid_words = []
        reference_text_lower = self.clean_text.lower()
        source_tokens = re.findall(r'<\w+>|\b\w+\b', reference_text_lower)

        for token in raw_tokens:
            clean_token = token.replace('##', '').strip().lower()
            if not clean_token or clean_token in STOP_WORDS or clean_token in ['<', '>', '[', ']']:
                continue
                
            matched_full_word = None
            for source_tok in source_tokens:
                if clean_token in source_tok:
                    matched_full_word = source_tok
                    break
            
            target_word = matched_full_word if matched_full_word else clean_token
            if target_word.startswith('<') and target_word.endswith('>'):
                target_word = target_word.upper()
            
            if target_word not in valid_words and target_word.lower() not in STOP_WORDS:
                valid_words.append(target_word)
                if len(valid_words) >= top_k:
                    break
        return valid_words

    def save_explanations(self, file_path):
        """Generates the identical scannable plaintext schema (.txt) output to support the experiment."""
        binary_probs = self.binary_information.get('prediction', {}).get('probabilities', {})
        malicious_probability = float(binary_probs.get('malicious', 0.0))
        predicted_class = str(self.binary_information.get('prediction', {}).get('predicted_class', 'benign')).capitalize()
        
        if malicious_probability >= 0.90: 
            risk_level = "Critical"
        elif malicious_probability >= 0.75: 
            risk_level = "High"
        elif malicious_probability >= 0.35: 
            risk_level = "Medium"
        else: 
            risk_level = "Low"

        shap_binary = self.binary_information.get('explanations', {}).get('shap', [])
        clean_binary_list = self._extract_clean_words(shap_binary, top_k=5)
        key_attributes_str = ", ".join(clean_binary_list) if clean_binary_list else "no prominent evidence"

        lines = [
            "Classification:",
            f"- Malicious probability: {malicious_probability:.2f}",
            f"- Classifier risk level: {risk_level}",
            f"- Key Attributes: {key_attributes_str}",
            f"- Overall Classification: {predicted_class}",
            "",
            "Technique associations:"
        ]

        multilabel_preds = self.multilabel_information.get('prediction', {})
        tech_probs = multilabel_preds.get('probabilities', {})
        active_techniques = self.multilabel_information.get('explanations', {}).get('active_labels_explained', [])
        tech_explanations = self.multilabel_information.get('explanations', {}).get('attributions', {})

        tech_list_data = []
        for label in self.multi_label_names:
            prob = float(tech_probs.get(label, 0.0))
            clean_tech_name = label.replace('_label', '').replace('_', ' ').capitalize()
            tech_list_data.append((clean_tech_name, label, prob))

        tech_list_data_sorted = sorted(tech_list_data, key=lambda x: x[2], reverse=True)

        for clean_name, label, prob in tech_list_data_sorted:
            lines.append(f"- {clean_name}: {prob * 100:.1f}%")

        lines.append("")
        lines.append("XAI features:")

        for clean_name, label, prob in tech_list_data_sorted:
            if label in active_techniques or prob >= 0.10:
                attributions = tech_explanations.get(label, {})
                shap_tech = attributions.get('shap', [])
                evidence_keywords = self._extract_clean_words(shap_tech, top_k=5)
                evidence_str = ", ".join(evidence_keywords) if evidence_keywords else "no prominent features"
                lines.append(f"- {clean_name}: {evidence_str}")
            else:
                lines.append(f"- {clean_name}: no prominent features")

        if file_path.endswith('.json'):
            file_path = file_path.rsplit('.json', 1)[0] + '.txt'
        elif not file_path.endswith('.txt'):
            file_path = file_path + '.txt'
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
            
        print(f"Successfully compiled baseline plaintext summary directly to: {file_path}")

    
        

    


        