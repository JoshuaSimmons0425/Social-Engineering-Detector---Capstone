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
    def __init__(self, binary_model, multilabel_model, binary_calibrators, multilabel_calibrators, tokenizer, input_text, device='cpu'):
        self.binary_model = binary_model
        self.multilabel_model = multilabel_model
        self.binary_calibrators = binary_calibrators
        self.multilabel_calibrators = multilabel_calibrators
        self.tokenizer = tokenizer
        self.input_text = input_text
        self.device = device

        self.binary_model.to(self.device).eval()
        self.multilabel_model.to(self.device).eval()

        self.binary_class_names = ['benign', 'malicious']
        self.multilabel_label_names = [
                'urgency_label', 
                'fear_label', 
                'authority_label', 
                'reciprocity_label', 
                'curiosity_label', 
                'pretexting_label', 
                'promotional_label',
                'transactional_label',
                'reminder_label',
                'personal_label'               
            ]

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

    def predict_binary(self):
        assert self.processed_tensors is not None, "Input has not been preprocessed. Call preprocess_input() first."

        tensors = {k: v.to(self.device) for k, v in self.processed_tensors.items()}
        with torch.no_grad():
            output = self.binary_model(**tensors)
            raw_logits = output.logits

        # Apply Platt Scaling calibration: P = sigmoid(A * logit + B)
        # Assuming binary outputs map to index 1 as 'malicious'
        calibrated_logits = raw_logits * self.binary_calibrators.get('A', 1.0) + self.binary_calibrators.get('B', 0.0)
        
        # Softmax over the calibrated pair
        exp_logits = np.exp(calibrated_logits - np.max(calibrated_logits))
        probabilities = exp_logits / exp_logits.sum()
        
        pred_idx = int(np.argmax(probabilities))
        self.binary_information['prediction'] = {
            'predicted_class': self.binary_class_names[pred_idx],
            'confidence': float(probabilities[pred_idx]),
            'probabilities': {self.binary_class_names[i]: float(p) for i, p in enumerate(probabilities)}
        }
        return self.binary_information['prediction']

    def predict_multilabel(self):
        tensors = {k: v.to(self.device) for k, v in self.processed_tensors.items()}
        with torch.no_grad():
            outputs = self.multilabel_model(**tensors)
            raw_logits = outputs.logits.cpu().numpy()[0] # array of shape (10,)
            
        calibrated_probs = {}
        active_techniques = []
        
        # Iterate over all 10 sigmoid classification heads
        for idx, label in enumerate(self.multilabel_label_names):
            cal_params = self.multilabel_calibrators.get(idx, {'A': 1.0, 'B': 0.0})
            cal_logit = raw_logits[idx] * cal_params.get('A', 1.0) + cal_params.get('B', 0.0)
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
        # 1. Pipeline function for LIME/SHAP black-box modeling
        def calibrated_binary_probs(texts):
            enc = self.tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.binary_model(**enc).logits.cpu().numpy()
            cal_logits = logits * self.binary_calibrators.get('A', 1.0) + self.binary_calibrators.get('B', 0.0)
            exp_l = np.exp(cal_logits - np.max(cal_logits, axis=-1, keepdims=True))
            return exp_l / exp_l.sum(axis=-1, keepdims=True)

        def calibrated_binary_logits(texts):
            enc = self.tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.binary_model(**enc).logits.cpu().numpy()
            return logits * self.binary_calibrators.get('A', 1.0) + self.binary_calibrators.get('B', 0.0)

        # Execute LIME
        lime_explainer = LimeTextExplainer(class_names=self.binary_class_names)
        lime_exp = lime_explainer.explain_instance(self.clean_text, calibrated_binary_probs, num_features=10)
        
        # Execute SHAP
        shap_explainer = shap.Explainer(calibrated_binary_logits, self.tokenizer)
        shap_values = shap_explainer([self.clean_text])
        
        # Execute Transformers-Interpret via temporary forward structural patch
        orig_forward = self.binary_model.forward
        def patched_forward(*args, **kwargs):
            out = orig_forward(*args, **kwargs)
            out.logits = (out.logits * self.binary_calibrators.get('A', 1.0)) + self.binary_calibrators.get('B', 0.0)
            return out
        self.binary_model.forward = patched_forward
        
        ti_explainer = SequenceClassificationExplainer(self.binary_model, self.tokenizer)
        ti_attributions = ti_explainer(self.clean_text)
        self.binary_model.forward = orig_forward # Restore original

        # Package explanations safely
        self.binary_information['explanations'] = {
            'lime': lime_exp.as_list(),
            'shap': [{'token': t, 'val': float(v)} for t, v in zip(shap_values[0, :, 1].data, shap_values[0, :, 1].values) if t.strip()],
            'transformers_interpret': ti_attributions[:15]
        }

    def explain_multilabel(self):
        # We target the specific active or maximum triggered technique class index for local attribution extraction
        probs_dict = self.multilabel_information.get('prediction', {}).get('probabilities', {})
        if not probs_dict:
            self.predict_multilabel()
            probs_dict = self.multilabel_information['prediction']['probabilities']
            
        # Target the technique with highest confidence score for the structural trace
        target_label = max(probs_dict, key=probs_dict.get)
        target_idx = self.multilabel_label_names.index(target_label)

        def calibrated_multilabel_logits(texts):
            enc = self.tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.multilabel_model(**enc).logits.cpu().numpy()
            for idx in range(logits.shape[-1]):
                params = self.multilabel_calibrators.get(idx, {'A': 1.0, 'B': 0.0})
                logits[:, idx] = logits[:, idx] * params.get('A', 1.0) + params.get('B', 0.0)
            return logits

        # SHAP explanation for the targeted technique index
        shap_explainer = shap.Explainer(calibrated_multilabel_logits, self.tokenizer)
        shap_values = shap_explainer([self.clean_text])

        # Transformers-Interpret for the target multilabel element index
        orig_forward = self.multilabel_model.forward
        def patched_forward(*args, **kwargs):
            out = orig_forward(*args, **kwargs)
            for idx in range(out.logits.shape[-1]):
                params = self.multilabel_calibrators.get(idx, {'A': 1.0, 'B': 0.0})
                out.logits[:, idx] = out.logits[:, idx] * params.get('A', 1.0) + params.get('B', 0.0)
            return out
        self.multilabel_model.forward = patched_forward
        
        ti_explainer = SequenceClassificationExplainer(self.multilabel_model, self.tokenizer)
        # Direct the explainer to calculate IG specifically toward target_idx position
        ti_attributions = ti_explainer(self.clean_text, class_index=target_idx)
        self.multilabel_model.forward = orig_forward # Restore original

        self.multilabel_information['explanations'] = {
            'target_technique_explained': target_label,
            'shap': [{'token': t, 'val': float(v)} for t, v in zip(shap_values[0, :, target_idx].data, shap_values[0, :, target_idx].values) if t.strip()],
            'transformers_interpret': ti_attributions[:15]
        }

    def save_explanations(self, file_path):
        self.overall_information = {
            'input_text_raw': self.input_text if isinstance(self.input_text, str) else "[Tensor Object]",
            'clean_text_processed': self.clean_text,
            'binary_classification_layer': self.binary_information,'multilabel_techniques_layer': self.multilabel_information}
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.overall_information, f, indent=4, ensure_ascii=False)
        print(f"Successfully serialized explanation logs to: {file_path}")
    
        


