Annotate the following message according to the provided taxonomy.

Requirements:

1. Analyse only the supplied text.
2. A message may contain multiple security attributes.
3. A message may contain multiple communication attributes.
4. Do not infer information that is not supported by the text.
5. Return valid JSON only.
6. For every detected attribute:
   - provide a confidence score between 0.00 and 1.00,
   - provide the exact evidence phrase(s),
   - provide a concise justification.
7. If an attribute is not present, set "label": false and omit the remaining fields for that attribute.
8. Do not classify whether the message is malicious or benign. Your task is only to identify the communication and social engineering attributes present in the message.

Message to annotate:

{{MESSAGES}}