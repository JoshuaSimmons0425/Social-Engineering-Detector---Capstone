You must return exactly one annotation object for every submitted message.

There are {{MESSAGE_COUNT}} messages in this request.

Requirements:
- Return exactly {{MESSAGE_COUNT}} objects in the annotations array.
- Preserve every message_id exactly as supplied.
- Do not combine messages.
- Do not omit messages, even when a message is benign or contains no techniques.
- Return each message_id exactly once.

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