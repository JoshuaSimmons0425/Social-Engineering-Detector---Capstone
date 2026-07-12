Return JSON only.

Do not include markdown, code fences, explanations, comments, or any additional text.

Always return every attribute shown below.

For each attribute, include:

"label": true or false
"confidence": a floating-point value between 0.00 and 1.00

Confidence should represent how strongly the message exhibits the attribute itself—not whether the message is malicious.

If an attribute is not present, set:

"label": false
"confidence": 0.00

Return the JSON in the following format:

{
  "security_attributes": {
    "authority": {
      "label": false,
      "confidence": 0.00
    },
    "urgency": {
      "label": false,
      "confidence": 0.00
    },
    "fear": {
      "label": false,
      "confidence": 0.00
    },
    "credential_request": {
      "label": false,
      "confidence": 0.00
    },
    "financial_request": {
      "label": false,
      "confidence": 0.00
    },
    "pretexting": {
      "label": false,
      "confidence": 0.00
    }
  },
  "communication_attributes": {
    "promotional": {
      "label": false,
      "confidence": 0.00
    },
    "transactional": {
      "label": false,
      "confidence": 0.00
    },
    "reminder": {
      "label": false,
      "confidence": 0.00
    },
    "personal": {
      "label": false,
      "confidence": 0.00
    }
  }
}

Your response must consist only of this JSON object and nothing else.