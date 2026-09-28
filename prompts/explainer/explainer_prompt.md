Assess the following message for social-engineering risk.

MESSAGE:
"""
{{message}}
"""

MODEL-DERIVED EVIDENCE:
"""
{{evidence}}
"""

## Task Instructions
Develop a holistic risk assessment of the message. Consider both the message itself and any supplied model-derived evidence simultaneously. If the "MODEL-DERIVED EVIDENCE" block is empty, assess the input message independently.

Ensure your assessment explicitly addresses:
* What the sender appears to want.
* What action the recipient is being encouraged to take.
* Relevant social-engineering techniques and risk indicators.
* Important contextual factors and uncertainty.
* Potential consequences that are reasonably supported by the text.
* The overall risk level and why it is appropriate.
* What the recipient should reasonably do.

If a specific sub-section is not relevant to the message, state "Not applicable." Do not add speculative information solely to populate a field. Keep the assessment concise but highly specific to the context.

Return the final assessment using exactly this markdown structure:

**## Risk Assessment**

**Level:** [Low / Medium / High / Critical]

**## Analysis**

**Apparent objective:**
...

**Requested action:**
...

**Relevant techniques:**
* ...

**Risk indicators:**
* ...

**Contextual factors:**
* ...

**Potential consequences:**
...

**## Justification**
...

**## Preliminary Guidance**
* ...