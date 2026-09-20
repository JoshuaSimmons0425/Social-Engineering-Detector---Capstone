Assess this message for social-engineering risk.

MESSAGE:
{{message}}

MODEL-DERIVED EVIDENCE:
{{evidence}}

If no evidence is provided, assess the message independently.

Determine:

* What the sender appears to want.
* What action the recipient is being encouraged to take.
* Relevant social-engineering techniques and risk indicators.
* Important contextual factors and potential consequences.
* Overall risk level.
* Why that risk level is appropriate.
* What the recipient should reasonably do.

Use model-derived evidence only when relevant and consistent with the message. Do not treat it as ground truth.

Return the assessment using exactly this structure:

## Risk Assessment

**Level:** [Low / Moderate / High / Critical]

## Analysis

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

## Justification

...

## Preliminary Guidance

* ...
* ...

Keep the assessment concise but sufficiently specific to justify the conclusion.
