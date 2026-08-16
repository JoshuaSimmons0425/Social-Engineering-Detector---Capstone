You must return exactly one valid JSON annotation object for every submitted message ({{MESSAGE_COUNT}} total). Do not combine, omit, or modify message_ids. Include no markdown, explanations, or extra text.

Task & Rules:

This is a multi-label classification task. Evaluate each attribute independently. Messages can have zero, one, or multiple labels.

Label an attribute only if clear, explicit textual evidence exists.

Do not infer missing information or assume malicious intent. Do not judge if a message is benign or malicious.

Masked Entities: The text contains masked placeholders (e.g., <EMAIL_ADDRESS>, <PHONE_NUMBER>, <URL>). Focus heavily on the linguistic structure, tone, syntactic patterns, and formatting cues around these placeholders to identify the techniques.

Attribute Output Schema:

If "label": true****: Provide a confidence score (0.00 to 1.00), the exact evidence phrase(s) from the text, and a concise justification.

If "label": false****: Provide only "label": false (omit confidence, evidence, and justification).

Taxonomy:

Urgency: Creates pressure to bypass critical thinking.

Look for: Countdowns ("within 24 hours"), immediate words ("urgent", "now", "ASAP"), claims that delay causes harm, or rapid payment demands.

Authority: Misuses official or trusted status to gain compliance.

Look for: Claims representing banks, government, police, or IT; titles (CEO, Manager); references to compliance/legal obligations; or mandatory policy phrasing.

Fear: Motivates action via concern over negative consequences.

Look for: Threats of account suspension, legal action, fines, or data loss; claims a breach occurred; and anxiety-inducing, emotionally charged warnings.

Reciprocity: Creates a sense of obligation by offering something first.

Look for: Gifts, rewards, vouchers, exclusive discounts, unexpected assistance, or language implying the sender did a favour that requires a return action.

Curiosity: Intentionally withholds details to force interaction.

Look for: Clickbait lines ("you won't believe"), unexplained links/attachments, vague hints ("someone mentioned you"), or intentional information gaps.

Pretexting: Fabricates a scenario or adopts a false identity to justify a request.

Look for: Impersonating colleagues, family, or IT support; detailed fake stories; or logical excuses for bypassing standard security procedures.

Promotional: Advertises a product, service, event, or market opportunity (not inherently malicious).

Look for: Marketing jargon, sales deals, limited-time commercial offers, product launches, or calls to subscribe, purchase, or sign up.

Transactional: Relates to operational, account-based, or non-advertising activity.

Look for: Order receipts, shipping updates, login alerts, password resets, or appointment confirmations. Note: Can co-occur with urgency/authority.

Reminder: Nudges the recipient about an existing timeline or obligation.

Look for: Phrases like "this is a reminder", upcoming due dates, renewal alerts, follow-up notifications, or references to past communication.

Personal: Uses familiarity or individualised data to build trust.

Look for: References to shared history, workplace dynamics, mutual friends, or deep conversational rapport. Note: Simply using the recipient's name is insufficient.

Messages to Annotate

{{MESSAGES}}