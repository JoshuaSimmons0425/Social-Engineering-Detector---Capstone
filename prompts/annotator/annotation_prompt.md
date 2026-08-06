You must return exactly one annotation object for every submitted message.

There are **{{MESSAGE_COUNT}}** messages in this request.

### Output Requirements

* Return exactly **{{MESSAGE_COUNT}}** objects in the `annotations` array.
* Preserve every `message_id` exactly as supplied.
* Do not combine messages.
* Do not omit messages, even when a message is benign or contains no techniques.
* Return each `message_id` exactly once.
* Return valid JSON only. Do not include markdown, explanations, or additional text.

---

## Task

Annotate each message according to the taxonomy below.

This is a **multi-label classification task**. Every attribute is **independent** and should be evaluated separately. A message may contain **zero, one, or multiple attributes simultaneously**. Do **not** attempt to identify only the single most prominent attribute.

Label an attribute **only when there is clear evidence in the text**.

### Taxonomy

#### Urgency

**Definition**

The message attempts to create pressure by implying that immediate action is required, reducing the recipient's opportunity to think critically or verify the request.

**Things to look out for**

* Deadlines or countdowns ("within 24 hours", "today only")
* Words implying immediacy such as "urgent", "immediately", "now", "as soon as possible", or "right away"
* Claims that delaying will cause negative consequences
* Requests for rapid responses or immediate payment
* Time-sensitive opportunities or threats
* Language discouraging hesitation or verification

---

#### Authority

**Definition**

The message attempts to gain compliance by presenting itself as coming from an authoritative, official, or trusted individual or organisation.

**Things to look out for**

* Claims to represent banks, government agencies, employers, IT departments, police, schools, or universities
* Official-sounding language
* References to policies, regulations, compliance, or legal obligations
* Titles such as Manager, Director, CEO, Administrator, or Support Team
* Official branding, signatures, or credentials intended to establish legitimacy
* Instructions presented as mandatory because of the sender's position

---

#### Fear

**Definition**

The message attempts to motivate action by creating concern about loss, punishment, danger, or other negative consequences.

**Things to look out for**

* Threats of account suspension, fines, legal action, security breaches, or data loss
* Claims that something bad has already occurred
* Warnings requiring immediate action to avoid harm
* Emotionally charged language intended to create anxiety or panic
* Severe or irreversible consequences

---

#### Reciprocity

**Definition**

The message attempts to influence the recipient by offering a favour, gift, reward, or assistance that creates an expectation of reciprocation.

**Things to look out for**

* Gifts, rewards, bonuses, or discounts
* Offers of assistance followed by a request
* Language implying obligation or gratitude
* Exclusive benefits requiring an action
* Statements suggesting the sender has already done something for the recipient

---

#### Curiosity

**Definition**

The message attempts to encourage interaction by exploiting curiosity or intentionally withholding information.

**Things to look out for**

* Vague or incomplete statements
* "You won't believe..."
* "Someone mentioned you..."
* Unexplained links or attachments
* Promises of surprising or exclusive information
* Information gaps intended to encourage clicking or replying

---

#### Pretexting

**Definition**

The sender invents or adopts a believable identity or scenario to justify requesting information or persuading the recipient to take action.

**Things to look out for**

* Fabricated stories or situations
* Impersonation of colleagues, family members, customers, IT support, banks, or other trusted parties
* Explanations for bypassing normal procedures
* Identity claims intended to establish legitimacy before making a request

---

#### Promotional

**Definition**

The message primarily advertises or promotes a product, service, event, offer, or opportunity.

**Things to look out for**

* Marketing or advertising language
* Discounts or special offers
* Product launches
* Promotional events
* Limited-time deals
* Calls to purchase, subscribe, register, or sign up

**Note:** Promotional messages are not necessarily malicious.

---

#### Transactional

**Definition**

The message relates to an existing transaction, account activity, service interaction, or operational event rather than advertising.

**Things to look out for**

* Order confirmations
* Shipping notifications
* Receipts
* Payment confirmations
* Login alerts
* Password reset messages
* Appointment confirmations
* Account activity notifications
* Service updates

**Note:** Transactional messages may also contain other techniques such as urgency or authority.

---

#### Reminder

**Definition**

The message reminds the recipient about an existing obligation, scheduled event, pending task, or previously communicated information.

**Things to look out for**

* "This is a reminder..."
* Upcoming appointments
* Payment reminders
* Renewal notices
* Due dates
* Event reminders
* Follow-up notifications
* References to previous communication

---

#### Personal

**Definition**

The message attempts to establish familiarity or trust by using personal relationships or personalised information.

**Things to look out for**

* References to family or friends
* Previous conversations
* Shared experiences
* Workplace or school relationships
* Personal interests
* Individualised details
* Conversational language intended to build rapport

**Note:** Simply addressing the recipient by name is **not sufficient**. The message should meaningfully leverage personal familiarity or personalised information.

---

## Annotation Rules

1. Analyse only the supplied message text.
2. Evaluate every attribute independently.
3. A message may contain multiple attributes.
4. A message may contain none of the attributes.
5. Do not infer information that is not explicitly or strongly supported by the text.
6. Base every decision solely on the message content.
7. Do not assume malicious intent.
8. Do not classify whether the message is malicious or benign. Your task is only to identify the communication and social engineering attributes present.

For every attribute where `"label": true`:

* provide a confidence score between **0.00** and **1.00**,
* provide the exact evidence phrase(s) copied from the message,
* provide a concise justification explaining why the evidence supports the attribute.

For every attribute where `"label": false`:

* include only `"label": false`,
* omit confidence, evidence, and justification.

---

## Messages to Annotate

{{MESSAGES}}
