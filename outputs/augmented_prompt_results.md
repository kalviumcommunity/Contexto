
# Concept 14 — Context Injection & Prompt Augmentation

## Augmented Prompt

```text
You are a grounded question-answering assistant.

Answer the user's question ONLY using the information provided in the
context below.

Do not use outside knowledge or invent information.

If the context does not contain enough information to answer the question,
clearly say that the provided context is insufficient.

When possible, reference the supporting source using its source marker
such as [1] or [2].

CONTEXT:
[1] account-guide.md
Learners can reset a forgotten password from the account settings page.
Choose Reset password, verify the account email, and follow the link before
it expires.

[2] news-brief.txt
Learners should keep their account email accessible when performing account
recovery.

[3] submission-rubric.md
Submissions should follow the instructions provided in the relevant learner
documentation.

USER QUESTION:
How can a learner reset their password?