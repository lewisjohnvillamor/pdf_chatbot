---
name: Retrieval quality
about: The assistant missed something that was in the documents
labels: retrieval
---

**The question you asked**

**What the documents actually say, and where**
<!-- File and page. A short quote is ideal. -->

**What the assistant returned**
<!-- Paste the answer, and the Sources panel showing which passages were retrieved. -->

**Was the correct passage retrieved but not cited, or not retrieved at all?**
<!-- The Sources panel answers this: expand it and check whether the passage is listed. -->

**Configuration**
```
```

**Can you add it to the gold set?**
<!--
The most useful contribution here is a case in evals/goldset.json:
{"question": "...", "must_contain": ["exact text that must be retrieved"], "note": "why it's hard"}
That turns a report into a regression test.
-->
