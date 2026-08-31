## What this changes

<!-- One or two sentences. What is different after this merges? -->

## Why

<!-- The problem being solved. Link an issue if there is one. -->

## Retrieval impact

<!--
Required if you touched cleaning, chunking, retrieval, fusion, reranking or
prompting. Paste the `make eval` table from before and after. Delete this
section if your change cannot affect retrieval.
-->

- [ ] Not applicable — this change cannot affect retrieval quality

```
before:

after:
```

## Checklist

- [ ] `make check` passes (lint + tests)
- [ ] New behaviour has a test that fails without this change
- [ ] User-facing errors say what to do next
- [ ] `.env.example` updated if a setting was added
- [ ] No network calls in tests
