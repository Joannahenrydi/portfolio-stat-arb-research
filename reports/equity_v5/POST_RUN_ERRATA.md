# Post-run factual erratum

The v5 protocol says the prior cohort contained 356 names and the stale gate left 353.
Inspection of the already-frozen v2 audit shows that 356 is the count with continuous
adjusted closes, before the v2 identifier-continuity exclusion. Sixteen symbols have a
greater-than-50% adjusted move; because four also lack continuous history, the actual v2
`included` count is 344. The v5 train-only stale gate removes COR, DOW and SNDK, so the v5
run contains 341 names.

This correction was written after the v5 selection result. It changes no data row, signal,
candidate parameter, acceptance gate or result. The frozen protocol file and its recorded
SHA-256 remain unchanged as evidence of the original pre-run specification.
