# CMPSC 448 — Midterm Project: Who Wrote It?

Identifying which LLM family generated a response from `(LLM_name, LLM_input, LLM_output)` data,
using a CNN and an RNN (LSTM) classifier.

**Deadline:** Oct 11, 2026 11:59pm. **Submission:** this repo's URL, containing the code and the report PDF
(with team leader and member names).

## Research questions

- **RQ1** (required): Classify the source LLM from the output alone, using a CNN and an RNN.
- **RQ2** (required): Compare input only, output only, and input + output.
- **RQ3** (+5): Do the fingerprints generalize across tasks or domains?
- **RQ4** (+5): Which characteristics distinguish the LLMs (length, vocabulary, formatting, ablations)?

## Layout

```
src/       code
data/      processed dataset (raw downloads in data/raw/, gitignored)
results/   metrics
figures/   plots used in the report
report/    project report: report.pdf (built from report.md by build.sh)
```
