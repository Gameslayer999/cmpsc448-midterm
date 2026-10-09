"""Run every experiment in the report, in priority order. Finished runs are skipped, so this can be
re-launched after an interruption and it picks up where it stopped.

CNN configs use 3 seeds. The LSTM is ~13x slower per epoch on MPS, so it gets one seed and a
6-epoch cap.

Usage: python src/run_all.py
"""
from train import run

CNN_SEEDS = (0, 1, 2)
LSTM = dict(epochs=6, patience=2)

PLAN = (
    # RQ1: output only
    [dict(model="cnn", field="output", seed=s) for s in CNN_SEEDS]
    + [dict(model="lstm", field="output", **LSTM)]
    # RQ2: input only, input + output
    + [dict(model="cnn", field=f, seed=s) for f in ("input", "both") for s in CNN_SEEDS]
    # RQ4: ablations that remove formatting / punctuation
    + [dict(model="cnn", field="output", transform=t, seed=s)
       for t in ("no_markdown", "words_only") for s in CNN_SEEDS]
    # RQ3: train without one domain, test on it
    + [dict(model="cnn", field="output", held_out=d) for d in ("code", "math", "writing")]
    # LSTM counterparts
    + [dict(model="lstm", field=f, **LSTM) for f in ("input", "both")]
    + [dict(model="lstm", field="output", transform=t, **LSTM) for t in ("no_markdown", "words_only")]
    + [dict(model="lstm", field="output", held_out=d, **LSTM) for d in ("code", "math", "writing")]
)

if __name__ == "__main__":
    for i, cfg in enumerate(PLAN, 1):
        print(f"\n=== {i}/{len(PLAN)} {cfg}", flush=True)
        run(**cfg)
