# runtime/gating — what the gate weighs and keeps

`gate.{h,cc}` (S7 #153): `GateState` holds three things:

- **The source model's held-out score.** It is measured once, by the first run, and carried through every checkpoint.
- **The best evaluated state** (E9 #92): the persistent segment, its step, its loss and its accuracy.
- **Patience:** the number of evaluations since the best last improved.

It has three transitions:

- `Snapshot` records the current state as the best.
- `Observe` handles one periodic evaluation.
- `RestoreBest` copies the best state back into the arena before merge and commit.

The engine runs the evaluations; `GateState` decides what each one means. The state round-trips through a checkpoint (SEKP v5, and v6 for the train-loss gate's `LossWindows`) field for field; a resume re-measures the two losses in it rather than trusting the file (CR4 #221).

The verdict itself (did the loss improve, did it improve by the margin, did accuracy hold) lives in `TrainReport::improved`, `ImprovedBy` and `AccuracyHeld`. The runner maps it to exit code 0 or 3.
