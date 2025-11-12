# Developer notes for exercise classifier tuning

The heuristics implemented in `src/exercise_detection` expose all tunable
thresholds in `constants.py`.  Adjustments can be made there without touching
the logic modules:

* **Sampling & smoothing** – `DEFAULT_SAMPLING_RATE`, `SMOOTHING_*`.
* **Segmentation** – knee hysteresis thresholds and bar drop margin.
* **View classifier** – per-feature weights and decision margins.
* **Exercise scores** – gating angles, penalties, and veto multipliers.

The classification pipeline emits a compact `INFO` log per clip summarising the
aggregated metrics and the raw/adjusted scores.  A `DEBUG` log lists the
per-repetition bottoms.  These logs are the quickest way to verify which cues
are dominating a prediction when calibrating the constants.

Run the synthetic unit tests with `pytest -q` to confirm that refactors keep the
expected qualitative behaviour across squat, deadlift, bench, and unknown
scenarios.
