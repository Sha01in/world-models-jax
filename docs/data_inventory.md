# Data Inventory: VizdoomTakeCover-v0

**Total Episodes:** 15,404
**Total Frames:** ~3.5 Million (Estimated)
**Location:** `data/series/VizdoomTakeCover-v0/`

## Dataset Breakdown

Based on project history (`GEMINI.md`) and file indices:

| ID Range | Phase | Type | Count | Description |
| :--- | :--- | :--- | :--- | :--- |
| **0 - 3,400** | Phase 1-19 | **Legacy** | ~3,400 | Mixed data from early experiments. Quality varies. |
| **3,401 - 9,400** | Phase 20 | **Random** | 6,000 | "Brownian Motion" driving. Good for background, bad for edge cases. |
| **9,401 - 12,400** | Phase 21 | **On-Policy (Linear)** | 3,000 | Data from a competent but simple Linear Controller. |
| **12,401 - 15,404** | Phase 24 | **Delusional (MLP)** | ~3,000 | **CRITICAL DATA.** Collected from the "Delusional" agent. Contains the specific "Sim2Real" failure cases (driving into fireballs). |

## Analysis of Current Training (Phase 24)

We previously trained on the **entire dataset** (15,404 episodes).
*   **Problem:** The "Delusional" data (which contains the correction signal) makes up only **~19%** of the dataset.
*   **Result:** The signal is diluted. The RNN minimizes error on the 81% of "easy/random" data and ignores the 19% of "hard/delusional" cases.

## Proposed Strategy (Phase 25)

To fix the Sim2Real gap, we must prioritize the Delusional data.

### Option A: Curated Fine-Tuning (Recommended)
Train *only* on a curated subset:
1.  **All Delusional Data:** Indices 12,401 - 15,404 (3,000 eps).
2.  **Top 10% of Others:** Filter the remaining 12,400 episodes for high scores (>350) to maintain general competency (~1,200 eps).
*   **Total:** ~4,200 Episodes.
*   **Benefit:** High signal-to-noise ratio. Fast training.

### Option B: Weighted Sampling
Train on all 15,404 episodes, but apply a **5x loss weight** to indices 12,401+.
*   **Benefit:** Keeps all diversity.
*   **Drawback:** Slower, requires code changes to `train_rnn.py`.
