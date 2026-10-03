# VizDoom reproduction audit — 2026-10-02

Branch: `feat/vizdoom`. Hardware: RTX 4070 Ti, Ubuntu under WSL2.

## GPU and training performance

WSL initially could not access the GPU. The authorized WSL restart restored CUDA
access; the three monitoring containers were restarted afterward. The GPU
preflight now fails on CPU fallback and checks synchronized matrix multiplication
and convolution gradients.

The locked environment uses Python 3.12, JAX 0.11.2 with CUDA 13, Equinox 0.13.8,
and Optax 0.2.8. VizDoom 1.2.4 and Gymnasium 1.2.2 remain fixed to avoid changing
the scenario during this comparison. The existing CarRacing checkpoints load and
produce finite RNN outputs on CUDA; CarRacing was not retrained.

On a synchronized synthetic RNN optimizer benchmark (batch 32, sequence 256,
64 latent dimensions, 512 hidden units), the old stack took 57.4 ms per update;
the final locked stack took 26.4 ms, approximately **2.17× faster**. This measures
optimizer throughput, not complete training including disk I/O. Reports:
`artifacts/benchmark_cuda12.json` and `artifacts/benchmark_cuda13_locked_idle.json`.

The dream simulator uses a `while_loop` and stops when every rollout in a batch
has died. Raising the candidate batch from 8 to 64 reduced observed generation
time from roughly 4 seconds to 1 second with concurrent searches. These timings
include GPU contention and are not a controlled speedup measurement.

## Why the previous Doom result was unreliable

1. Stored `rewards[t]` and `dones[t]` describe the outcome of `actions[t]` from
   `obs[t]`. Training assigned outcomes to the preceding action and omitted the
   actual fatal action. Transition targets now retain all actions and
   mask only missing next observations out of the latent loss.
2. The rare death class was unweighted. Summing latent likelihood across 64 axes
   also dominated that head. Doom now averages the latent loss by dimension and
   weights positive death labels by 10, as in the reference implementation.
3. The RNN used VAE means, whereas the reference samples the VAE posterior.
   Median posterior standard deviation was about 0.954, versus about 0.085 for
   the original RNN's full mixture predictive distribution on the paired audit.
   Posterior sampling is now the default; consistent mean-latent training and
   inference remains a valid extension to compare experimentally.
4. The original Doom reference uses five mixtures independently for each latent
   dimension. This implementation previously shared one mixture choice across
   the entire vector. New Doom checkpoints use the reference factorization;
   legacy checkpoints remain readable.
5. Temperature previously scaled standard deviation by tau and left mixture
   weights unchanged. The reference divides mixture logits by tau and scales
   standard deviation by square root of tau.
6. Dreams could start midway through an episode with zero memory. They now start
   from recorded episode starts. Controllers see both cell and hidden state,
   `[z,c,h]`, rather than just `[z,h]`.
7. A learned reward near 1 was interpreted as survival confidence. Doom's unit
   reward cannot estimate death probability. Evolution now counts survival steps
   directly, including the terminal transition. The old checkpoint's dream
   score of 2104.37 exceeded the 2100-step maximum.
8. Real evaluation added 50 forced no-op steps absent from dream training; Doom
   now acts immediately. Actual game seeds are forwarded to VizDoom, and a
   2100-step wrapper limit distinguishes timeout from death.
9. Vector collection's default next-step autoreset could insert fake transitions.
   It now uses same-step autoreset and spawned workers. Processed data now use
   stable source IDs and a VAE fingerprint instead of shuffled numeric IDs.

These changes address concrete mismatches. They do not establish that free-running
model dynamics are accurate merely because reconstruction or one-step MSE looks
good.

## Data and world-model checks

The existing processed set contains **15,404 episodes and 3,703,860 frames**.
An exhaustive content check found three extra duplicate episodes, with **zero
duplicate groups crossing the current train/validation split**. Episode lengths
and internal termination labels were consistent. Re-encoding first, middle and
last observations from 64 matched raw episodes agreed with the current VAE.
That sample supports encoding consistency; it does not verify every encoding.

The corrected RNN trained for 20 epochs, seed 42, batch 32, learning rate 0.001,
global gradient clipping at 1, and a 2100-step maximum. The episode split was
14,634 training / 770 validation. Final validation loss was 1.02393.

On the same sampled 256 validation episodes, the original RNN detected **2.34%**
of actual terminal transitions at its death cutoff; the final corrected RNN
detected **72.66%**. The old model's training split is unknown, so this is a
diagnostic comparison, not a held-out generalization claim for the old model.
The corrected model's false-alarm/death scores must still be assessed under
free-running rollouts. The sigmoid of a positively weighted BCE logit is a
detection score, not a calibrated probability.

Reports: `artifacts/doom_series_integrity.json`,
`artifacts/doom_latent_consistency.json`,
`artifacts/doom_audit_original_paired_updated.json`, and
`artifacts/doom_audit_full_final.json`.

## Controller experiments

All new searches use a linear controller, 64 CMA candidates, 16 dreams per
candidate, and 64 fixed validation dreams. Candidates share random streams within
a generation. Checkpoints are selected by dream validation score instead of the
last generation's luckiest rollout. The linear controller has 1089 parameters,
including one bias; the reference controller has 1088 without bias.

An initial controller evolved 100 generations against an intermediate corrected
RNN. Full-model searches warm-started from that controller. Two smaller-batch
searches were intentionally interrupted after saving candidates, then continued
as fresh CMA searches with larger GPU batches. Their history files record the
actual generation counts; the filenames containing `500` are requested budgets,
not evidence that an interrupted stage completed 500 generations.

Four longer searches compare temperatures 1.15 and 1.30 and two death rules:

- **Threshold:** stop when the weighted death logit crosses zero, matching the
  reference simulator.
- **Sampled:** draw death events after subtracting `log(positive_weight)` from
  the weighted logit. This analytically undoes the class-weight odds multiplier;
  it is an extension, and does not guarantee empirical calibration.

Real inference is checked both with posterior samples and with VAE means. The
latter is another extension. Policy choices use 20 validation games with seeds
11000–11019. The final selected policy is evaluated separately on 100 games.

## Measured real-game results

Results below use seeds 20000–20099, a 2100-step cap, and mean ± population standard
deviation. The early corrected snapshot is fixed before the longer searches.

| Policy | Episodes | Mean survival ± SD |
| --- | ---: | ---: |
| Random actions | 100 | 222.58 ± 98.63 |
| Original checkpoint, legacy 50-step warmup | 100 | 247.30 ± 109.81 |
| Original checkpoint, no warmup | 100 | 238.99 ± 114.07 |
| Corrected intermediate model/controller | 100 | 382.53 ± 147.97 |
| Iteration 1 selected: sampled death, tau 1.15, real inference using VAE means | 100 | 505.74 ± 308.09 |

The [World Models paper](https://worldmodels.github.io/) reports 1092 ± 556 real
steps at temperature 1.15. Its solve criterion is mean survival above 750 over
100 episodes. The corrected policies improve this repository's original policy
but do **not** satisfy that criterion. The iteration-1 policy improved matched-seed
mean survival by 266.75 steps (2.12×) over the original without warmup. Its 20-game
validation mean of 701.85 was optimistic; use the separate 100-game result.
The paired bootstrap 95% interval for the improvement is 216.35–322.45 steps
(10,000 resamples, seed 123); this interval describes improvement over the old
policy, not attainment of the paper's target.
Reports: `artifacts/doom_selected_test100.json` and its checkpoint selection
provenance in `checkpoints/VizdoomTakeCover-v0/reproduction_selected/`.

This is not yet an exact replication: it reuses the repository's VAE, full-frame
preprocessing and mixed historical data, rather than the reference's cropped
images and 10,000 random episodes. Mean-latent inference and sampled termination
must be reported as extensions if selected.

## Failure-data refinement

The stronger policy still visits states the model handles poorly. On 128 episodes
from an early 213-episode sample of its new failure data, death recall was 28.13%
with posterior inputs and 28.91% with mean inputs, versus 72.66% on the older
held-out data. Low one-step MSE did not establish reliable transfer. All 213
encoded episodes were unique and their transition labels were consistent.

The next iteration collects 1000 episodes from this policy, retains separate
new failure holdouts plus the original 770 validation episodes, and mixes the
remaining failures with historical training data for fine-tuning. Processed
episodes now store source-content fingerprints and skip unchanged encodings on
reruns. The collector also fuses policy inference and memory update into one JIT
call; a 12-episode collection smoke check verified episode and terminal labels.

The completed collection contains 1000 unique episodes and 512,770 frames, with
mean survival 512.77 ± 319.30. Fine-tuning used 2560 historical training episodes
and 900 new episodes; validation retained the old 770 and a separate 100 new
episodes. After 12 epochs at learning rate 0.0003, terminal recall on those 100
new holdouts increased from 35% to 49% with posterior training, or 55% with mean
training. These teacher-forced diagnostics do not establish dream accuracy.

Each refined model received two 500-generation controller searches, with threshold
and sampled termination at tau 1.15. Seven policies, including the incumbent and
posterior/mean inference variants, were compared on seeds 12000–12019:

| Model / policy | Real validation mean (20 games) |
| --- | ---: |
| Posterior model, threshold, posterior inference | 800.70 |
| Posterior model, threshold, mean inference | 595.90 |
| Posterior model, sampled, posterior inference | 765.55 |
| Posterior model, sampled, mean inference | **832.45** |
| Mean model, threshold | 766.05 |
| Mean model, sampled | 694.50 |
| Iteration-1 incumbent | 392.50 |

The winner was frozen in `reproduction_refined_selected/` before a separate
100-game test using seeds 30000–30099:

| Policy | Mean survival ± SD (100 games) |
| --- | ---: |
| Random actions | 220.21 ± 104.18 |
| Original checkpoint, no warmup | 227.57 ± 111.85 |
| Refined posterior model, sampled-death dreams, mean real inference | **840.06 ± 524.48** |

The refined policy meets the paper's reported **mean >750 over 100 episodes**
solve criterion on this fresh test. It improves the matched original policy by
612.49 steps (3.69×); paired bootstrap 95% interval 519.50–712.73 steps. Its
mean's bootstrap interval is 741.16–946.20, so this single test does not establish
that the population mean exceeds 750. It also remains below the paper's reported
1092-step mean. No policy was selected or revised using these test outcomes.
The reused VAE, mixed data, full-frame preprocessing, sampled termination and
mean-latent inference make this an extension that reaches the solve criterion,
not an exact reproduction of the paper's complete protocol.

Reports: `artifacts/doom_refined_selected_test100.json`,
`artifacts/doom_original_matched30000.json`,
`artifacts/doom_random_matched30000.json`, and
`artifacts/doom_refinement_paired_comparison.json`. The frozen policy is runnable:

```bash
uv run python test_agent.py --env VizdoomTakeCover-v0 \
  --checkpoint_dir checkpoints/VizdoomTakeCover-v0/reproduction_refined_selected \
  --episodes 5 --no_video
```

The current WSL limits are 8 GB RAM / 8 CPU threads on a 32 GB / 16-thread host.
There was no active swapping and CPU capacity remained available during the
measured runs. 16 GB / 12 threads would provide headroom for larger VAE/data
jobs, but would require a restart; no WSL configuration changes were made.

## Longer training budget

The reference RNN trainer uses 400 epochs, batch 100, packed 500-step sequences,
carried LSTM memory, per-value gradient clipping at 1, and a decaying learning
rate. The shorter runs above therefore leave a substantial training-budget
difference. `train_rnn_packed.py` implements those training controls with explicit
episode resets, no latent targets across resets, physical death labels distinct
from timeouts, and separate complete-episode validation. Fragments beginning
mid-episode after stream partitioning receive no loss until the next true start.
The shuffled tail shorter than one batch is omitted each epoch. Posterior samples
are shared across adjacent chunk boundaries.

A two-epoch GPU pilot (1000 episodes, 485,587 frames) passed with finite losses;
the second epoch took 1.56 seconds of training plus validation. The longer run
uses all 14,631 unique historical training episodes and 900 new failures, preserving
the original 770 and new 100 validation episodes. It starts from the corrected
20-epoch model and Adam moments, then trains for 400 additional packed epochs.
This is continued training with the reference budget controls, not an exact
from-scratch reference replication. It saves the best held-out model separately
from the final model and immutable snapshots every 100 epochs.

Run status and arguments: `artifacts/doom_packed_long.status`,
`artifacts/doom_packed_long.log`, `artifacts/doom_packed_manifest.json`, and
`checkpoints/VizdoomTakeCover-v0/reproduction_packed_long/rnn.eqx.json`.

## Reproducing and inspecting runs

```bash
uv sync --python 3.12
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run python scripts/tools/check_gpu.py

uv run python train_rnn.py --env VizdoomTakeCover-v0 --epochs 20 --batch_size 32 \
  --output checkpoints/VizdoomTakeCover-v0/new_experiment/rnn.eqx
# Copy the matching VAE into new_experiment before controller evaluation.
uv run python scripts/tools/audit_doom.py --heldout \
  --rnn checkpoints/VizdoomTakeCover-v0/new_experiment/rnn.eqx
uv run python scripts/tools/audit_series_integrity.py \
  --manifest checkpoints/VizdoomTakeCover-v0/new_experiment/rnn.eqx.json

uv run python train_dream.py --env VizdoomTakeCover-v0 --strategy jax_cma \
  --checkpoint_dir checkpoints/VizdoomTakeCover-v0/new_experiment \
  --pop_size 64 --rollouts 16 --mini_batch_size 64 --generations 500 \
  --temperature 1.15 --seed 43
uv run python scripts/tools/evaluate_doom.py \
  --checkpoint-dir checkpoints/VizdoomTakeCover-v0/new_experiment \
  --episodes 100 --seed 30000 --output artifacts/new_experiment_test100.json
```

Use a fresh output name for each controller search. Exact completed-run arguments,
model fingerprints and histories are beside the checkpoints in
`checkpoints/VizdoomTakeCover-v0/reproduction_full/`. Keep the RNN JSON sidecar
with its checkpoint. Original model files were retained.

Validation: 13 regression tests passed, including packed-stream resets,
terminal-label alignment,
posterior temperature, death weighting, controller memory and terminal-step
survival scoring. All three optimizer paths passed short smoke runs; sequential
and spawned parallel inference were exercised. Ruff and whitespace checks passed.
The frozen refined policy also ran through `test_agent.py` at seed 30099 and
matched its separate evaluator's 250-step result.

Reference code: [Doom simulator](https://github.com/hardmaru/WorldModelsExperiments/blob/master/doomrnn/doomrnn.py),
[real Doom wrapper](https://github.com/hardmaru/WorldModelsExperiments/blob/master/doomrnn/doomreal.py),
[RNN training](https://github.com/hardmaru/WorldModelsExperiments/blob/master/doomrnn/rnn_train.py).
