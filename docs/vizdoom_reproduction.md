# VizDoom reproduction audit — 2026-10-02

Branch: `feat/vizdoom`. Hardware: RTX 4070 Ti, Ubuntu under WSL2.

**Final status, 2026-10-03:** the reproduction effort is stopped on observed
diminishing returns. Three completed follow-up loops established no reliable
paired improvement; the paper's 1092-step mean remains unmatched. All GPU jobs
have exited, checkpoints are preserved, and no further experiment is queued.
The final measurements and stopping rationale are recorded below.

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

### Early stop and next refinement

The continued packed run was deliberately stopped after a fully saved **epoch
255 of 400**, rather than resumed to its original budget. Its best full-episode
held-out loss was **1.033261 at epoch 1** and never improved. At epoch 255,
training loss was 0.963839 and validation loss 1.106310. Both the best and last
models were frozen with matching VAE and metadata in
`reproduction_packed_stopped_255_best/` and
`reproduction_packed_stopped_255_latest/`. The last model's optimizer count was
verified against global step 29305; optimizer, history and RNG state are retained
for resumption. Stop reason and file hashes: `artifacts/doom_packed_early_stop.json`.

A paired posterior-sampled audit used the same 256 historical holdout episodes
and all 100 previous policy-failure holdouts, with seed 42 for all three models.
At a detection threshold of 0.5:

| World model | Historical death recall | New death recall | New live-frame false death rate |
| --- | ---: | ---: | ---: |
| Packed best, epoch 1 | 77.73% | 46% | 0.208% |
| Packed latest, epoch 255 | 54.69% | 40% | 0.116% |
| Frozen 840-step incumbent | 72.66% | 49% | 0.305% |

The later model reduced teacher-forced latent MSE and false alarms, but detected
fewer actual held-out deaths at this threshold. These death scores come from
weighted BCE and are not calibrated probabilities. The audit does not establish
which model produces a better controller; no new real-game score is available.
Reports: `artifacts/doom_packed_stop_audit_comparison.json` and the individual
`doom_packed_stop_*_audit.json` files.

The next iteration collects 1000 fresh training episodes from the frozen
incumbent using eight environments, CUDA inference, mean latents, no warmup,
and initial seed 50000. Collection survival statistics are training-data
statistics, not a new independent test result. The split retains all 870 prior
holdouts and reserves 100 of the new episodes. The planned refinement uses a
maximum of ten epochs at learning rate 0.0001, validation after each epoch, and
patience three. `train_rnn.py --save_best --early_stopping_patience 3` evaluates
the starting model as an eligible epoch-zero checkpoint, preserves matching
weights, optimizer and RNG for both best and last models, and stops if validation
fails to improve. Two integration tests exercise degrading refinement and an
improvement followed by three worse epochs; all 15 regression tests passed on
CUDA. Best-checkpoint selection precedes controller optimization. Reserved real
validation seeds 13000–13019 and
final test seeds 40000–40099 remain separate. Data-job status and provenance:
`artifacts/doom_refinement_round2_data.status` and
`artifacts/doom_refinement_round2_collection.json`.

Collection and encoding completed on 2026-10-03: 1000 episodes, 797,382 frames,
970 physical deaths and 30 timeouts. Collection survival was 797.38 ± 494.12;
this is a training-cohort statistic, not an independent controller test. The
selected pool has 3460 training and 970 validation episodes. All 4430 episodes
are unique, with no duplicate leakage, inconsistent lengths or internal terminal
labels. Reports: `artifacts/doom_refinement_round2_integrity.json` and
`artifacts/doom_refinement_round2_coverage.json`.

The short CUDA refinement stopped after epoch 6, selecting **epoch 3**. On its
fixed validation split the starting model scored 1.045809, best loss was
1.036670 and last loss was 1.037441. Best weights, matching optimizer and RNG
were frozen in `reproduction_refined_round2_selected_world/`; final weights
remain in `reproduction_refined_round2/`.
These losses use this run's split and aggregation and should not be compared
directly with the earlier packed run. Best and last checkpoints remain separate.
Real-game selection and the completed paired test are recorded below.

Paired posterior audits compared the selected world with the unchanged incumbent:

| Holdout cohort | Incumbent death recall | Refined death recall | Incumbent / refined live-frame false death rate |
| --- | ---: | ---: | ---: |
| Historical, 256 episodes | 72.66% | 76.17% | 0.430% / 0.480% |
| Previous failures, 100 episodes | 49% | 54% | 0.305% / 0.262% |
| Current policy, 100 episodes | 43% | 53% | 0.154% / 0.172% |

Using mean latents on the current-policy holdouts, recall improved from 46% to
55%, with false death rates 0.160% and 0.183%. Latent MSE also decreased in each
cohort. These are detection-threshold diagnostics, not gameplay results or proof
of calibrated death probabilities. Full reports:
`artifacts/doom_refinement_round2_audit_comparison.json`.

Two 500-generation CMA searches completed on the selected world at temperature
1.15, one with threshold death and one with sampled death. They share seed 63,
start-state data, population 64, 16 rollouts per candidate and batch 64. A separate
temperature comparison also completed on an unchanged copy of the incumbent world
at 1.0, 1.15 and 1.3, all with seed 64, sampled death, the original start-state
data, the same incumbent initialization and the same 500-generation budget.
The extra 1.15 run controls for the additional optimization budget; the untouched
840-step policy remains an additional baseline. At most two searches run at once.
Arguments, hashes and individual status paths:
`artifacts/doom_round2_controller_jobs.json`.
All five finished without search errors. Their best dream validation scores were
1153.72 (refined threshold), 1150.44 (refined sampled), and 1482.33 / 1222.08 /
967.42 (unchanged-world temperatures 1.0 / 1.15 / 1.3). These do not establish
real performance. Real validation compared each controller and the unchanged
incumbent under both mean and posterior inference on seeds 13000–13019; selection
preceded the reserved paired 100-game test. Supervisor status:
`artifacts/doom_round2_real_evaluations.status`.

## Final round-two real-game results (2026-10-03)

All twelve combinations used the same twenty validation seeds 13000–13019:

| Controller | Mean-latent validation mean | Posterior validation mean |
| --- | ---: | ---: |
| Refined world, threshold death, temperature 1.15 | 934.40 | 892.00 |
| Refined world, sampled death, temperature 1.15 | 718.40 | 612.75 |
| Unchanged world, sampled death, temperature 1.0 | 847.05 | 853.70 |
| Unchanged world, sampled death, temperature 1.15 | 876.60 | 668.40 |
| Unchanged world, sampled death, temperature 1.3 | **938.85** | 837.20 |
| Unchanged incumbent controller | 778.55 | 808.00 |

The predefined maximum-validation-mean rule selected the temperature-1.3 policy
with mean-latent inference. Its VAE, RNN, controller and selection provenance were
frozen in `reproduction_round2_real_selected/` before reserved test outcomes were
observed. Its world is identical to the incumbent world; this selected result
therefore concerns the separate temperature/controller experiment, not a new-data
world improvement. The refined threshold candidate was close on validation but
was not tested after selection. Twenty games and one CMA seed per condition do
not establish a general temperature ranking.

Both frozen policies then ran on the same 100 fresh test seeds 40000–40099:

| Policy | Mean ± population SD | Bootstrap 95% interval for mean |
| --- | ---: | ---: |
| Validation-selected temperature-1.3 policy | **775.85 ± 496.21** | 680.66–874.54 |
| Unchanged incumbent | **728.75 ± 483.70** | 635.61–825.00 |

Paired mean improvement was **47.10 steps**, with percentile bootstrap 95%
interval **−51.60 to +146.61** (50,000 resamples, seed 73400). The selected policy
survived longer in 52 games, the incumbent in 45, and three tied. This cohort
does not establish a reliable improvement. The selected sample mean exceeds the
paper's 750-step criterion, but its mean interval includes values below 750 and
the score remains below the paper's 1092 mean. The incumbent's earlier
840.06 ± 524.48 result on seeds 30000–30099 is retained separately; different
seed cohorts must not be treated as a paired comparison. Neither controller was
chosen or retrained using these final test outcomes.

Protocol remains a reused VAE with full-frame preprocessing, a controller bias,
mean real-game latents and sampled-death dream training, with no forced warmup,
a 2100-step cap and CUDA inference. These are extensions to the original paper.
Eight game workers preserve the serial calculation shapes and per-seed RNG;
replay checks matched existing serial outcomes. Fingerprints verified the frozen
models after both tests, including the unchanged incumbent controller.
Reports: `artifacts/doom_round2_selected_test100.json`,
`artifacts/doom_round2_incumbent_paired_test100.json`, and
`artifacts/doom_round2_paired_comparison.json`.

This is the requested stopping point. All experiment training and evaluation
processes have exited; no additional GPU work is queued. Original checkpoints,
the earlier incumbent, selected policy, refined best/last models, optimizer states
and datasets remain intact. No further loop, benchmark or video will run without
a new request. Resume instructions are in `artifacts/task_state.json`.

## GPU throughput checks (2026-10-03)

With two searches running, twelve samples measured 97–99% GPU activity,
roughly 2.6 GiB allocated VRAM, 55–62°C and a stable 2820 MHz core clock.
GPU activity measures time with a kernel executing; it does not measure peak
compute efficiency. A controlled benchmark on the same frozen world, initial
controller, starts and noise compared one and two processes after three warmup
generations. Each process measured eight generations with population 64,
16 rollouts and length 2100. A barrier synchronized the measurement windows;
each GPU stage was synchronized before timing. One process achieved 2.239
generations/second versus 1.923 combined with two, about 16% higher throughput
for serial searches. The dream stage accounted for about 94% of the single-job
GPU loop. Future jobs use one GPU process at a time for this workload. This is
a short benchmark, not a guarantee for different batch sizes or models.
Report: `artifacts/doom_search_concurrency_benchmark_synchronized.json`.

The real evaluator now supports `--workers 8`, with spawned games, per-seed
posterior RNG, explicit masked reseeding and reset LSTM states. Policy inference
and memory updates share one GPU call. Single-game calculation shapes are
preserved using `lax.map`: a direct `vmap` trial changed recurrent trajectories
and was rejected. Twenty mean-latent games matched the original frozen test
exactly in survival, rewards and action counts; eight posterior games matched
the serial evaluator exactly. In an eight-game benchmark with other searches
active, mean evaluation decreased from 18.943 to 14.887 seconds (1.27×).
These times include worker startup and compilation but exclude model loading;
they are not a full overnight speedup estimate. Seed/memory scheduling and
freezing explicit inference metadata passed regression tests. The GPU smoke
check now requests full multiplication precision for its known-value convolution
check, avoiding a false failure from reduced-precision rounding without changing
experiment inference precision.
Reports: `artifacts/doom_gpu_bench_mean_workers8_map.json`,
`artifacts/doom_gpu_bench_mean_reseed20_map.json`, and
`artifacts/doom_gpu_bench_posterior_workers8_map.json`.

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
  --episodes 100 --seed 30000 --workers 8 --output artifacts/new_experiment_test100.json
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

## Next iteration: vision audit and architecture comparison

CPU preparation for the new iteration is recorded in
[the vision experiment protocol](vizdoom_vision_experiment.md). The original
840.06-step incumbent and the completed paired round-two results remain
unchanged. The active goal now resumes bounded GPU experiments: CUDA preflight
passed and a serial current/reference VAE comparison started at 11:15 UTC on
2026-10-03. Progress and model provenance are recorded in the linked protocol
and `artifacts/doom_vision_round3_vae_jobs.json`.

Both 20-epoch pilots are complete (current held-out loss 45.2557, reference
46.0313). Because both were still improving, a serial continuation to at most
60 total epochs completed in fresh directories; see
`artifacts/doom_vision_round3_vae_extension_jobs.json`. No new real survival
result has been established.

The initial atlas shows the reused VAE preserving large approaching fireballs.
A fixed linear probe of its means on 100 separate newer holdout episodes gave
AUC 0.8872 for larger bright-color components and 0.7094 for smaller components.
These are color-based proxies without verified projectile labels; they support
investigating earlier/smaller visual cues but do not identify the gameplay
bottleneck. The shared data pool, reference VAE architecture and validated
training/encoding path support the running controlled comparison. No dream or
reconstruction result will substitute for real survival.

Both 60-epoch VAEs and the fixed vision diagnostics are now complete. Current
held-out VAE loss is 42.6414; reference geometry is 43.4174. Current reconstructs
the small audit cohort better but does not establish classification-proxy gains
and increases horizontal position error. Reference geometry scores worse on
both color-component proxies. Full paired intervals and limitations are in the
[vision protocol](vizdoom_vision_experiment.md#completed-feature-comparison-and-downstream-decision).
The next bounded test uses the fixed incumbent VAE and current epoch-60 VAE,
each with a fresh matched RNN/controller pipeline. The unchanged incumbent
policy remains the real survival control; all original checkpoints are retained.

The matched fresh RNNs and six 500-generation controller searches are complete.
Best RNN epochs were 15 (incumbent VAE) and 18 (current VAE), with matching
weights/Adam/RNG bundles verified and final states preserved. Current's RNN
missed more deaths than the fresh control on both fixed holdout cohorts,
despite fewer false alarms. These audits use corrected float32 posterior
inputs; legacy reports used float16 rounding. Neither model loss nor dream
scores establish a survival gain.

Real evaluation started at approximately 14:22 UTC on 2026-10-03. Fourteen
fixed policy/inference combinations receive 100 common validation games on
fresh seeds 60000–60099. The winner is frozen using validation only, then
tested against the unchanged mean-inference incumbent on reserved seeds
80000–80099. One GPU job runs at a time with eight game workers. Full protocol,
measured audit results and provenance are in the
[vision experiment record](vizdoom_vision_experiment.md#completed-fresh-rnns-and-death-audits).
All fourteen validation combinations and both reserved reports are now complete.
The best new policy averaged 467.51 validation steps; the unchanged incumbent
averaged 793.16 on the same seeds and was retained. The frozen selected policy
has identical controller parameters to the original and scored **817.91 ±
524.13** on test seeds 80000–80099, exactly matching the paired incumbent in
all 100 games. Mean bootstrap interval is [717.39, 922.52]; paired improvement
is zero. This iteration produced no improvement and remains below the paper.
The original 840.06 report is a different cohort and remains unchanged.

The next bounded pilot tests the preserved packed epoch-one best RNN against
the unchanged incumbent world, with fixed original VAE/tau 1.15 and four
matched 500-generation searches covering both termination modes. It does not
resume packed training. Proposed fresh validation/test ranges, guards and
limitations are recorded in the
[vision protocol](vizdoom_vision_experiment.md#next-bounded-slice-preserved-rnn-transfer).

CPU consolidation of the unchanged incumbent's three distinct reserved cohorts
(seeds 30000, 40000 and 80000, 100 games each) gives a descriptive mean
**795.57 ± 513.39** and cohort-stratified bootstrap mean interval
**[738.40, 854.28]**. All report weight fingerprints match the original frozen
incumbent and all 300 seeds are unique. Duplicate selected/paired copies are
excluded. This does not replace the original 840.06 benchmark or establish an
exact cross-study comparison: the oldest report lacks an explicit worker count
and inference field; its saved controller metadata/override establish mean
inference. Cohort protocols remain recorded and training uncertainty is excluded.
Artifact: `artifacts/doom_incumbent_3_cohort_summary.json`. No new games were run
and this descriptive analysis selects or tunes no policy.

## Final frozen-world comparison and stopping decision

All four frozen-RNN searches and ten real validation combinations completed.
The original VAE, controller architecture and temperature 1.15 stayed fixed;
both RNNs used the same incumbent controller initialization and CMA seed77.
Each entry is mean survival over the same 100 validation seeds 90000–90099.

| Frozen world / controller | Mean-latent inference | Posterior inference |
| --- | ---: | ---: |
| Incumbent / threshold | 825.38 | 745.97 |
| Packed best / threshold | 756.63 | 752.15 |
| Incumbent / sampled death | 802.56 | **879.74** |
| Packed best / sampled death | 742.60 | 772.87 |
| Unchanged original controller | 733.51 | 737.71 |

Validation selected the incumbent-world sampled-death controller with posterior
inference. Its world, policy, inference mode and all validation-report hashes
were frozen before the reserved tests. On 100 fresh paired seeds
110000–110099 it scored **853.53 ± 572.26**, versus **819.15 ± 478.46** for the
unchanged original mean-inference incumbent (population SDs). The paired mean
difference was **+34.38**, with 95% whole-game bootstrap interval
**[-85.29, +156.04]**; 47 wins, 52 losses and one tie. The selected mean's
interval is [743.72, 967.45]. These 50,000-resample intervals hold fitted
policies fixed and exclude training variability. A reliable gain is not
established. Test outcomes did not select or retrain a policy.

The selected candidate remains in
`checkpoints/VizdoomTakeCover-v0/frozen_transfer_round4_real_selected`;
the canonical original remains in `reproduction_refined_selected`. Keeping
the validation-selected candidate does not establish superiority on unseen
games. The packed best was evaluated as frozen weights; the deliberately
stopped epoch255 training run was never resumed. Its best has no saved best
optimizer; its latest weights, optimizer and RNG remain preserved.

Artifacts: `artifacts/doom_frozen_transfer_round4_real_evaluation_result.json`,
`artifacts/doom_frozen_transfer_round4_frozen_selection.json`, and
`artifacts/doom_frozen_transfer_round4_paired_comparison.json`.

The unchanged incumbent's four distinct reserved cohorts (30000, 40000, 80000
and 110000, 100 games each) give **801.47 ± 504.99** across 400 unique games,
with cohort-stratified bootstrap mean interval **[752.44, 851.16]**. All three
weight fingerprints match across the reports. Duplicate paired copies are
excluded. This descriptive consolidation replaces neither the original
**840.06 ± 524.48** cohort nor its protocol; the oldest report lacks explicit
worker/inference fields, with mean inference established by its saved metadata
and override. No additional games were run for this analysis. Artifact:
`artifacts/doom_incumbent_4_cohort_summary.json`.

The stopping decision uses the accumulated outcomes:

| Completed loop | Selected minus unchanged incumbent, paired steps | 95% interval |
| --- | ---: | ---: |
| Fresh data/refinement and temperature comparison, round2 | +47.10 | [-51.60, +146.61] |
| Vision and matched fresh downstream training, round3 | 0.00, unchanged incumbent retained | [0.00, 0.00] |
| Frozen RNN and termination comparison, round4 | +34.38 | [-85.29, +156.04] |

Round3's zero is an identical-policy check: all twelve new combinations lost
to the incumbent on validation, so it is not evidence that other policies
have zero uncertainty. The current VAE pipeline did outperform its matched
fresh incumbent-VAE controls by 169.75–275.81 validation steps across the six
CMA-seed/inference comparisons. Both fresh pipelines lost to the original
incumbent; that comparison cannot isolate the VAE from RNN training history.
The VAE result should not be described as an isolated VAE failure.

Across these follow-ups, fifteen 500-generation controller searches, new
policy-relevant data, short RNN refinement, two 60-epoch VAE comparisons and
matched fresh downstream models produced no reliable improvement on the
reserved real-game comparisons. Long packed training also failed to improve
held-out loss after epoch1. Continuing small changes within these tested
families has diminishing measured returns, so this effort stops here. This is
an empirical resource decision, not a claim that 853 steps is a fundamental
ceiling or that every possible future experiment will fail.

The paper reports **1092 ± 556** at temperature 1.15 over 100 real games
([primary paper](https://worldmodels.github.io/#cheating-the-world-model)).
Our environment/preprocessing, sampled frame pool, training histories,
initialization and affine controller differ from the reference procedure;
round2's on-policy data is an extension. Reconstruction loss, weighted death
scores and dream scores do not establish paper-level real performance. The
remaining protocol differences are recorded in the linked vision protocol.

All supervisors exited and final CPU analysis completed. The GPU is released
from this repository's work. The five-minute follow-up is paused after
completion; `artifacts/task_state.json` records consumed seeds, candidate and
incumbent paths, and a future-resume boundary. A future GPU experiment needs a
new request; completed test seeds must not become tuning or validation seeds.
Completion verification is in `artifacts/doom_experiment_completion_audit.json`.

The preceding closure describes rounds2–4. The renewed user goal to match or
beat1092, with GPU availability reconfirmed, authorizes round5; the goal remains
active. The source-compatible imported reference world now reaches991.20
±506.48 on100 diagnostic games. This is a supplied-model control, not a new
training reproduction. A bounded own-controller refinement at temperature1.15
has completed500 generations. Six fresh20-game validation cases selected the
generation270 controller with posterior inference at958.05 ±596.39, versus
supplied posterior877.75 ±478.91. The policy was frozen before reserved100-game
testing. The selected controller completed100 fresh games at1035.22 ±550.45,
below1092. The paired supplied-policy control completed955.39 ±541.20;
gain+79.83 has95% interval[-22.26,+183.00], so no reliable improvement is
established. The complete independent CPU audit passed, and both processes
exited. A controlled temperature1.25 follow-up completed500 generations with
all other training settings fixed; its best dream checkpoint is generation210.
All four80-game real validations completed: supplied control923.11 ±544.26
was retained over prior1.15 policy904.83 ±511.93, new1.25 best914.99 ±521.24
and final914.28 ±514.03. Neither new policy won, so the reserved150000 cohort
was not consumed. Independent CPU verification checked records, selection,
fingerprints and absence of reserved games; both evaluation processes exited.
A preregistered1.15 local search with sigma0.005 and1024 dream validation
rollouts then completed500 generations, best950.0654 at generation120.
Independent CPU training-state review passed, including500-generation history,
optimizer/final weights/RNG and unchanged inputs/source/incumbent. The serial
four-policy80-game real validation completed: prior920.03, supplied944.91,
new best947.80 and final875.79. Generation120 narrowly won; its paired
validation gain over supplied is+2.89 with95% interval[-113.21,+118.34], so
no reliable gain is established. The winner and highest validation control,
supplied posterior, were frozen before testing and independently CPU-audited.
The selected100-game160000 test completed945.73 ±570.08, mean95% interval
[836.80,1057.28], below1092. Its records, frozen policy and timing passed CPU
verification; supplied pairing completed922.62 ±562.71 on the same seeds.
Paired gain+23.11 has95% interval[-64.56,+109.06], establishing no reliable
gain. The complete protocol-aware CPU audit passed; both evaluation processes
exited and canonical checkpoints remain intact. A
64-versus16 fitness-rollout comparison was preregistered from
training/validation evidence; its readiness check passed after this pair closed.
The single CUDA search completed500 generations, with best953.3916 at
generation300 and final CMA mean901.1104. Independent CPU review passed for
history, optimizer/final weights/RNG, inputs/source and preserved controls;
the search exited0. Four80-game real validations then started serially after
fresh-seed/process-exit checks and CUDA preflight. The prior control completed80
validation games at973.29 ±520.30, mean95%[860.60,1090.03]; actual records, cohort,
source/controller fingerprints and preserved incumbent passed CPU checks.
Supplied completed922.10 ±543.40 on the same80 seeds, mean95%[804.71,1042.63].
Prior-minus-supplied paired gain+51.19 has95%[-56.91,+162.31], establishing no
reliable advantage. Both complete cohorts and shared world/source passed CPU
verification. New best-generation300 completed931.41 ±523.98; final-generation500
completed1016.39 ±560.94 and won validation. Its paired gain over prior is
+43.10,95%[-86.03,+176.65], establishing no reliable improvement. The winner
and prior comparator were frozen before reserved170000 tests; independent CPU
reconstruction checked all complete cohorts, identities, world/source and
timing. The selected100-game test completed **1026.99 ±597.70**,90 deaths/
10 timeouts, mean95% **[911.33,1145.64]**. CPU verification checked actual
records, frozen controller/source/world/provenance and preserved canonical
weights. The frozen prior completed1040.31 ±616.27 on the same100 seeds;
paired selected-minus-prior **−13.32**,95% **[−129.75,+103.78]**, establishes
no reliable improvement. The protocol-aware paired/source/provenance audit
and independent completion review passed; session10253 exited0, original
weights remain intact and no prior GPU work remains queued. The selected
observed mean stays below1092. A validation-winner initialization recipe
is registered from training/validation evidence, with CPU compatibility checks
passed; its materialized launch gate requires the completed pair/audit, actual
process absence, unchanged inputs, fresh cohorts and CUDA preflight. It keeps
the validation-selected initializer fixed regardless of the reserved test scores.
Readiness passed with no live prior process, then the single CUDA search
started in session83493 after matrix multiplication/convolution preflight.
Its settings and exact initializer match the registered recipe; performance
remains unproven until the separate fresh real evaluation.
The search completed500 generations with best965.6523 at generation380,
initial923.2109 and final CMA mean892.9414. Independent CPU review verified
history, exact initializer/source, best/final parameters, complete optimizer
and RNG state, and preserved controls. Session83493 exited0 and both training
processes exited. Fresh-cohort/current-state/exclusive-lock readiness passed
before serial real validation started in session30040; CUDA preflight passed
again. The complete80-game cohorts and independent selection audit precede
any eligible fresh100-game paired test. Real1092 performance is still unproven.
The unchanged initializer completed80 validations at1031.74 ±617.01,
mean95%[897.89,1168.05]; actual records, registered source/provenance and
preserved canonical weights passed independent CPU checks. The supplied
control completed80 games at982.68 ±520.17; independent CPU comparison of
both controls found prior-minus-supplied+49.06, paired95%[−73.88,+172.28],
with no reliable gain established. Both updated candidates and the complete
validation-only selection remain pending under the same serial supervisor.
The new best-generation380 checkpoint subsequently completed80 validations
at961.74 ±501.45. Independent CPU checks passed; minus prior−70.00 has
paired95%[−207.94,+67.68]. The final-generation500 cohort is running, and
complete selection and any eligible reserved tests remain pending.
The final checkpoint subsequently completed80 games at940.85 ±535.48.
All four complete cohorts retained the prior1031.74 control; neither new
candidate won. Independent selection/no-test closure and current runtime,
controller/source/training-state/canonical/process audits passed. Session30040
exited0; reserved180000–180099 were not consumed, and1092 remains unproven.
Round10 is a fixed-settings seed92 repeat from the same initializer; seed91→92
changes CMA samples, dream streams and start-pool partition together. Fresh
validation130340–130419 and contingent paired100-test190000–190099 are
registered before outcomes. Exact arguments/initializer/source, disjoint
start pools, prior closure, fresh cohorts and exclusive lock passed CPU checks.
One search started in session49055 after CUDA matrix multiplication/convolution
backward preflight, preserving best/final/optimizer/RNG state. A distinct new
validation winner is required before reserved testing; no real result exists yet.
The serial real supervisor and independent auditors are prepared; seven CPU
fixtures pass selection, pairing, deduplication and audit-rejection gates.
Actual readiness defers incomplete training/audit, numerical inference is
unchanged and no real evaluation is launched. Existing search progress239/500
at the measured snapshot does not establish the1092 target.
Round10 then completed500 generations: best954.8711 at generation1 and
final CMA mean953.3369. Independent CPU history/initializer/source/start-pool/
checkpoint/optimizer/RNG checks passed. Session49055 exited0 and both training
processes exited. Actual complete-state/fresh-cohort/exclusive-lock readiness
passed before one serial real supervisor started in session87088. CUDA
matrix multiplication/convolution backward preflight passed again. First
stage is80 fresh unchanged-prior posterior validations130340–130419; the
complete four cohorts and independent frozen choice precede eligible paired
100-test190000–190099. No real1092 result is yet established.
The unchanged prior completed80 validations at936.94 ±547.92,
mean95%[818.01,1057.30]; raw records, registered source/input/inference/current
runtime and canonical preservation passed independent CPU checks. The supplied
control and both new policies remain under the same serial supervisor; no
validation winner is frozen yet.
The supplied control then completed80 validations at936.08 ±561.21.
Independent CPU checks passed for both controls' records, source/input hashes,
current packages and canonical preservation. Prior minus supplied+0.86 has
paired95%[−126.09,+126.04]; neither control has an established advantage.
The two updated policies and full validation selection remain pending.
The new best-generation1 checkpoint subsequently completed80 validations at
915.18 ±529.99; independent raw-record/source/input/current-runtime/controller
checks passed. Best minus prior−21.76 has paired95%[−109.93,+69.14]. The
final checkpoint is running; no policy is frozen and1092 remains unverified.
The final-generation500 checkpoint then completed80 validations at929.45
±573.13. Final minus prior−7.49 has paired95%[−133.54,+116.25]. Complete
validation retained the unchanged prior936.94 control. Independent selection/
no-test closure and current source/runtime/controller/training-state/canonical/
process checks passed. Session87088 exited0, both processes exited and no work
is queued. Reserved190000–190099 remain unused;1092 remains unverified.
A controlled diagnostic then demonstrated current batch-shape sensitivity:
identical weights/starts/keys scored923.2109 at1024 trajectories and910.6230
at2048. Within-shape repeats and duplicated slots were exact;783/1024 outcomes
differed between shapes. Independent CPU raw-score/source/input/RNG checks
passed and session84793 exited0. This does not fully explain the older901.1104
result or establish real survival improvement. A new trainer preserves earlier
snapshots and validates every candidate with the same single-candidate shape
as the baseline. Actual-function CPU fixtures, syntax/Ruff and diff checks
pass; its GPU check and new experiment registration remain pending.
The corrected function subsequently passed GPU verification: all six copied-
policy score arrays used1024-trajectory calls and matched exactly. Independent
CPU source/input/raw-score/runtime checks and actual session/process exit passed.
Round11 is registered and running as a same-settings seed92 replay with fixed
held-out call shape. Fresh validation130420–130499 and contingent paired
100-test200000–200099 are reserved; failed round10 policy identities are
excluded. Session55459, supervisor1123810/trainer1124156 were verified live
after CUDA preflight. The initial mean887.4785 differs from the prior870.5869
despite matched recorded inputs/source/runtime; cause remains unresolved. At
83/500, all83 population statistics match round10, while full replay and real
improvement remain unproven. Nine CPU real-routing fixtures and guards passed;
actual readiness defers incomplete training. No real evaluation is dispatched.
Round11 then completed500 generations: best954.5557 at generation470,
initial887.4785 and final915.3770. Independent CPU history/input/source/
checkpoint/optimizer/RNG checks passed. All500 population mean/best statistics
and final raw weights match round10; the previously failed final is excluded.
The best is a distinct eligible candidate. Session55459 exited0; both training
processes exited and actual complete-state/fresh-cohort/lease readiness passed.
One real supervisor started in session15768, supervisor1149459/evaluator1149804
verified live after CUDA preflight. It validates both controls and generation470
on80 fresh posterior games130420–130499. Complete independent selection and
any eligible paired100-test200000–200099 remain pending;1092 is unverified.
The unchanged prior controller then completed80 validations at953.56 ±529.86,
76 deaths/4 timeouts, mean95%[839.74,1071.74]. Independent CPU raw-record/
controller/source/input/current-runtime/canonical checks passed. The public
control is running under the same supervisor, followed by generation470;
complete selection remains pending. Artifact:
`artifacts/doom_reference_round11_validation_progress_1_cpu_audit.json`.
The public control subsequently completed80 validations at972.89 ±546.46,
75 deaths/5 timeouts, mean95%[854.02,1094.39]. Independent CPU checks passed
for both controls. Public minus prior+19.33 has paired95%[−94.28,+130.69],
so no reliable advantage is established. Artifact:
`artifacts/doom_reference_round11_validation_progress_2_cpu_audit.json`.
Generation470 is now under the same live supervisor; complete selection
remains pending. These control validations do not establish1092.
Generation470 then completed80 validations at963.89 ±626.80,72 deaths/8
timeouts, mean95%[828.96,1101.34]. Candidate minus public−9.00 has paired95%
[−136.10,+121.86]. Complete validation retained the unchanged public control.
Independent selection/no-test closure verified all240 records, frozen source/
world/parameters, initializer provenance and no200000–200099 consumption.
Current-runtime/training-state/canonical and actual session/process/lease checks
passed; session15768 exited0, all related processes exited and no GPU work
is queued. Artifacts: `artifacts/doom_reference_round11_validation_closure_cpu_audit.json`
and `artifacts/doom_reference_round11_completion_cpu_audit.json`. The corrected
validation selected a different policy but did not establish better real transfer
or1092. The goal remains active.
Round12 is preregistered as matched1.15/1.10 dream-temperature searches,
seed91/public initialization, sigma0.005,500 generations,pop64,fitness64 and
1024 held-out rollouts with the unchanged fixed-validation trainer. Only
temperature and output paths differ between arms. Six previously failed
identities are excluded; fresh validation130500–130579 and contingent paired
100-test210000–210099 passed CPU reservation. Protocol:
`artifacts/doom_reference_round12_temperature_pair_protocol.json`. Serial
launcher/readiness and independent audits remain to be prepared; no new GPU
job has started.1.10 is our hypothesis, not a paper-prescribed setting or claim.
The serial launcher/independent arm auditor then passed CPU argument parsing,
complete synthetic public-initializer capsule and corruption/routing/recovery
fixtures; syntax/Ruff passed and helper sources were frozen. Actual parent/
source/runtime/initializer/fresh-cohort/lease readiness passed. Session42213,
supervisor1202880 and first1.15 trainer1203226 were verified live after CUDA
preflight,7/500 with96% GPU utilization measured. Independent live initial
checks passed at27/500; initial dream928.3779296875 equals round8's recorded
initial mean, without proving full-training replay or real gain. Artifacts:
`artifacts/doom_reference_round12_training_preparation_cpu.json`,
`artifacts/doom_reference_round12_training_readiness_cpu_audit.json` and
`artifacts/doom_reference_round12_tau115_initial_cpu_audit_correction.json`.
The same supervisor will audit the finished first arm before launching1.10;
real workflow preparation and survival evidence remain pending.
The 1.15 arm then completed 500 generations with best held-out dream score
948.20703125 at generation 70. Independent training-state/source/checkpoint/
RNG/canonical checks passed. The same supervisor started 1.10 after CUDA
preflight; it was verified live at generation 23 on 2026-10-04T06:18:11Z.
Initializer, start partitions, random-stream metadata and calculation shapes
match between arms. Artifacts:
`artifacts/doom_reference_round12_tau115_training_cpu_audit.json` and
`artifacts/doom_reference_round12_tau110_initial_cpu_audit.json`.
The two-arm real workflow is prepared and frozen in
`artifacts/doom_reference_round12_real_workflow_preparation_cpu.json`. Twelve
isolated CPU fixtures passed, covering independent selection before tests,
audit-rejection blocking, retained-control closure and paired100 analysis.
Real evaluation has not started; both completed searches and actual training
exit are required before fresh readiness and dispatch. The target remains
unproven, regardless of dream scores.
Both searches then completed and actual training session 42213 exited 0
(tool chunk `515ec3`). The 1.10 best held-out dream score is 1076.3115234375
at generation 310. Both independent arm audits and current source/runtime/
checkpoint/optimizer/RNG/canonical/process/lease checks passed. The 1.15
final raw vector exactly duplicates the prior control, so the final comparison
has five distinct policies and 400 validation games. Fresh readiness passed;
one real supervisor started in session 16340, PID 1253545. CUDA preflight
passed and evaluator 1253615 was verified live at 1/80 public-control games
on 2026-10-04T06:43Z. Complete validation selection and any eligible reserved
paired100 remain pending. Artifacts:
`artifacts/doom_reference_round12_training_completion_cpu_audit.json`,
`artifacts/doom_reference_round12_real_dispatch_readiness_cpu.json` and
`artifacts/doom_reference_round12_registered_validation_cases_cpu.json`.
The public control then completed all 80 validation games at 957.56 ±599.62,
73 deaths / 7 timeouts, fixed-policy 95% mean interval [828.66,1092.14].
Independent raw-record/controller/source/input/current-runtime/environment/
canonical checks passed. Artifact:
`artifacts/doom_reference_round12_validation_progress_1_cpu_audit.json`.
The same supervisor is evaluating the prior local controller, evaluator
1261348 verified live. All five cohorts and independent selection remain
pending; this control validation is not a reserved test or proof of 1092.
The prior local controller completed 80 validations at 913.01 ±502.18,
78 deaths / 2 timeouts, mean95% [803.20,1024.40]. Prior minus public −44.55
has paired95% [−171.36,+84.65], so no reliable improvement is established.
Independent checks passed for both complete raw cohorts, actual control
identities, source/inputs/current packages/shared environment and canonical
preservation. Artifact:
`artifacts/doom_reference_round12_validation_progress_2_cpu_audit.json`.
The same supervisor advanced to the distinct 1.15 generation-70 candidate,
evaluator 1271193 verified live. Three candidate cohorts and independent
complete selection remain pending; no reserved test has started.
The distinct 1.15 generation-70 candidate completed 80 validations at
926.31 ±579.07, 75 deaths / 5 timeouts, mean95% [801.87,1055.04]. Candidate
minus public −31.25 has paired95% [−141.41,+77.48]. Independent raw-record/
actual-policy/generation/temperature/provenance/source/input/runtime/shared-
environment/canonical checks passed for all three complete cohorts. Artifact:
`artifacts/doom_reference_round12_validation_progress_3_cpu_audit.json`.
The same supervisor advanced to temperature 1.10 best, evaluator 1281065
verified live. Both 1.10 cohorts and complete selection remain pending;
the 1.15 candidate did not outscore public, and no reserved test has started.

Round12 is now complete: each policy below played the same 80 fresh real
validation games, seeds 130500–130579, using posterior inference.

| Policy | Mean ± standard deviation | Deaths / timeouts | Paired difference from public, 95% interval |
| --- | --- | --- | --- |
| Public control | 957.56 ±599.62 | 73 / 7 | 0 |
| Prior local control | 913.01 ±502.18 | 78 / 2 | −44.55 [−171.36,+84.65] |
| Temperature 1.15 best, generation 70 | 926.31 ±579.07 | 75 / 5 | −31.25 [−141.41,+77.48] |
| Temperature 1.10 best, generation 310 | 904.69 ±524.83 | 78 / 2 | −52.88 [−181.60,+75.13] |
| Temperature 1.10 final, generation 500 | 922.06 ±522.69 | 76 / 4 | −35.50 [−174.74,+103.65] |

Complete validation-only selection retained the unchanged public control.
None of the three new policies won; **no reserved test was permitted or run**,
and seeds 210000–210099 remain unused. The intervals use 50,000 whole-game
paired bootstrap resamples, seed 74180, with policies and world fixed; they
exclude training and selection uncertainty. The higher 1.10 dream score did
not translate into better real survival, and the 1092-step goal remains unmet.
This used the imported public world and the documented modern runtime,
environment and RNG differences; it does not establish our own world-model
training reproduction. The canonical own-world 840.06 ±524.48 result is intact.

Independent CPU audits verified all 400 raw records, frozen selection,
unused test seeds, actual source/input/runtime/control/provenance fingerprints,
training checkpoints and optimizer/RNG state. Real session 16340 exited 0
(tool chunk `3c3cbf`); the 2026-10-04T07:47:48Z final review confirmed all
experiment processes had exited, the GPU lease was available and no further
GPU work was queued. Resume from the completed round12 state rather than
restarting its closed sessions. Artifacts:
`artifacts/doom_reference_round12_frozen_selection_cpu_audit.json`,
`artifacts/doom_reference_round12_validation_closure_cpu_audit.json` and
`artifacts/doom_reference_round12_completion_cpu_audit.json`.

Round13 has started one bounded controller search using **real-game survival
as fitness**. This changes the paper's dream-only controller training method;
it retains the imported public world, controller architecture and posterior
inference. The initializer is the public control selected from complete
round12 validation. Settings are seed94, sigma0.005,16 generations,
population16,four common fitness games per candidate and eight persistent
game workers. Fitness uses seeds500000–500063; the public baseline and means
at generations4,8,12,16 use sixteen training holdouts510000–510015. These
training measurements cannot select the final policy for reserved tests.

Fresh validation remains80 posterior games per eligible policy on
130580–130659, with both controls and distinct best/final candidates, excluding
prior failed identities. All cohorts must complete before selection; controls
win ties. Only a new winner, frozen and independently verified from validation,
gets100 reserved games220000–220099 and100 games for the frozen paired control.
No validation or test outcome informs this training run. Maximum training is
1,024 fitness games plus80 training-holdout games; maximum validation is320
games, with the200-game reserved pair conditional on a new winner.

Seven CPU tests passed, including exact interruption recovery and independent
CMA reconstruction. Actual-model CPU/GPU actor checks matched the original
actions, hidden states, keys and dtypes on the tested short inputs. Fresh
source/input/runtime/seed/process/lease readiness and CUDA preflight passed.
One supervisor is running in session52524; trainer1323841 was independently
verified live at2026-10-04T08:10:33Z with3/16 baseline holdout games complete.
The next gate is completed training and independent CPU reconstruction before
fresh validation. The1092-step target remains unproven. Estimated full-loop
duration is4–6 hours, based on prior measured80-game evaluations of606–643
seconds; new scheduling and compilation costs may differ. Protocol and logs:
`artifacts/doom_reference_round13_direct_real_protocol.json`,
`artifacts/doom_reference_round13_training.log` and
`artifacts/doom_reference_round13_training.status`.

The fresh-validation and conditional paired-test workflow is now prepared;
13 CPU tests passed, including complete-cohort requirements, independent
audit rejection preventing testing, ties/no-test closure, recovery, policy
eligibility, world/seed drift and incomplete paired-test rejection. Preparation:
`artifacts/doom_reference_round13_real_workflow_preparation_cpu.json`.
Its actual readiness check deferred GPU evaluation until the existing trainer
and supervisor exit successfully and the independent training audit passes.
At2026-10-04T08:38:18Z, three generations were complete and generation4 had
16/64 fitness games recorded. Generation3's best four-game fitness was1092.0;
this is a training score, not a fresh100-game result. The fixed training
holdout still retains the baseline768.5625 until the next checkpoint check.
Actual supervisor and trainer processes remain live; no fresh validation or
reserved test has started.

Round8's initial1024-dream mean differs by0.2979 despite matching
recorded inputs, source, initial parameters and runtime metadata; the cause
is unresolved and is documented in the reference audit.
Actual1092 performance remains unproven. Dream scores
establish no real survival improvement. Source migration
evidence, complete diagnostic results and the controlled protocol are in
[the reference audit](vizdoom_reference_audit.md). Historical reports and
checkpoints remain intact, and the stopped packed run is not resumed.

At 2026-10-04T08:49:36.958151+00:00, generation4 improved the same16 training holdouts from
**768.5625 ±343.2047** to **895.25 ±585.4425** (sample standard deviations).
The paired mean change was **+126.6875**; a fixed-cohort whole-game bootstrap
95% interval was **[-86.8750,343.6250]**, with5 improved games,4 worse
and7 ties. Both complete raw cohorts, seeds, policy fingerprints and the
current immutable optimizer snapshot were verified on CPU in
`artifacts/doom_reference_round13_g004_holdout_progress_cpu.json`. This is
a repeated training holdout used for checkpoint selection, not fresh policy
validation or a reserved100-game result. Generation5 fitness is running;
supervisor1323498 and trainer1323841 were independently verified live.
The1092-step goal remains unproven.

At 2026-10-04T09:24:38.740839+00:00, generation8 scored **888.625 ±583.4805** on the same16
training holdouts (sample standard deviation), below generation4 **895.25**.
Generation4 remains best. Generation8 minus baseline768.5625 was
**+120.0625**, fixed-cohort paired bootstrap95% **[-112.5641,382.6875]**;
generation8 minus generation4 was **−6.625**, interval
**[-234.0000,200.1875]**. CPU reconstruction checked all8 candidate
populations, fitness means/history, mean and best weights, next pending
population and **560 complete raw training-game records**, with frozen
source/input fingerprints, in
`artifacts/doom_reference_round13_g008_holdout_progress_cpu.json`. These are
repeated training holdouts used for checkpoint selection, not fresh policy
validation or the reserved100-game target. Generation9 fitness is running;
the original supervisor1323498 and trainer1323841 were verified live.
The registered16-generation search continues; fresh validation remains pending.

At 2026-10-04T10:01:38.356194+00:00, generation 12 became the best training checkpoint, scoring
**966.0625 ±620.9954** on the same 16 training holdouts (sample standard
deviation), above generation 4 **895.25** and generation 8 **888.625**.
The paired gain over baseline **768.5625** was **+197.5**, with fixed-cohort
whole-game bootstrap 95% interval **[-52.0000,467.0625]**. The gain
over generation 4 was **+70.8125**, interval **[-113.6250,302.8141]**.
CPU reconstruction verified all 12 populations, raw fitness means/history,
optimizer mean and best weights, the next pending population and **832 complete
raw training-game records**, with frozen source/input fingerprints. Evidence:
`artifacts/doom_reference_round13_g012_holdout_progress_cpu.json`. These are
repeated training holdouts used for checkpoint selection; they do not establish
fresh policy performance or the reserved 100-game 1092-step mean. Generation
13 fitness is running, with supervisor 1323498 and trainer 1323841 verified
live. The registered 16-generation search continues before fresh validation.

At 2026-10-04T10:33:47.970071+00:00, round13 completed all16 generations and **1104 training games**.
The independent CPU auditor reconstructed every candidate population, complete
raw fitness/holdout cohort, final mean and selected best weights, history,
optimizer and next RNG population. Its current file hashes, registered
source/inputs/runtime and canonical checkpoints were verified again after
actual supervisor session52524 exited0 (tool `f6f8db`). Supervisor1323498,
trainer1323841 and auditor1417088 were absent, repository training/evaluation
workers had exited, and the exclusive GPU lease was available. Evidence:
`artifacts/doom_reference_round13_training_cpu_audit.json` and
`artifacts/doom_reference_round13_training_exit_cpu_review.json`.
Generation16 is best at **1011.375 ±672.5837** on16 repeated training
holdouts (sample standard deviation), versus baseline768.5625. Paired gain:
**+242.8125**, fixed-cohort bootstrap95% **[-11.6891,495.6891]**. These
training/selection measurements do not establish fresh policy performance.
Best and final raw weights are identical, so fresh validation has three
distinct policies: public, prior initializer control and the generation16
candidate, each on80 new games130580–130659. No validation or reserved
220000–220099 test has started yet; the1092-step goal remains unproven.

At 2026-10-04T10:39:23.574595+00:00, the prepared fresh real comparison passed readiness (tool
`0a73c1`) and started in **session2935**, supervisor1417956. CUDA preflight
passed; actual public-control evaluator1418302 was verified live. The three
distinct policies each use80 posterior games on the same fresh seeds
130580–130659. Current measured completion: public control **43/80**, prior control **0/80**,
and new candidate **0/80**. Global seed
separation outside independently verified own reports passed, with reserved
220000–220099 unused. Evidence:
`artifacts/doom_reference_round13_real_dispatch_cpu_review.json`. The existing
supervisor must finish all240 validation games and independently audit its
frozen validation-only choice before any eligible100-game candidate/control
pair. No winner is frozen yet, no reserved test has begun, and no fresh
performance claim or1092-step goal completion is established.

At 2026-10-04T10:45:41.253394+00:00, the public control completed all80 fresh validation games
130580–130659 at **875.85 ±513.6677** (population standard
deviation), with 78 deaths and 2 timeouts.
The complete raw cohort, role, policy/world/input/source/runtime metadata and
reported statistics were checked on CPU. Report:
`artifacts/doom_reference_round13_val_supplied.json`, SHA256
`54d6a8ae68ff2ced285e67bcd411db110514cf92d2295f2f96bb1786bbaae718`. Existing session2935 has moved to the unchanged
prior-control cohort; the new candidate has not yet been evaluated. All three
complete80-game cohorts remain required before selection. No winner is frozen
and reserved220000–220099 seeds remain unused; this control result does not
establish the1092-step target.

At 2026-10-04T10:54:09.828026+00:00, the unchanged priorR8 control completed all80 fresh validation
games at **926.35 ±492.7647** (population standard deviation),
with 77 deaths and 3 timeouts. Its complete
raw records, source/world/runtime/provenance and statistics were independently
checked on CPU; public control remains875.85 ±513.6677 on the same seeds.
Prior report: `artifacts/doom_reference_round13_val_prior_initializer.json`,
SHA256 `1166d07da2fd2c07e2df41a465fc1a87b9c10b94e8fe662c0e68cc81e1266e2c`. Existing session2935 is now evaluating
the generation16 candidate. Its complete80-game cohort is still required
before any winner/control is frozen; reserved220000–220099 remains unused.

Round 13 completed all 240 fresh validation games, with each policy evaluated
on the same 80 seeds, 130580–130659:

| Policy | Mean steps | Population standard deviation | Deaths / timeouts |
| --- | ---: | ---: | ---: |
| Public controller | 875.85 | 513.67 | 78 / 2 |
| Prior round 8 controller | 926.35 | 492.76 | 77 / 3 |
| New round 13 controller | 885.29 | 562.21 | 76 / 4 |

The prior controller was retained using complete validation only. The new
controller's paired mean change was **−41.06 steps**, with a whole-game
bootstrap 95% interval of **[−130.30, +47.24]**. This interval conditions on
the fixed policies and excludes training and sequential selection uncertainty.
The training-holdout improvement did not establish fresh validation improvement.
No reserved test was run; seeds 220000–220099 remain unused, and the paper's
1092-step target remains unmet.

The independent CPU result audit verified all raw validation records, current
source/input/runtime fingerprints, checkpoint eligibility and the frozen choice:
`artifacts/doom_reference_round13_real_result_cpu_audit.json`. The supervisor
exited successfully; its recorded processes and repository workers were absent
and the GPU lease was available at the precommit check. This experiment trained
the controller directly on real simulator survival using an imported public
VAE/RNN, which differs from the paper's dream-only controller training and does
not establish reproduction of our own world-model training.

Round 14 was registered and started on CUDA at 2026-10-04T11:20Z. It warm-starts
the prior round 8 controller selected from complete round 13 validation, uses
eight fitness games per candidate per generation and 64 repeated training
holdouts, and is bounded to eight CMA generations with population 16 and
sigma 0.005. The imported VAE/RNN, architecture and inference protocol stay
fixed. This combined refinement does not isolate which change affects results.
All 21 CPU workflow/replay/recovery tests and the actual trainer readiness check
passed; CUDA preflight passed on the RTX 4070 Ti. Current baseline holdout
completion was 15/64 at 2026-10-04T11:22:29.934276+00:00. This is training progress,
not fresh policy performance. Supervisor session 49443 and its trainer were
independently verified live.

Fresh fitness seeds 520000–520063, training holdouts 530000–530063, validation
seeds 130660–130739 and reserved tests 230000–230099 were checked unused and
separated before dispatch. All eligible policies require complete 80-game
validation; only a distinct new validation winner proceeds to the complete
100-game reserved comparison with the unchanged control on the same seeds.
The planned four-hour window is an estimate. Protocol and resume evidence:
`artifacts/doom_reference_round14_direct_real_protocol.json`,
`artifacts/doom_reference_round14_training_preparation_cpu.json`, and
`artifacts/task_state.json`. The paper target remains unmet.

At 2026-10-04T11:29:41.782983+00:00, round 14 completed its 64-game training baseline at
**970.125 ±537.5760** steps (sample standard deviation), with
62 deaths and 2 timeouts. CPU reconstruction verified all raw records,
registered seeds/settings and initializer weights, saved generation-zero best
and final weights, and the first 16-candidate CMA population and RNG state.
Evidence: `artifacts/doom_reference_round14_baseline_cpu_audit.json`.
Generation 1 is evaluating its 128 candidate/game pairs, with the existing
supervisor session 49443 and trainer independently verified live. This is
a repeated training holdout, not fresh validation or the reserved 100-game
result. The 1092-step target remains unmet.

At 2026-10-04T11:41:44.408753+00:00, round 14 completed generation 1 and started generation 2.
The independent CPU prefix audit verified all **192 completed training-game
records** (64 baseline holdouts plus 128 fitness pairs), reconstructed the
first population and fitness means, saved optimizer mean/best/final weights
and history, and the next pending population and RNG. Evidence:
`artifacts/doom_reference_round14_g001_training_prefix_cpu_audit.json`.
Its hashes describe the saved files at that check; mutable checkpoint/pointer
files will advance as training continues, while the audited optimizer snapshot
and completed raw cohorts are preserved. The best training holdout remains
the generation-zero baseline **970.125**; the next holdout evaluation is at
generation 4. Existing supervisor session 49443 and its exact trainer command
were verified live. No fresh validation or reserved test has started, and
these training fitness scores do not establish the 1092-step target.

At 2026-10-04T11:53:19.751702+00:00, round 14 completed generation 2. CPU prefix replay
verified all **320 completed training-game records**, both fitness populations
and rankings, saved mean/best/final weights and history, and the pending third
population and RNG. Evidence:
`artifacts/doom_reference_round14_g002_training_prefix_cpu_audit.json`.
Generation 3 is running in the same verified live session 49443. The best
training holdout remains generation zero at **970.125**; its next comparison
is at generation 4. Fresh validation and reserved testing remain pending,
so the 1092-step objective is still unproven.

At 2026-10-04T12:10:22.267991+00:00, round 14 completed generation 3. CPU replay verified
all **448 completed training-game records**, three populations and rankings,
saved mean/best/final weights and history, and pending fourth population/RNG.
Evidence: `artifacts/doom_reference_round14_g003_training_prefix_cpu_audit.json`.
Generation 4 is running in the same verified live session 49443; after its
fitness population it will compare the mean controller on all 64 training
holdouts against the generation-zero baseline **970.125**. Fresh validation
and reserved testing remain pending. These training records do not establish
the 1092-step objective.

At 2026-10-04T12:32:05.070744+00:00, round 14 generation 4 scored **899.8281 ±490.4217**
on all 64 repeated training holdouts (sample standard deviation), below
baseline **970.125 ±537.5760**. It had 63 deaths and 1 timeout versus
baseline 62 deaths and 2 timeouts. The paired change was **−70.2969** steps,
with whole-game bootstrap 95% interval **[−174.0785,+27.8910]**. This
conditions on the fixed checkpoints and repeated training cohort; it excludes
training/checkpoint/sequential selection uncertainty and is not fresh validation.
The baseline remains best. CPU replay verified all **640 completed training
records**, four populations/rankings, selected best and current mean weights,
history, and the pending fifth population and RNG. Evidence:
`artifacts/doom_reference_round14_g004_training_prefix_cpu_audit.json` and
`artifacts/doom_reference_round14_g004_holdout_comparison_cpu.json`.
Generation 5 is running in verified live session 49443. The registered search
continues to generation 8 before fresh validation; reserved testing has not
started, and the 1092-step target remains unmet.

At 2026-10-04T12:43:34.778264+00:00, round 14 completed generation 5. CPU prefix replay
verified **768 completed training-game records**, five populations/rankings,
saved mean/best/final weights and history, and pending sixth population/RNG.
Evidence: `artifacts/doom_reference_round14_g005_training_prefix_cpu_audit.json`.
Generation 6 is running in verified live session 49443. The baseline remains
best at **970.125** on repeated training holdouts; the next holdout comparison
is at generation 8. Fresh validation and reserved testing remain pending,
so the 1092-step objective is unproven.

At 2026-10-04T12:56:29.454931+00:00, round 14 completed generation 6. CPU prefix replay
verified **896 completed training-game records**, six populations/rankings,
saved mean/best/final weights and history, and pending seventh population/RNG.
Evidence: `artifacts/doom_reference_round14_g006_training_prefix_cpu_audit.json`.
Generation 7 is running in verified live session 49443. The baseline remains
best at **970.125** on repeated training holdouts; the next comparison is at
generation 8. Fresh validation and reserved testing remain pending. The
1092-step objective is unproven.

At 2026-10-04T13:13:29.223055+00:00, round 14 completed generation 7. CPU prefix replay
verified **1024 completed training-game records**, seven populations/rankings,
saved mean/best/final weights and history, and pending eighth population/RNG.
Evidence: `artifacts/doom_reference_round14_g007_training_prefix_cpu_audit.json`.
The final generation 8 is running in verified live session 49443, followed
by its 64-game training holdout and independent complete capsule audit. The
baseline remains best at **970.125**. Actual supervisor exit and current audit
checks are required before the separately prepared fresh validation and any
eligible reserved pair. The 1092-step objective remains unproven.


Round 14 training completed and its supervisor session 49443 exited with code 0
at 2026-10-04T13:31Z. Independent current CPU replay verified all **1216 training
games**, eight CMA populations/rankings, selected/final weights, complete
history and saved optimizer/RNG. The parent, trainer and auditor had exited;
the GPU lease was available before the separate fresh evaluation dispatch.
The final generation 8 became best on the repeated 64-game training holdout:
**1002.0781 ±577.5202** steps (sample SD), with 58 deaths and 6 timeouts,
versus baseline **970.125 ±537.5760**, 62 deaths and 2 timeouts. Paired change
was **+31.9531**, bootstrap 95% interval **[−94.2813,+153.4691]**. This interval
conditions on these fixed checkpoints and the repeated training cohort; it
excludes training, checkpoint and sequential-selection uncertainty and does
not establish fresh performance or the 1092-step target. Evidence:
`artifacts/doom_reference_round14_training_cpu_audit.json`,
`artifacts/doom_reference_round14_training_exit_review_cpu.json`, and
`artifacts/doom_reference_round14_final_holdout_comparison_cpu.json`.

The prepared v2 fresh comparison passed actual CPU readiness and CUDA preflight
and started as supervisor session **62687**, parent **1563012**. Three distinct
policies require all 80 validation games each on seeds 130660–130739: public,
prior round 8, and new round 14 generation 8. Best/final raw weights are identical
and therefore evaluated once. Only a new validation winner, frozen and checked
by the independent CPU choice audit, may proceed to its reserved 100 games and
the highest-validation unchanged control on the same seeds 230000–230099.
No new training is queued. This trains a controller on real survival with an
imported public world, which differs from the paper's dream-only controller
training and does not reproduce our own world-model training. Original
checkpoints and the own-world incumbent score 840.06 ±524.48 are preserved.


Round 14 fresh validation completed all **240 games** on seeds 130660–130739.
The public control scored **972.575 ±509.5249**, the prior round 8 controller
**979.0875 ±570.5231**, and the new round 14 generation 8 controller
**921.525 ±521.1782** steps (population SD). New versus prior was
**−57.5625**, whole-game paired validation bootstrap 95% interval
**[−190.5878,+73.5631]**. The prior controller was retained after complete
validation; generation 8's observed training-holdout gain did not transfer to
this fresh cohort. The interval does not establish population degradation,
and conditions on the fixed policies while excluding training and sequential
selection uncertainty. No policy was selected or retrained using test data:
the reserved range 230000–230099 remains unused because no new policy won.

The actual real supervisor session **62687** exited with code **0** (tool
`2806f9`). CPU reconstruction rechecked the full result, frozen selection,
current source/input/runtime/checkpoints/training capsule and all raw records.
Known training/evaluation/auditor processes and game workers had exited; the
repository GPU lease was available and no additional GPU job was queued.
Evidence: `artifacts/doom_reference_round14_real_result_cpu_audit.json`,
`artifacts/doom_reference_round14_completion_cpu_audit.json`, and
`artifacts/doom_reference_round14_controller_geometry_cpu.json`.
The 1092-step goal remains unmet. The next hypothesis is a bounded coherent
controller-gain search from the retained prior weights, with fixed world and
inference, fresh training/validation seeds and reserved testing only after an
independently audited new validation winner. It has not been dispatched or
shown to improve performance. Original checkpoints and the canonical
own-world incumbent remain preserved; imported-world direct-real controller
training remains a protocol departure from the paper.


The next CPU prototype projects three log gains onto the retained prior
controller's z/c/h weight blocks (64/512/512), preserving the zero point's
raw FP64 weights exactly and the original singleton inference computation.
Four CPU projection tests passed, including block order, bounds, invalid
weights and overflow rejection. Recorded prior validation chose wait on only
**0.2119%** of steps; relative block gains can affect left/right sign as well
as action magnitude. Their raw action still feeds the fixed RNN, so this is
not merely an environment threshold adjustment. These are implementation
facts and a search hypothesis, not evidence of better real survival.
`artifacts/doom_reference_round15_subspace_design_cpu.json` is a **draft**:
the resumable projected CMA trainer, full independent optimizer replay and
fresh comparison gates must be implemented and verified before registration
and CUDA dispatch. Proposed budget is four generations, population 8, sixteen
fitness games, 64 training holdouts, three complete 80-game validation cohorts
and a conditional 100+100 reserved pair (at most 1080 games). Proposed fresh
seed ranges were checked unused. No GPU work has been dispatched.


Round 15 was registered and started on CUDA at 2026-10-04T14:29Z after
**32 passing CPU tests**, actual initializer/trainer readiness and launcher
source/input/runtime/process/lease/fresh-seed checks. Protocol:
`artifacts/doom_reference_round15_subspace_protocol.json`
(SHA `a91688c85b27913bb7368ce440e988869af74b82b5b02def90330d8dac92af67`).
It optimizes three bounded log gains (z/c/h blocks) from the retained prior,
with seed 96, sigma 0.25, four generations, population 8, sixteen common
fitness games per candidate per generation, and 64 baseline/final training
holdouts. The world, preprocessing, architecture and singleton inference
math remain fixed. These combined hyperparameter changes do not isolate a
cause of improvement. Best and final projected FP64 controller weights,
log gains and complete three-dimensional optimizer/RNG/history/raw games are
retained; CPU replay reconstructs the bounded phenotype mean, all populations,
projection, covariance/adaptation and next RNG before fresh evaluation.

Actual supervisor session **87997**, parent **1599849** and trainer **1599915**
were verified live; CUDA matrix multiplication and convolution backward
preflight passed on the RTX 4070 Ti. Initial immutable optimizer snapshot,
raw zero-gain prior identity, first eight-candidate population and RNG were
independently reconstructed on CPU. Evidence:
`artifacts/doom_reference_round15_initial_capsule_cpu_audit.json` and
`artifacts/doom_reference_round15_training_preparation_cpu.json`.
The initial 64-game baseline is still running; no complete survival result or
fresh validation has been claimed. Estimated complete-loop duration is two
to three hours, dependent on actual game lengths and compilation. Training
uses up to 640 games, followed by up to 240 fresh validation games and a
conditional 200 reserved candidate/control games. Validation seeds
130740–130819 and test seeds 240000–240099 remain separated from training;
only a distinct new full-validation winner can consume the reserved pair.
The goal remains unmet and original checkpoints/canonical incumbent remain
preserved. Historical round 14 sessions are closed and must not be restarted.


At 2026-10-04T14:40:01.346943+00:00, round 15 completed its fresh 64-game training baseline:
**853.8750 ±558.7786** steps (sample SD), with
62 deaths and 2 timeouts. CPU reconstruction verified every raw
record, seed/settings/initializer identity, generation-zero best/final projected
weights and gains, and the pending first3D population and next RNG. Evidence:
`artifacts/doom_reference_round15_baseline_cpu_audit.json`. The eight-candidate
first population is running in independently verified live session87997. This
baseline uses new training seeds and is not fresh validation or the reserved
100-game result. A lower or higher baseline mean on a different cohort does not
by itself establish a policy change. The1092-step goal remains unmet.


At 2026-10-04T14:55:58.708133+00:00, round 15 completed generation 1 and began generation 2.
CPU replay verified all **192 completed training games**, reconstructed the
first population and raw fitness means, three-dimensional covariance/adaptation
and optimizer mean, best/current projected1088-weight checkpoints and history,
and the pending second population and next RNG. Current mean feature gains
(z/c/h) are **0.9137/1.0923/0.9410**.
These are training updates, not fresh survival validation. Best remains the
baseline853.875 until the scheduled final64-game holdout at generation4.
Evidence: `artifacts/doom_reference_round15_g001_training_prefix_cpu_audit.json`.
Its hashes describe the saved files at that check; optimizer snapshots and
completed raw cohorts are immutable while checkpoint/pointer/history files
advance. Existing session87997 and exact parent/trainer commands were verified
live. Fresh validation and reserved testing remain pending; goal completion
is unproven.
