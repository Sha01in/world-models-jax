# VizDoom vision experiment

The completed vision comparison tested whether the reference VAE improves useful
visual features. Matched fresh downstream models and the subsequent frozen-RNN
transfer comparison are also complete. This effort stops on observed diminishing
returns; the paper's real-game mean remains unmatched and all GPU work has exited.
The preceding paired controller test
showed +47.10 steps with a 95% interval of -51.60 to +146.61; a reliable gain
has not been established. The target remains the original paper's 1092 mean
over 100 real games. Reconstruction and dream scores are intermediate evidence.

## Current evidence and resource state

The VAE comparisons, full encodings, two fresh RNNs, fixed death audits and six
matched 500-generation controller searches and all real evaluations are complete.
Real evaluation started on 2026-10-03 at approximately 14:22 UTC, after another
successful CUDA matrix and convolution-gradient check. Its terminal session was
36447 and its status is `artifacts/doom_vision_round3_real_evaluations.status`.
It compared 14 fixed policy/inference combinations on 100 common validation
games, retained the original incumbent, and completed both reserved paired
reports. The new pipelines did not improve survival against the original
incumbent. The subsequent four-search frozen-RNN comparison and its reserved
tests are complete too. All GPU jobs have exited, no new work is queued, and
the five-minute follow-up is paused on completion. The deliberately stopped
packed training was never resumed.

The frozen incumbent VAE reconstructs large approaching projectiles in a
48-frame atlas from twelve newer held-out fatal episodes. Mean-latent pixel MSE
was 0.0026013 and sampled-latent MSE was 0.0028844 on normalized RGB. This
small fatal-window sample does not establish general object retention.

A fixed linear ridge probe used the same VAE's recorded means from 900 training
episodes / 57,600 frames and 100 separate newer holdouts / 1,600 frames. Targets
were bright yellow color components, **not human-verified fireball labels**.
Held-out AUC was 0.8872 for larger components and 0.7094 for one/two-pixel
components. Horizontal position MAE for the largest component was 8.56 pixels,
versus 14.71 for a constant training-mean predictor; vertical MAE was 1.09 versus
1.59. Hyperparameters and standardization used training data only.
These results motivate checking smaller/earlier visual cues; they do not prove
that the VAE is the gameplay bottleneck. Component reconstruction thresholds
are particularly sensitive to blur and should not be interpreted as detection
recall for actual fireballs.

Resampling all frames of each of the 100 held-out episodes together gives
95% percentile intervals of **0.8698–0.9042** for larger-component AUC and
**0.6790–0.7391** for small-component AUC (5,000 resamples, seed 73501).
Horizontal position MAE is **8.06–9.09 pixels**; its paired error reduction
relative to the constant predictor is **6.15 pixels [5.34, 6.95]**. These
intervals condition on the fitted probe and fixed color targets; they do not
include variability from training or label quality. Per-frame predictions,
episode identities and frame indices are saved for paired architecture
comparisons rather than treating 1,600 correlated frames as independent games.

Artifacts:

- `artifacts/doom_vision_incumbent_cpu/report.json` and `atlas.png`.
- `artifacts/doom_vision_incumbent_linear_probe.json`.
- `artifacts/doom_vision_incumbent_episode_bootstrap.json` and its matching
  `_predictions.npz` file.
- `artifacts/doom_vision_incumbent_cpu/bright_component_proxy.json`.

## Immutable data and architectures

`artifacts/doom_vision_round3_split.json` maps the preceding verified RNN split
back to raw archives using transition signatures and file fingerprints. All
4,430 selected episode groups have an unambiguous raw match, with no overlapping
groups across training/validation. The split has 3,460 training episodes /
1,373,500 frames and 970 validation episodes / 317,522 frames. Training has 30
timeout labels; all 970 validation episodes have fatal labels. This validation
set consequently cannot assess whole-episode timeout behavior.

Split SHA256: `c625cd847aaffc4d07b68c8041fcc019e137c601c87822d3146c4da5ccc50751`.

The shared uint8 frame pool is `artifacts/doom_vision_round3_frames_s73/`:
221,440 training frames (64 uniform samples per episode), 15,520 validation
frames (16 per episode), seed 73. Sampling is fixed across architectures and
epochs, without replacement within each episode. Its metadata fingerprints
both arrays. This is an initial training-budget comparison, not full-dataset
training. If both learning curves are still improving at the budget limit,
undertraining must be investigated before declaring diminishing returns.

| Variant | Encoder spatial sizes | Decoder spatial sizes | Parameters, latent width 64 |
| --- | --- | --- | ---: |
| Current architecture | 64 → 32 → 16 → 8 → 4 | 4 → 8 → 16 → 32 → 64 | 2,170,691 |
| Reference architecture | 64 → 31 → 14 → 6 → 2 | 1 → 5 → 13 → 30 → 64 | 4,446,915 |

The reference implementation uses valid encoder convolutions and a dense
1024-wide decoder reshaped to 1×1×1024. The loss is pixel-summed squared error
plus KL, with a floor of 0.5 times latent width per image. The existing loss
already has this scaling; no KL-weight change is part of the architecture test.
Source: [original ConvVAE](https://github.com/hardmaru/WorldModelsExperiments/blob/master/doomrnn/doomrnn.py).

Both variants retain existing full-frame RGB64 preprocessing to isolate
architecture. The authors crop the native frame before resizing and invert
colors; that is a separate comparison requiring suitable native observations,
and must not be mixed into this first experiment. Source:
[original real-game preprocessing](https://github.com/hardmaru/WorldModelsExperiments/blob/master/doomrnn/doomreal.py).

## GPU comparison

Verify CUDA first. Run one architecture at a time, each initialized from scratch
with seed 73, learning rate 0.0001, batch 128, at most 20 epochs, validation every
epoch and patience five. The untrained epoch-zero checkpoint is eligible.
Retain best and final weights, matching Adam states, RNG, history and settings.
Use fresh output directories; the incumbent is never a training output.

The current VAE completed epoch 20 with best held-out loss 45.2557 (34,600 Adam
updates, 674 elapsed seconds). Epochs 15 to 20 still improved loss by 1.75%, so
this cap does not establish convergence. `--resume-dir` can continue a completed
budget-limited run into a fresh directory with a larger **total** epoch cap.
It verifies bundle fingerprints, immutable settings, Adam count, RNG and the
saved validation value; it retains prior best/patience history. It refuses
automatically continuing an early-stopped run. Source checkpoints remain intact.

Both initial CUDA runs completed 20 epochs with best/final Adam counts 34,600
verified from the actual serialized states. The reference VAE's best loss was
46.0313 at epoch 20, versus the current VAE's 45.2557. Reference elapsed time
was 800 seconds. Both were still improving over epochs 15–20 (current 1.75%,
reference 1.84%); neither reached the patience stop. These are single-seed,
equal-presentation pilots, not a general verdict about architecture.

A bounded serial continuation started at 11:44 UTC on 2026-10-03, preserving
both 20-epoch source directories. It allows at most 60 **total** epochs per
architecture with the same settings and patience. Its status, manifest and
per-architecture logs are `artifacts/doom_vision_round3_vae_extension.status`,
`artifacts/doom_vision_round3_vae_extension_jobs.json`, and
`artifacts/doom_vision_round3_vae_*_e60.log`. The current architecture runs first,
then the reference. CUDA is checked before each training job. No downstream
RNN/controller training has started yet.

The current architecture completed 60 total epochs: best epoch 60, held-out
loss **42.6414**, reconstruction **10.5411**, 103,800 optimizer updates. Its
epochs 55–60 still improved total loss by 0.316%, so this remains a bounded
training result rather than proof of convergence. The reference continuation
also completed 60 epochs: best loss 43.4174, reconstruction 11.3472, and 103,800
optimizer updates. Its last five epochs improved loss by 0.399%. The actual
selected/final optimizer counts and RNG bundles were verified on CPU, and all
incumbent fingerprints remained unchanged. The user explicitly confirmed GPU
availability.

```bash
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
.venv/bin/python scripts/tools/check_gpu.py

.venv/bin/python train_vae.py \
  --split artifacts/doom_vision_round3_split.json \
  --frame-cache-dir artifacts/doom_vision_round3_frames_s73 \
  --architecture current --seed 73 --epochs 20 --patience 5 \
  --output-dir checkpoints/VizdoomTakeCover-v0/vision_round3_current_s73

.venv/bin/python train_vae.py \
  --split artifacts/doom_vision_round3_split.json \
  --frame-cache-dir artifacts/doom_vision_round3_frames_s73 \
  --architecture paper --seed 73 --epochs 20 --patience 5 \
  --output-dir checkpoints/VizdoomTakeCover-v0/vision_round3_paper_s73
```

Repeat the same atlas on each held-out VAE with `audit_doom_vision.py`. Encode
the explicit split with `encode_vae_split.py`; it writes separate `training/`
and `validation/` directories and records VAE/raw/split fingerprints. Repeat
the fixed probe using the new encodings. Keep each `vae.eqx.json` sidecar with
its weights when copying or selecting a model.

For the initial feature comparison, `encode_vae_frame_features.py` encodes just
the existing sampled pools (236,960 frames total), rather than re-encoding
1,691,022 full-episode frames first. It writes fingerprinted float16 means in
the same row order as the immutable image pools; these are probe inputs, **not
RNN trajectories**. `probe_doom_vision.py --encoded-frame-cache ...` accepts
them. All three frozen VAEs, including the incumbent, will use encoder batch
128 and identical frames/targets for this comparison. The incumbent's original
probe remains preserved as a separate diagnostic.

The completed `artifacts/run_vision_round3_diagnostics.py` started after
the continuation completed. It used the same exclusive GPU-work lock,
verifies both model/Adam/RNG bundles on CPU, then runs serial CUDA feature
encoding and CPU probes/atlases/paired comparisons. It recovers completed
matching reports and preserves failed/partial outputs. It does not start RNN
or controller training. The reviewed evidence below determines the downstream
candidates. Full-episode `encode_vae_split.py` remains required for RNN inputs.

Changing the VAE changes latent coordinates. Train a fresh factorized MDN-RNN
for each selected VAE (the frozen incumbent and current epoch-60 VAE), using
the encoded training/validation directories,
posterior inputs, the existing terminal alignment, 512 memory units and
positive-death weight 10. Start with equal maximum 30-epoch budgets, learning
rate 0.001, batch 32, seed 74 and patience three; retain baseline, best, final
weights and corresponding optimizer states. Freeze the held-out best RNN
beside its matching VAE before controller optimization. Do not resume an old
RNN in the new latent space or compare raw NLL across different latent spaces
as a gameplay improvement.

Keep the linear controller architecture and dream temperature 1.15 fixed.
Use threshold termination for both worlds, 500 CMA generations, population 64,
16 rollouts and candidate batch 64. Run three matched CMA seeds (74, 75, 76),
with fresh, matched initialization; old controller features have also changed
coordinate systems. Compare both posterior and mean real inference.

Use 100 real validation games (proposed fresh seeds 60000–60099), including the
unchanged incumbent. Freeze one validation-selected winner before evaluating
100 new reserved games (proposed seeds 80000–80099), paired with the unchanged
incumbent. Verify that proposed seeds remain unused before dispatch. Do not
select/retrain from reserved test outcomes. Report mean, SD, paired improvement
and uncertainty, along with remaining differences from the
[paper](https://worldmodels.github.io/).

## Verification so far

The legacy incumbent VAE loads and serializes to exactly the original bytes.
The reference geometry and parameter count passed CPU checks. A four-episode
CPU smoke with two optimizer updates verified finite loss, selected/final
bundles, architecture metadata and matching final Adam count. These smoke
weights are test artifacts, not a scientific training result. Regression
coverage checks loss scaling/padding masks, episode alias exclusion, cache
identity/content, best checkpoint patience and encoding of the fatal action.
No new real-game result is available for this slice. The current architecture's
first nine epochs took 287 seconds including initial compilation and validation;
held-out loss was 47.895 at epoch nine. This is an early training measurement,
not a completed architecture comparison or gameplay result.
Four additional CPU tests verified whole-episode resampling, unchanged intervals
when correlated frames are duplicated, shared indices for paired comparisons,
and explicit missing-statistic handling. The probe also accepts the nested
training/validation directory layout produced by the new encoder.

Two continuation checks passed, and a real CLI CPU smoke produced byte-identical
best/final weights and Adam states to uninterrupted four-update training, with
the same RNG and final optimizer count four. All source files were unchanged.
Ruff passed the continuation code. The smoke is not scientific training.

The completed current VAE's identical 12-episode / 48-frame atlas had mean pixel
MSE 0.002154 and sampled MSE 0.002432. Paired whole-episode bootstrap gives a
mean-MSE reduction of **17.2% [9.9%, 22.9%]** against the incumbent and sampled
reduction of **15.7% [5.3%, 23.3%]**. Large fireballs remain visible, while small
ones remain blurred. This small fatal-window reconstruction comparison does
not establish object retention, architecture superiority or real survival;
the incumbent also has a different training history. Artifacts:
`artifacts/doom_vision_current20_cpu/` and
`artifacts/doom_vision_current20_paired_audit.json`.

Reference epoch-20 audit MSE was 0.002502 for mean latents and 0.002600 for
posterior samples. On the same 12 episodes / 48 frames, reference mean error
was 16.2% higher than current [13.2%, 19.8%] and sampled error 6.9% higher
[2.4%, 11.0%]. This comparison holds the two trained VAEs fixed, covers a small
fatal-window sample, and does not establish a difference in projectile features
or survival. Both atlases were inspected. Artifacts:
`artifacts/doom_vision_paper20_cpu/` and
`artifacts/doom_vision_architectures20_paired_audit.json`.

The current epoch-60 atlas was also inspected: large projectiles remain visible
and small/early cues remain blurred. Its mean MSE is **0.0019467**, sampled MSE
**0.0021704**. Against the incumbent on the same twelve fatal episodes / 48
frames, paired mean-error reduction is **25.2% [19.4%, 30.1%]**, sampled
reduction **24.8% [16.2%, 31.2%]**. These are whole-image reconstruction
statistics, not projectile recall or real-game gains. Artifacts:
`artifacts/doom_vision_current60_cpu/` and
`artifacts/doom_vision_fast_incumbent_vs_current60_audit_comparison.json`.

Fourteen affected CPU checks passed, including exact equality of the sampled
feature cache and full episode encodings/probe results, cache recovery and
fingerprint/calculation-shape rejection. A separate paired-probe check verified
known AUC/position-error gains and rejection of mismatched or duplicate frames.
Ruff passed the changed feature code, and the diagnostic supervisor parses.

## Completed feature comparison and downstream decision

Both VAEs completed 60 epochs. Frozen diagnostics completed on 2026-10-03 at
12:35 UTC with matching encoder batch 128, float16 means and exactly the same
sampled frames and train-only probes for all three VAEs. The incumbent remains
unchanged. These targets are color components, not verified fireball labels.

| Fixed VAE | Larger-component AUC | Small-component AUC | Horizontal MAE, pixels |
| --- | ---: | ---: | ---: |
| Incumbent | 0.8873 | 0.7094 | 8.56 |
| Current, epoch 60 | 0.8787 | 0.7212 | 9.16 |
| Reference geometry, epoch 60 | 0.8298 | 0.6714 | 9.42 |

Against the incumbent, current's larger-component AUC difference is -0.0086
[95% paired interval -0.0236, +0.0071], and its small-component difference is
+0.0119 [-0.0120, +0.0334]. Neither establishes a feature gain. Horizontal
error increases 0.60 pixels [0.17, 1.01]. Reference geometry reduces larger AUC
by 0.0575 [0.0372, 0.0780] and small AUC by 0.0379 [0.0126, 0.0639]; horizontal
error increases 0.86 [0.38, 1.33]. Intervals resample 100 whole holdout episodes
5,000 times, holding fitted VAEs/probes fixed. They exclude training variability.

The reference epoch-60 atlas was inspected: approaching large projectiles are
visible, but small early cues remain blurred. Mean MSE is 0.0022373 and sampled
MSE 0.0023822, respectively 14.9% and 9.8% higher than current on the same
48-frame fatal-window sample. The reference geometry alone is not promoted to
downstream training in this iteration. Both use Equinox default initialization
and existing full-frame preprocessing; this is not an exact original-paper VAE
replication or a general architecture conclusion.

Current's better reconstruction and uncertain classification changes justify
one bounded survival comparison, not further VAE extension on pixel loss. The
downstream comparison is now **fixed incumbent VAE versus current epoch-60 VAE**.
Both receive fresh RNNs and fresh matched controllers on identical episodes,
seeds and training budgets. This control separates the fixed VAE difference
from downstream retraining. The original incumbent policy remains a separate
real-game performance control. No RNN/controller weights cross latent bases.

`artifacts/run_vision_round3_rnns.py` serially encodes both full episode splits,
verifies episode lineage, content fingerprints, fatal/timeout labels and split
separation, then trains the two fresh RNNs with the settings above. It uses the
shared exclusive lock and CUDA preflight before each GPU stage. An interrupted
RNN requires explicit recovery; the script refuses blindly overwriting its
directory. The CPU bundle verifier reads actual serialized Adam counts and RNG
states, checks best/final selection against history and freezes the selected
world beside the correct VAE. No controller search starts automatically.

Artifacts: `artifacts/doom_vision_round3_diagnostics_result.json`, the three
`doom_vision_fast_*_probe.json` reports and paired comparisons,
`artifacts/doom_vision_round3_vaes60_bundle_verification.json`, and
`artifacts/doom_vision_round3_rnn_jobs.json`. No new survival result exists yet.

Both full-episode encodings completed and passed lineage/content checks: each
has 3,460 training episodes / 1,373,500 frames (3,430 fatal, 30 timeout) and 970
held-out episodes / 317,522 frames (all fatal). The two fresh RNNs subsequently
completed serial CUDA training. These inputs retain every
observation/action/outcome row, including the fatal action, unlike the sampled
frame-only probe caches.

The follow-up `artifacts/run_vision_round3_holdout_audits.py` is prepared but
must wait for both frozen RNNs. It compares 256 fixed historical holdouts and
all 100 newer holdouts, in posterior and mean modes, for the fresh control,
fresh candidate and original incumbent RNN. Comparisons retain whole-episode
detection counts and paired uncertainty, with identical sampling order and
labels. New audits match `train_rnn.load_batch` float32 posterior inputs/noise;
historical audits rounded samples to their float16 storage dtype. Existing
reports remain intact, and `--legacy-float16-posterior` reproduces the old
calculation when explicitly needed. New versus old numerical audit results
must not be compared without identifying this protocol difference. Two CPU
checks verify exact training-input equality and known paired death metrics,
including rejection of mismatched labels/duplicate episodes.

`artifacts/run_vision_round3_controllers.py` is also prepared. After reviewing
the actual frozen-RNN audit evidence, record that decision in
`artifacts/doom_vision_round3_controller_review.json` with the audit-result
fingerprint and `proceed_with_matched_controller_comparison`. This is an agent
experiment-review gate within the authorized work, not a new user approval.
The runner alternates incumbent/current worlds for CMA seeds 74, 75 and 76,
uses one GPU job at a time, and checks complete 500-generation histories and
selected controller provenance. No search is started while RNN training runs.

A CPU seed audit found no overlap for proposed validation 60000–60099 or
reserved test 80000–80099 among 50 recorded real-game/collection reports and
1,460 distinct recorded seeds. The audit must repeat immediately before real
evaluation; it does not cover unrecorded external runs or consume these seeds.
Artifact: `artifacts/doom_vision_round3_real_seed_audit.json`.

## Completed fresh RNNs and death audits

Both RNNs stopped after three epochs without held-out improvement. Actual
serialized best/final Adam counts, RNG states, history and source fingerprints
were verified on CPU; selected worlds contain the correct VAE and best RNN.
Final weights and optimizer states remain separate and intact.

| Fixed VAE | Best RNN epoch / loss / Adam count | Final epoch / loss / Adam count |
| --- | --- | --- |
| Original incumbent | 15 / 1.087586 / 1,635 | 18 / 1.092155 / 1,962 |
| Current epoch 60 | 18 / 1.083214 / 1,962 | 21 / 1.085893 / 2,289 |

NLLs use different learned latent spaces and do not establish a cross-world
performance ranking. The selected RNN SHA256 values are
`deac21c7dace5715e2ba259071f16a5ad5ded03cf74815a770440f347e8ffb1c`
(incumbent VAE) and
`bea6e56ac74bf8f80683bcd7485df8bb2820310954caf95e9593816ec6ecc665`
(current VAE). Artifact: `artifacts/doom_vision_round3_rnns_result.json`.

Twelve fixed audits and twelve paired comparisons completed, using 256
historical and all 100 newer holdouts, both mean and posterior inference. The
posterior results below all use the corrected float32 protocol and the same
episode order; they are not direct comparisons with legacy float16 reports.

| RNN world | Historical death recall / false death rate | Newer recall / false death rate |
| --- | --- | --- |
| Unchanged original incumbent | 70.70% / 0.3542% | 45% / 0.1740% |
| Fresh RNN, incumbent VAE | 66.41% / 0.5513% | 52% / 0.3566% |
| Fresh RNN, current VAE | 60.16% / 0.3627% | 38% / 0.2110% |

Current versus fresh control loses 6.25 percentage points of historical recall
(95% paired interval -10.94 to -1.56) and 14 points on newer holdouts
(-23 to -5), while reducing false alarms. Intervals resample whole episodes
5,000 times, holding trained models fixed. Weighted-BCE scores at threshold
0.5 are not calibrated death probabilities. The regression is concerning,
but real survival remains the endpoint. The recorded review proceeds with
the bounded matched controller comparison and preserves the original control.
Artifacts: `artifacts/doom_vision_round3_holdout_audits_result.json` and
`artifacts/doom_vision_round3_controller_review.json`.

## Completed controller searches and running real evaluation

All six searches completed 500 generations at temperature 1.15, threshold
termination, population 64, 16 rollouts and candidate batch 64. CMA seeds
74/75/76 and fresh initialization are matched across the two latent spaces;
no controller weights cross those spaces. Selected policies, settings and
complete histories were fingerprinted. The controller has 1,089 parameters
including its bias. Best fixed dream-validation scores were 1,802.80/1,810.83/
1,804.34 for the incumbent-VAE world and 982.64/1,054.92/1,096.17 for current.
Scores from different dream worlds are not evidence of real survival gains.
Artifact: `artifacts/doom_vision_round3_controllers_result.json`.

The 1,089-parameter affine controller is a retained protocol difference: the
authors' Doom controller uses only its 1,088 input weights and no learned
bias. This was confirmed in the primary
[controller implementation](https://github.com/hardmaru/WorldModelsExperiments/blob/master/doomrnn/model.py).
Current candidates remain fixed; changing this during their evaluation would
confound the comparison. The difference alone does not explain the transfer
gap or establish that removing the bias would improve survival.

The real supervisor evaluates six policies plus the unchanged original
incumbent, each with mean and posterior inference, on seeds 60000–60099.
Eight game workers preserve single-game inference shapes and per-seed RNG.
The pre-dispatch audit again found no overlap with recorded prior games or
collection ranges; the reservation fingerprints the controller manifest.
It cannot account for unrecorded external games.

Selection uses only the highest validation mean across those 14 combinations.
The selected world and inference mode, all validation-report fingerprints and
a timestamp must be frozen before any reserved test. Only that frozen winner
and the unchanged mean-inference incumbent run seeds 80000–80099. Test outcomes
cannot influence selection or retraining. Matching completed reports are
verified and recovered without overwrite after an interruption.

Two CPU fixture checks verified report/seed guards, and two preserved real
reports (200 games) independently passed the same weight, inference, seed,
reward, action-count and summary checks without rerunning games. The prepared
CPU summary computed mean, population SD and paired whole-game bootstrap
uncertainty after both reserved reports completed. Artifacts:
`artifacts/run_vision_round3_real_evaluations.py`,
`artifacts/doom_vision_round3_real_seed_reservation.json`,
`artifacts/doom_vision_round3_evaluation_cpu_verification.json`, and
`artifacts/summarize_vision_round3_paired_results.py`.

## Completed real survival comparison

Each entry below is the mean over the same 100 validation seeds 60000–60099.

| World / CMA seed | Mean-latent inference | Posterior inference |
| --- | ---: | ---: |
| Fresh RNN, incumbent VAE / 74 | 240.55 | 231.88 |
| Fresh RNN, incumbent VAE / 75 | 193.13 | 195.36 |
| Fresh RNN, incumbent VAE / 76 | 174.49 | 175.12 |
| Fresh RNN, current VAE / 74 | 410.30 | 409.55 |
| Fresh RNN, current VAE / 75 | 467.28 | 467.51 |
| Fresh RNN, current VAE / 76 | 441.54 | 450.93 |
| Unchanged original incumbent | **793.16** | 775.04 |

All twelve new combinations underperformed the predetermined original
mean-inference control. The best new combination's paired validation difference
was -325.65 steps, ordinary 95% whole-game interval [-422.56, -234.02]. A
Bonferroni-adjusted percentile interval for twelve comparisons is
[-467.20, -192.27]; every new combination's adjusted upper bound is negative.
These fixed-policy bootstrap intervals approximate simultaneous coverage and
exclude training variability. No reserved test outcomes enter this analysis.
Artifact: `artifacts/doom_vision_round3_validation_comparison.json`.

Validation selected the original incumbent with mean inference. It was copied
and frozen before test seeds 80000–80099 were dispatched; the copied controller
parameters were independently verified identical, despite changed selection
metadata in its archive. Both selected and unchanged copies scored
**817.91 ± 524.13** (population SD), with mean bootstrap interval
**[717.39, 922.52]**. Every paired episode tied: improvement **0.00**, interval
**[0.00, 0.00]**. This is a repeated baseline, not a new model success or evidence
that untested alternatives have zero uncertainty. The paper's reported mean
1092 is still not matched. The earlier 840.06 cohort is preserved and uses
different seeds. Test seeds 80000–80099 are now consumed and cannot be reused
for tuning. Artifact: `artifacts/doom_vision_round3_paired_comparison.json`.

The iteration improved pixel reconstruction but regressed real survival after
matched fresh downstream training. High dream scores did not transfer, which
is consistent with the model-exploitation issue discussed by the paper. It
does not prove a particular failure mechanism. Extending these VAEs on pixel
loss alone is not supported by the measured gameplay results.

## Next bounded slice: preserved RNN transfer

The preserved packed run's best RNN has never had a controller transfer test.
Its epoch-one weights and metadata still match the frozen early-stop record;
the deliberately stopped run will not be resumed. A four-search pilot compares
this frozen RNN with the unchanged incumbent RNN, keeping the original VAE,
affine controller architecture and tau 1.15 fixed. Both start from the same
incumbent controller and receive 500 CMA generations, population 64, 16
rollouts, candidate batch 64 and seed 77, with threshold and sampled termination
as a two-by-two comparison. Sampled termination undoes the BCE odds weighting;
that correction alone does not guarantee calibration.

The RNNs have different training histories, so this tests available frozen
worlds and does not isolate packing or gradient clipping alone. The packed
checkpoint used sequence 500, batch 100, per-value clipping and learning-rate
decay, closer to the authors' training procedure. Sources:
[original trainer](https://github.com/hardmaru/WorldModelsExperiments/blob/master/doomrnn/rnn_train.py)
and [original RNN](https://github.com/hardmaru/WorldModelsExperiments/blob/master/doomrnn/doomrnn.py).
The best packed checkpoint has no saved best optimizer; it is an evaluation
source, not a resumable training bundle. Its recorded global step is 9,239.

The pilot uses one GPU job at a time and fresh output directories. Its review
gate fingerprints the completed vision paired summary. Proposed validation
seeds are 90000–90099, proposed reserved test seeds 110000–110099; a CPU check
found no recorded overlap, and the audit must repeat before dispatch. Four
policies plus the original control, each under mean and posterior inference,
receive 100 common validation games. Freeze the validation winner before its
100-game paired test against the unchanged incumbent. This single-CMA-seed
pilot excludes training variability and does not establish performance until
the actual games complete. All original checkpoints remain intact.

Prepared launchers: `artifacts/run_frozen_transfer_round4_controllers.py` and
`artifacts/run_frozen_transfer_round4_real_evaluations.py`. Controller definitions
and evaluator imports/seed boundaries passed CPU checks; Ruff passed. The
evaluator uses the exact already-verified report guard. Preparation itself
ran no new GPU jobs or games.

After reviewing the completed real reports, the four-search pilot started at
approximately 14:58 UTC on 2026-10-03, session 44487. CUDA preflight passed,
and its supervisor and first controller child were verified live. Follow
`artifacts/doom_frozen_transfer_round4_controllers.status` and
`artifacts/doom_frozen_transfer_round4_controllers.log`; no duplicate training
or controller supervisor is authorized. Real evaluation follows only after
all four complete histories and frozen source checks pass.

All four searches completed 500 generations at 15:14 UTC. Selected policies,
settings and complete histories passed verification. Best fixed dream-validation
scores were 1,010.09 (incumbent threshold), 1,076.94 (packed threshold),
1,234.58 (incumbent sampled) and 1,164.33 (packed sampled); these are simulation
scores and establish no real improvement. Artifact:
`artifacts/doom_frozen_transfer_round4_controllers_result.json`.

The single real-evaluation supervisor started at approximately 15:17 UTC,
session 37461. CUDA matrix/convolution-gradient preflight passed, and both the
supervisor and first evaluation child were verified live. Its fresh seed audit
found no overlap against 1,660 distinct recorded real/collection seeds, and
its reservation lists exactly ten validation combinations. Follow
`artifacts/doom_frozen_transfer_round4_real_evaluations.status` and
`artifacts/doom_frozen_transfer_round4_real_evaluations.log`. Reserved tests
110000–110099 remain sealed until validation selection is frozen.

## Final frozen-world transfer result

All ten combinations completed 100 common validation games on90000–90099.
Mean/posterior survival means were: incumbent threshold 825.38/745.97,
packed threshold 756.63/752.15, incumbent sampled 802.56/879.74,
packed sampled 742.60/772.87, and original controller 733.51/737.71.
Validation selected incumbent sampled with posterior inference. Its weights,
mode and report fingerprints were frozen before110000–110099 were dispatched.

The selected candidate scored **853.53 ± 572.26** on those100 reserved games;
the unchanged mean-inference incumbent scored **819.15 ± 478.46**. Paired gain
was **34.38**, bootstrap95% **[-85.29, 156.04]**, with47 wins/52 losses/one tie.
Selected mean95% is [743.72, 967.45]. These are population SDs and50,000
whole-game paired resamples, seed74110, conditioning on fixed policies.
No reliable gain is established; no selection or retraining used test outcomes.
The selected candidate is preserved without replacing the canonical original.
All source weights, report hashes, seeds, action counts and summaries passed
verification; the supervisor exited with code zero.

Artifacts: `artifacts/doom_frozen_transfer_round4_paired_comparison.json` and
`artifacts/doom_frozen_transfer_round4_frozen_selection.json`. The selected
candidate lives in `checkpoints/VizdoomTakeCover-v0/frozen_transfer_round4_real_selected`.
The packed best remains an evaluation source with no best optimizer; latest
packed weights/optimizer/RNG are preserved and the stopped run was not resumed.

For interpretation, the round3 current-VAE pipeline beat the matched fresh
incumbent-VAE pipeline across all six controller-seed/inference comparisons
(mean differences169.75–275.81 steps). Both lost to the original incumbent.
This does not isolate the VAE from the original RNN's training history or
prove that improved reconstruction caused the regression against that original.

## Completion on diminishing returns

Three successive reserved comparisons yielded+47.10[-51.60,146.61], unchanged
incumbent retained with identical paired results, and+34.38[-85.29,156.04].
Combined with the packed loss plateau and unsuccessful new-model validation,
these results support stopping incremental experiments in the tested families.
Fifteen additional500-generation searches, new data/refinement and matched
vision/downstream comparisons established no reliable independent test gain.
This is a stopping judgment about measured returns, not a fundamental model
ceiling. The1092-step paper mean remains unmatched; the experiment retains its
documented frame-sampling, preprocessing, initialization, controller and
training-history differences from the reference.

CPU consolidation of400 unique original-incumbent test games gives801.47±504.99
and stratified mean95%[752.44,851.16]. It excludes duplicate paired copies and
does not replace the original840.06 cohort or establish exact paper replication.
Artifact: `artifacts/doom_incumbent_4_cohort_summary.json`.

No further GPU job is queued. All test ranges30000–30099,40000–40099,
80000–80099 and110000–110099 are consumed. The task state preserves all model
paths and prohibits automatic packed resume or reuse of tests for tuning.
The follow-up is paused after completion, and future experiments require a
new user request. `artifacts/doom_experiment_completion_audit.json` records
source/checkpoint verification and the stopping decision.
