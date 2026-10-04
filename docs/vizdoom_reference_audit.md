# Supplied-model reference audit

This diagnostic port loads the authors' public Doom VAE, RNN and controller
without a TensorFlow installation. It is separate from the reproduction's
trained checkpoints and evaluation pipeline. Loading supplied weights does not
reproduce training or establish the paper's 1092-step real-game mean.

The reference files come from
[`hardmaru/WorldModelsExperiments`, revision
`fd982b9691a941b52c6addbde29bc801ca6202c8`](https://github.com/hardmaru/WorldModelsExperiments/tree/fd982b9691a941b52c6addbde29bc801ca6202c8/doomrnn).
The downloader saves source URLs, byte counts and SHA-256 fingerprints in a
manifest. It preserves existing files and rejects modified or unrecognized
files. Downloads, weights and generated audit reports remain ignored by Git.

The port explicitly handles:

- TensorFlow's LSTM gate order, forget-bias offset and restart input/state reset.
- The factorized mixture-head layout and unclamped reference log standard deviations.
- NHWC-to-CHW encoder flattening and transpose-convolution kernel orientation.
- Native-frame cropping and the original SciPy RGB bytescaling/byte-cast behavior.

The last point is easy to misread: the old resize function returns bytes before
the wrapper computes `((1.0 - obs) * 255).round().astype(np.uint8)`. The final
cast wraps modulo 256; applying the expression to normalized floats produces
different pixels. A test executes the pinned legacy preprocessing functions
on three native images and checks byte-for-byte agreement. The harness uses
equivalent replacements for removed `np.float`, `ndarray.tostring` and
`Image.isImageType` APIs. Both paths use the installed Pillow; this is not a
comparison against the historical Pillow runtime.

## Reproduce the checks

From the repository root in the configured Linux/WSL virtual environment:

```bash
.venv/bin/python scripts/tools/fetch_doom_reference.py \
  --output-dir artifacts/doom_reference_round5/fd982b9

JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python -m unittest tests.test_doom_reference -v

JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python scripts/tools/audit_doom_reference_port.py \
  --reference-dir artifacts/doom_reference_round5/fd982b9 \
  --output artifacts/doom_reference_round5_port_cpu_audit_checked.json
```

The preprocessing comparison skips when the optional downloaded legacy source
is absent. The model audit requires the completed, fingerprint-checked download
and refuses to replace an existing report. Use a new output filename for a
later audit.

## Measured results, 2026-10-03

All 39 CPU unit tests passed, including the three reference tests and their
legacy preprocessing comparison. The full supplied-weight CPU audit also passed
against independent NumPy convolutions, transpose-convolution scatter additions
and recurrent equations:

| Calculation | Maximum absolute difference |
| --- | ---: |
| Encoder means, three synthetic frames | 0.0000008345 |
| Encoder log variances, three synthetic frames | 0.0000019074 |
| Decoder, one synthetic latent vector | 0.0000002980 |
| Recurrent states, 128 steps including restarts | 0.0000133514 |
| Controller actions on those states | 0.0000107884 |

The local report records the downloaded-file fingerprints, port/test source
fingerprints and tolerances. These checks establish numerical agreement with
independent implementations of the reference equations, not with a running
TensorFlow 1.8 session. The historical Pillow version, scenario/engine equality
and provenance of the exact checkpoint used for the paper's reported score
remain unverified. No real-game performance evaluations or training runs were
launched for this audit; there is no new measured real-game performance result.

The next diagnostic boundary is to verify the reference environment and
observation/action/recurrent-state timing before evaluating the supplied
controller on fresh seeds. That comparison should identify protocol differences
before changing our trained models.

## Reference environment comparison, round 5

The renewed goal is to match or exceed the paper's 1092-step mean. Earlier
stopping on diminishing returns remains a historical decision; it does not
complete this stricter goal. Original checkpoints and all consumed test cohorts
remain preserved.

The pinned `gym-doom` configuration and `doom-py` 0.0.15 distribution revealed
these differences from our prior pipeline:

| Setting | Reference recipe | Prior reproduction |
| --- | --- | --- |
| Effective difficulty | Original request5 clamped to4 by doom-py | 4 in installed config |
| Effective native color order | Old BGR24 emits RGB bytes | Modern RGB24 |
| Native image handling | Crop to first 400 rows, legacy resize/cast | Full-frame OpenCV resize |
| Episode start time, runtime getter | 14 | 1 |
| Engine timeout, runtime getter | 2100 | 0; wrapper caps at 2100 actions |
| Render HUD, config | True | False |
| Action threshold | 0.3333 | 0.3 |
| Controller | Bias-free `tanh([z,c,h] @ w)` | Affine controller |
| RNN input | Latent, raw action, restart | Latent and physical action in canonical mode |
| Game assets | Freedoom 0.10.1 from original download recipe | Installed ViZDoom assets |

The Take Cover WAD is identical in both packages, SHA-256
`6fcd3c50c7f303628a9c36f1af68dd5b683c56a81302cd1118f569dbb6637990`.
The Freedoom WAD differs: reference
`113ea2d9677074f54cbde29b8f739629f438ed20e3b1d3ba69c9478318ea504d`,
installed `729f100448d0b48e12ecc8004e096a9ea6df024467d962f002bb286805be3a8e`.
The pinned archive is inspected without installing old bindings, and selected
members are read without extracting arbitrary archive paths. The installed
engine has no difficulty getter; difficulty is a recorded setter/config request,
not a runtime readback. Other available getters, button order and one-step action
mapping were checked in live CPU environment smoke tests on seeds 120998/120999.

The first performance comparison was a fixed supplied-model diagnostic on 100
fresh games, seeds 120000–120099, with eight workers and no warmup. Policy choice
comes from the supplied file, not these outcomes. It preserves the reference
order: choose an action from current `z,c,h`, consume `z,raw action,restart` in
the RNN, then obtain the next game observation. Posterior noise and controller
arithmetic use FP64; VAE/RNN use FP32 with highest multiplication precision.
Independent JAX keys and actual game seeds are explicit per game. This differs
from the legacy NumPy/Gym streams and its game-seeding behavior. The installed
ViZDoom engine also differs; this is a port of the source recipe, not an exact
historical-runtime replay. Reports record actual controlled actions and reward
separately. A CPU check with a 32-tic timeout and start time14 completed32
actions, from engine tick14 through46, earning32 points. That check did not
show a timeout accounting mismatch.

Each validated episode is saved immediately, with frozen source/input hashes,
action counts and death/timeout labels. Recovery schedules only unfinished seeds
and requires the same frozen protocol. Future own-controller validation seeds
130000–130099 and final test seeds 140000–140099 are reserved separately; the
seed inventory must be rechecked before dispatch. Supplied-policy diagnostics
do not establish reproduction of our training or complete the model goal.

The literal skill5 port completed at **119.42 ± 38.97** over those100 games,
all deaths, mean bootstrap95% **[112.33,127.50]**. This is a preserved diagnostic
failure, not the faithful reference benchmark: inspection of the hash-pinned
`doom-py` source showed `setSkill(5)` clamps to4, passes `-skill4`, and the CLI
maps that to its fourth difficulty. Current ViZDoom clamps only above5 and
passes `-skill5` for the same API call. The portable wrapper now requests4 to
preserve effective difficulty. This source-based correction precedes a second
fixed-model diagnostic on the same100 seeds; no controller is selected from
those results. Both inputs/configuration and code fingerprints are retained.
The original prior reproduction already requested effective4; its difficulty
should not be described as different from the legacy environment. The skill4
control still using modern BGR completed **236.13 ±117.25**, all100 deaths.

Further source inspection found that the legacy renderer's nominal `BGR24`
places red, green and blue at offsets0,1,2, while current BGR24 uses2,1,0.
Modern **RGB24** preserves the original effective bytes. A CPU probe of three
reference frames independently supports that correction: supplied-VAE mean
reconstruction SSE is58.49–59.41 with modern BGR versus0.29–0.73 after swapping
toRGB; normalized inversion performs much worse. These are diagnostics, not
survival evidence. The correct comparison keeps effective skill4 and changes
only this source-verified color mapping, again with frozen models and the same
diagnostic seeds. The previous skill4/BGR source files were fingerprint-checked
and archived before the correction. Our prior reproduction already used modern
RGB, so this fixes reference-port compatibility rather than an RGB/BGR bug in
the original trained pipeline.

Tools: `scripts/tools/fetch_doom_reference_assets.py`,
`scripts/tools/evaluate_doom_reference.py`; local preparation
`artifacts/doom_reference_round5_preparation.json`, supervised job
`artifacts/run_reference_round5_control.py`, and intended report
`artifacts/doom_reference_round5_supplied_real100.json`. CUDA preflight is
required before dispatch and only one GPU job may run at a time. The corrected
supervisor/report are `artifacts/run_reference_round5_corrected_control.py` and
`artifacts/doom_reference_round5_supplied_effective_skill4_real100.json`;
the source/uncertainty audit is `artifacts/doom_reference_round5_difficulty_migration.json`.
The color-corrected supervisor/report are
`artifacts/run_reference_round5_rgb_control.py` and
`artifacts/doom_reference_round5_supplied_effective_rgb_real100.json`;
source audit `artifacts/doom_reference_round5_color_migration.json` and CPU
frame diagnostics `artifacts/doom_reference_round5_pixel_probe.json`.

The effective skill4/RGB diagnostic completed **eight games at 1083.88
± 526.27**, seeds 120000–120007, all deaths and no timeouts. All eight had
reward equal to controlled actions. This small diagnostic uses supplied author
weights and does not establish the paper's100-game result or a reproduction of
our training. Its report is `artifacts/doom_reference_round5_error_diagnostic8.json`.
The separate100-game attempt stopped at the reward-versus-step validation
check after preserving one valid episode, seed120002 with244 steps. The
offending record was not saved, and the eight-game replay did not reproduce
the error. The failure's cause remains unresolved; worker reassignment is a
candidate to inspect, not an established diagnosis. The full corrected-color
benchmark is incomplete, and no assertion has been relaxed to claim a result.

To prepare a fresh local reference directory and run the evaluator from the
repository root:

```bash
.venv/bin/python scripts/tools/fetch_doom_reference.py \
  --output-dir artifacts/doom_reference_round5/fd982b9
.venv/bin/python scripts/tools/fetch_doom_reference_assets.py \
  --output-dir artifacts/doom_reference_round5/doom_py_0015
.venv/bin/python scripts/tools/check_gpu.py
.venv/bin/python scripts/tools/evaluate_doom_reference.py \
  --reference-dir artifacts/doom_reference_round5/fd982b9 \
  --asset-dir artifacts/doom_reference_round5/doom_py_0015 \
  --output artifacts/doom_reference_round5_supplied_effective_rgb_real100.json \
  --episodes 100 --seed 120000 --workers 8 --difficulty 4 --color-order rgb
```

The commands document the diagnostic cohort already used here. Before a new
experiment, audit seed overlap and reserve a separate cohort. Existing partial
reports require matching input and source fingerprints; changed protocols need
a separate output. Downloaded weights, assets, logs and detailed reports remain
local under the ignored `artifacts/` directory. Never resume the packed run
deliberately stopped at epoch255.

Commit validation: all 41 tests passed on CPU, including per-seed RNG/memory
reset, recovery callbacks, reference policy timing, button mapping and timeout
labels. Ruff lint/format checks and `git diff --check` passed. These checks do
not resolve the preceding real-game evaluation failure described above.

## Full reference replay and controller refinement

A new full replay of the same fixed models and diagnostic schedule completed
all100 games on120000–120099: **991.20 ± 506.48**,98 deaths and2 timeouts.
Every reward equalled its controlled action count, and the first eight records
exactly matched the separate eight-game replay. The preceding scoring failure
did not recur; its original offending record is unavailable, so its cause
remains unresolved. An unchanged scoring assertion accepted both complete
2100-action timeouts. The current evaluator captures future failing records
separately from validated partial reports and refuses blind failure recovery.

The fixed-policy mean's whole-game bootstrap95% interval is
**[893.45,1091.04]**. Against the preserved effective-skill4/BGR control on the
same seeds, the effective-RGB correction adds **755.07** steps on average,
with paired95% **[662.22,850.52]**. These50,000 resamples, seed74121, condition
on fixed policies and exclude training variability. Neither the mean nor this
diagnostic establishes the requested1092-step performance of our trained model.
The public controller checkpoint is not proven to be the exact checkpoint
behind the paper's reported result, and engine/RNG/Pillow differences remain.

Report: `artifacts/doom_reference_round5_error_capture100.json`, SHA-256
`ebd003d07721c96308a9f12b4e763287a9ddbc2a34ef3b080661445be21bde27`.
CPU analysis: `artifacts/doom_reference_round5_rgb_analysis.json`. All frozen
input/source fingerprints passed before editing the evaluator; the exact
four source files also match the preserved
`artifacts/doom_reference_round5/source_snapshots/rgb_before_validator_fix/`.

The next bounded comparison trains a controller on the imported frozen public
world. It does not claim newly trained VAE/RNN weights. Settings are
temperature1.15,500 CMA generations,population64,16 trials per candidate,
candidate batch64,initial sigma0.02,seed91,starting from the public controller.
The controller has1088 bias-free weights over `[z,c,h]`; it sends raw actions
and an initial restart flag into the RNN. Latent/control arithmetic usesFP64,
the RNN usesFP32, and multiplication precision is highest. Dream death requires
a strictly positive logit, the terminal action earns one point, and invalid
active trajectories cause a failure rather than receiving a survival score.

Start distributions come from the reference `initial_z.json`, divided by10000.
A fixed shuffled90/10 split of start indices separates training from dream
validation. Candidates share16 starts and noise streams within each generation;
every10 generations, the population's best candidate and CMA mean face64 fixed
held-out dream rollouts. The initial policy remains eligible as the best
checkpoint. Best and latest policies, CMA optimizer, NumPy RNG and next JAX key
are retained. Dream validation is a training check, not proof of real transfer.

`scripts/tools/train_doom_reference_controller.py` and
`src/doom_reference_training.py` implement this comparison. All45 CPU tests
passed before GPU dispatch, including independent scalar rollout/state/action
checks, strict zero-logit survival, and an isolated actual-CMA checkpoint test
that reproduces the next population after restoring optimizer and RNG state.
CUDA matrix multiplication and convolution backward preflight passed. One GPU
job is supervised by `artifacts/run_reference_round5_controller_search.py`;
checkpoints are in
`checkpoints/VizdoomTakeCover-v0/reference_round5_controller_s91/`.

Real selection will compare distinct best/final/controller and posterior/mean
combinations on20 fresh validation games130000–130019, deduplicating identical
policies in the same inference mode. Freeze the validation winner before100
reserved tests140000–140099 and evaluate the supplied reference control on
those same test seeds for a paired comparison. The rest of the reserved
validation range remains unused. Re-audit seeds immediately before dispatch;
test outcomes must never choose or retrain a policy. The new evaluator checks
controller/world fingerprints and records inference mode and evaluation role.
Its additional isolated CPU test passed for mismatched-world rejection,
failed-record preservation, mean inference metadata, and refusal to blindly
resume a captured failure. The original840.06 result and all original
checkpoints remain preserved; packed training stopped at255 must not resume.

The search completed500 generations with input/source and checkpoint hashes
verified. Best held-out dream score was **1162.40625 at generation270**, versus
initial1110.796875; final CMA mean scored961.53125. This is an own controller
update on an imported public world, with real performance still unproven.
Best policy SHA-256:
`4bab7eabccabdc4cc259de156a7bcabb0b6c0a5cedf768f630242f29db4c4525`;
final policy:
`968bd53f8d66500be1891180f98b657ecd91095b6bdc2226567ddbf4723ea9c8`.
Optimizer/RNG and all originals passed preservation checks. Search result:
`artifacts/doom_reference_round5_controller_s91_result.json`.

The serial real supervisor `artifacts/run_reference_round5_real_evaluations.py`
completed six distinct validation combinations. Each completed report
must match its policy, inference, role, seeds, frozen inputs and source. Valid
partial reports recover unfinished seeds; captured failures require diagnosis.
`artifacts/doom_reference_round5_frozen_selection.json` records selection
and fingerprints before tests. Final job result will be
`artifacts/doom_reference_round5_real_evaluation_result.json`. Neither those
pending test results nor dream improvement currently proves the1092-step objective.

All six real validation reports passed independent CPU checks for exact seeds,
game records, summaries, policy/inference and frozen input/source hashes:

| Frozen controller | Posterior mean ± population SD | Mean-latent mean ± population SD |
| --- | ---: | ---: |
| Supplied reference | 877.75 ± 478.91 | 854.50 ± 400.92 |
| Best dream checkpoint, generation270 | **958.05 ± 596.39** | 918.85 ± 523.56 |
| Final CMA mean, generation500 | 697.65 ± 285.45 | 772.75 ± 353.09 |

The maximum20-game validation mean selected generation270 with posterior
inference. Selection was frozen at2026-10-03T21:39:24.496998Z before its reserved
100-game test began. Test seeds140000–140099 are also used for the unchanged
supplied posterior control. No test outcomes choose a policy. The validation
lead of80.30 steps is not an independent test improvement.
Its descriptive paired whole-game bootstrap95% interval is[-111.75,+281.35];
the selected validation mean's interval is[702.45,1226.75] (50,000 resamples,
seed74130). These intervals hold policies fixed and do not correct for choosing
the winner on these same20 games or for training variability. The validation
lead alone does not establish a reliable gain. CPU analysis:
`artifacts/doom_reference_round5_validation_uncertainty.json`; it read no test
outcomes and did not change the frozen selection.

`src/doom_reference_results.py` checks complete, unique game cohorts and
recomputes survival summaries. The CPU-only completion tool additionally
recomputes selection from validation, verifies frozen report/input/source
hashes and selection timing, and jointly resamples whole games for selected,
control and paired-difference95% intervals. Run it after both test reports
and the final supervisor result exist:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=4 .venv/bin/python \
  scripts/tools/summarize_doom_reference_results.py
```

The default output is `artifacts/doom_reference_round5_paired_comparison.json`;
existing analyses are preserved. It records own-controller versus imported
world provenance and leaves goal completion for review. Its isolated fixtures
reject changed validation selection, missing games, wrong rewards/summaries
and mismatched paired worlds, and verify paired intervals with known gains.

The selected generation270/posterior controller has now completed all100
reserved games140000–140099 at **1035.22 ± 550.45**,92 deaths and8 timeouts.
Its fixed-policy whole-game bootstrap95% mean interval is **[929.91,1144.23]**
(50,000 resamples,seed74140). The100-game observed mean does not meet1092;
an interval containing the paper's mean does not prove the requested target.
This is our controller update on imported frozen public VAE/RNN weights.
All exact seed/game records, means/SDs, input/source fingerprints and agreement
with the frozen validation policy passed independent CPU checks in
`artifacts/doom_reference_round5_selected_test100_cpu_audit.json`. Report SHA-256:
`5378ec02257be63e2978a84eb3e03558de80f5e2574fe0e039b89d36968893db`.
The supplied posterior policy's paired100-game control completed at
**955.39 ± 541.20**,93 deaths and7 timeouts. Selected-minus-control mean gain
is **+79.83**, with paired whole-game bootstrap95% **[-22.26,+183.00]**;
48 wins,43 losses and9 ties. The control mean's interval is[850.09,1061.53].
These50,000 joint resamples use seed74140 and hold both policies/world fixed;
they exclude training variability and historical engine/RNG differences.
A reliable gain is not established, and1035.22 remains below1092.

The final CPU audit independently recomputed the validation-only winner,
verified all six validation reports and both exact100-game test cohorts,
checked report/world/source hashes and selection timing, and confirmed the
controller is an own update on imported public world weights. Artifact:
`artifacts/doom_reference_round5_paired_comparison.json`. Both supervisor and
evaluation child exited, and the GPU compute-process query was empty before
the next guarded experiment. Original checkpoints remain intact.

## Controlled temperature follow-up

The pinned public source sets dream temperature1.25
([source](https://github.com/hardmaru/WorldModelsExperiments/blob/fd982b9691a941b52c6addbde29bc801ca6202c8/doomrnn/doomrnn.py#L24)),
whereas the paper's strongest reported result uses1.15. The paper's temperature
effect is not monotonic:1.30 transferred worse than1.15
([paper](https://worldmodels.github.io/#cheating-the-world-model)). A matched1.25
comparison is a source-motivated hypothesis, not a guaranteed improvement.

`artifacts/doom_reference_round6_temperature_protocol.json` fixes the next
bounded comparison before dispatch. Only temperature and output path differ
from the1.15 search: the public world, starting policy, CMA seed91,sigma0.02,
500 generations,population64,16 trials,candidate batch64 and fixed64-dream
validation every10 generations stay identical. The frozen1.15 generation270
policy is retained as a control. Using64 dream validation rollouts still
differs from the paper's1024-rollout checks; this matched comparison isolates
temperature within one training realization, not general superiority.

Four distinct policies—supplied, frozen1.15 best, new1.25 best and new1.25 final—
use posterior inference on80 fresh validation games130020–130099, with
identical parameter/inference combinations deduplicated. The larger cohort is
motivated by the wide20-game validation interval. Only a newly updated1.25
policy that wins validation against both controls will be frozen for100 new
test games150000–150099, paired with the frozen1.15 policy on those same seeds.
Retaining an unchanged control will not trigger another100-game test of that
same policy to seek a lucky mean. No test outcomes select training settings,
checkpoint or inference mode.

The CPU seed precheck found no overlap with2084 recorded unique real seeds;
it consumes no games and must be repeated before dispatch. The prepared
launcher `artifacts/run_reference_round6_temperature_search.py` refuses to
start until the current paired reports and independent CPU comparison are
complete, the prior processes have exited and the exclusive GPU lock is free.
It then requires CUDA preflight, current source/input hashes, preserved
controls, and a fresh output path; it retains best/final, CMA optimizer and RNG.
Its CPU `--check-only` correctly returned not-ready while the existing paired
evaluation was live. Lint/format passed. Preparation started no GPU job, and
evaluation sources were left unchanged during that preparation.

After the completed paired CPU audit and confirmed prior process exit, the
guarded1.25 search completed all500 generations. Its best fixed held-out dream
score was889.296875 at generation210, versus initial768.1875; the final
checkpoint scored698.296875. Independent CPU checks verified checkpoint,
optimizer and RNG preservation, matched dream starting indices, and unchanged
world/source fingerprints and original checkpoints. Artifact:
`artifacts/doom_reference_round6_training_cpu_audit.json`. Dream scores at
different temperatures do not establish real survival performance.

All four posterior-inference policies completed the same80 validation games
on seeds130020–130099. CUDA preflight passed before evaluation.

| Policy | Mean ± population SD | Paired difference from supplied | 95% paired interval |
| --- | ---: | ---: | ---: |
| Frozen1.15 generation270 | 904.83 ±511.93 | −18.29 | [−156.40,+119.00] |
| Supplied control | **923.11 ±544.26** | — | — |
| New1.25 best, generation210 | 914.99 ±521.24 | −8.13 | [−131.61,+116.06] |
| New1.25 final, generation500 | 914.28 ±514.03 | −8.84 | [−155.79,+134.65] |

Validation retained the supplied control. Neither new candidate won, so the
reserved100-game cohort150000–150099 was not consumed. These descriptive
intervals use50,000 paired whole-game resamples, seed74180, and hold fitted
policies fixed; they exclude training and selection variability. They establish
no reliable improvement or1092-step result.

The independent CPU closure recomputed the selection from actual ordered game
records, verified source/world/controller and report fingerprints, and checked
that no reserved reports or partial games existed. Both evaluation processes
exited. Artifact:
`artifacts/doom_reference_round6_validation_closure_cpu_audit.json`.

A CPU-only training-history review confirmed52 dream checkpoint checks reused
the fixed64-rollout cohort in each search. The top two recorded dream scores
differ by14.34 at1.15 and38.41 at1.25. These aggregate gaps cannot establish
selection noise or overfitting without per-rollout variance. The
[paper's Doom appendix](https://worldmodels.github.io/#doomrnn) evaluated its
best controller over1024 dream rollouts. Increasing dream validation to1024
is therefore a source-backed follow-up hypothesis to reduce small-cohort
selection noise, not a promised real-game gain. Review artifact:
`artifacts/doom_reference_dream_selection_cpu_review.json`. The review itself
started no GPU job and used no reserved-test outcomes.

Inspection of the preserved, hashed500-generation optimizer states found
final CMA scales0.01452 at1.15 and0.01512 at1.25, from initial0.02. Aggregate
population dream fitness averaged916.04 then900.91 over the first/last50
generations at1.15, and738.11 then727.46 at1.25. These descriptive means show
no upward trend and provide no basis to assume simply extending the same
search will improve real survival. They do not establish optimizer collapse
or a causal explanation: candidates change and share random dream streams.
CPU artifact: `artifacts/doom_reference_optimizer_progress_cpu_review.json`.

## Smaller controller search with larger dream validation

The preregistered round7 recipe returns to temperature1.15 on the unchanged
public world and starting policy. Compared with the previous1.15 search,
initial CMA sigma changes from0.02 to0.005 and fixed dream validation grows
from64 to1024 rollouts. Seed91,500 generations,population64,16 fitness trials,
candidate batch64 and validation every10 generations remain fixed. This
tests the two changes together and cannot attribute an effect to either alone.

The CUDA search completed500 generations at2026-10-03T23:22:55Z. Its best
1024-dream score was950.0654 at generation120, versus initial928.0801; final
CMA mean scored911.2314. Best/final checkpoints and optimizer are preserved.
The independent CPU audit verified actual0–500 history, exact registered
arguments and start indices, optimizer count500 and final mean, NumPy RNG and
next JAX key, plus unchanged sources/world and incumbent fingerprints.
Artifact: `artifacts/doom_reference_round7_training_cpu_audit.json`.
These are dream scores; real performance remains unproven. The search session
exited0, and its supervisor and child have exited.

The frozen protocol reserves80 fresh validation seeds130100–130179 for the
new best/final policies, supplied control and frozen1.15 generation270 control.
Round6's retained supplied policy adds no distinct control. Identical parameter
sets are deduplicated, and controls win ties. Only a distinct new candidate
winning validation gets100 reserved games160000–160099, paired with the
highest validation control whose identity is frozen before test outcomes.
Retaining a control starts no reserved test. The serial real supervisor has
started, with CUDA matrix multiplication and convolution backward preflight
passed on the RTX4070Ti. Supervisor843089 and first evaluator843158 were
verified live; session94377 and current logs are in `artifacts/task_state.json`.
This dispatch started no concurrent repository GPU job. The completed selection
and subsequent test stage are recorded below.

The first control, frozen1.15 generation270, completed all80 validation games
at **920.03 ±534.36** (77 deaths,3 timeouts). Its fixed-policy whole-game
bootstrap95% mean interval is **[805.16,1038.41]**, using50,000 resamples,
seed74180. Complete ordered records, controller/world/source fingerprints and
registered cohort passed the independent CPU check in
`artifacts/doom_reference_round7_prior_control_validation_cpu_audit.json`.
The supplied control subsequently finished at **944.91 ±553.37** (74 deaths,
6 timeouts), mean95% **[824.55,1069.56]**. Prior-minus-supplied paired gain is
**−24.89**,95% **[−159.98,+111.19]**,30 wins/43 losses/7 ties. The paired
whole-game calculation uses the same50,000 resamples and seed74180; shared
world, numerical protocol, source fingerprints and both complete cohorts
passed verification in
`artifacts/doom_reference_round7_controls_validation_cpu_audit.json`.
The new best-generation120 candidate then completed **947.80 ±543.20**
(75 deaths,5 timeouts), mean95% **[831.36,1068.06]**. Its paired gain over
supplied is **+2.89**,95% **[−113.21,+118.34]**,25 wins/30 losses/25 ties;
over prior1.15 it is **+27.78**,95% **[−105.80,+161.58]**. The three complete
cohorts, shared world/protocol, source/input fingerprints and candidate
provenance passed the independent CPU check in
`artifacts/doom_reference_round7_best_validation_cpu_audit.json`.
This small observed lead establishes no reliable improvement. The final
checkpoint subsequently finished **875.79 ±544.29** (75 deaths,5 timeouts),
mean95% **[759.27,997.34]**; its paired difference from supplied is **−69.13**,
95% **[−211.14,+70.38]**.

All four80-game validation reports selected generation120. The policy and
the highest validation control, supplied posterior inference, were frozen at
**2026-10-04T00:09:49.485111Z** before reserved testing. The independent CPU
audit reconstructed the distinct parameter identities, all complete cohorts,
control-first selection and comparator, verified input/source/incumbent hashes
and freeze timing, and checked the test protocol against the frozen policy.
Artifact: `artifacts/doom_reference_round7_frozen_selection_cpu_audit.json`.
The selected policy completed100 fresh tests on160000–160099 at
**945.73 ±570.08** (93 deaths,7 timeouts), with fixed-policy bootstrap mean95%
**[836.80,1057.28]** from50,000 whole-game resamples, seed74140. The exact
cohort, records, frozen parameter identity, input/source fingerprints and
selection-before-test timing passed independent CPU verification in
`artifacts/doom_reference_round7_selected_test100_cpu_audit.json`.
This observed mean is below1092. The frozen supplied control completed
**922.62 ±562.71** on the same100 seeds, mean95% **[813.64,1033.16]**.
Selected-minus-supplied paired gain is **+23.11**,95% **[−64.56,+109.06]**,
41 wins/34 losses/25 ties. The50,000 paired whole-game resamples use seed74140
and hold fitted policies fixed; training/selection variability and historical
runtime differences are excluded. No reliable improvement is established.

The protocol-aware final CPU audit verified validation-only choice, strongest
control, both complete100-game cohorts, report/input/source fingerprints,
candidate provenance and freeze timing. Artifact:
`artifacts/doom_reference_round7_paired_comparison.json`. Session94377 exited0,
supervisor843089 and child885763 exited, the GPU compute-client query was
empty, and canonical incumbent fingerprints remained intact. Test outcomes
did not select or retrain a policy. The1092-step goal remains active.

Protocol: `artifacts/doom_reference_round7_fine_search_protocol.json`, SHA256
`9f9cb63038ea87a0926d770a4ed6ca5aad95693840fefe01c41ab383e867f5e8`.
Search result: `artifacts/doom_reference_round7_fine_search_result.json`.
This remains own-controller training on imported public VAE/RNN weights,
with modern runtime differences; the1092-step goal is unproven.

## Larger fitness average

The next recipe was registered before dispatch. Relative to
round7 it changes only the output path and candidate-fitness rollout count
from16 to64. Temperature1.15, sigma0.005, author initialization, CMA seed91,
500 generations, population64, candidate batch64 and1024 fixed dream
validation rollouts every10 generations stay identical.

Training-only history shows first/last50-generation population means912.95
and906.09. No per-rollout variance was saved, so this does not diagnose noisy
ranking or prove a cause. Increasing the fitness average is a controlled
hypothesis: under independent equal-variance draws,64 versus16 rollouts halves
the Monte Carlo standard error of an individual fitness estimate. That is an
expectation, not a measured reduction or a promised real-game gain. Evidence:
`artifacts/doom_reference_fitness_rollouts_cpu_review.json`.

The immutable protocol reserves80 fresh validation seeds130180–130259 and,
only for a new validation winner,100 test seeds170000–170099. Controls are the
supplied policy and the round7 generation120 winner chosen by validation.
The highest validation control is frozen before paired testing. Seed precheck
found no overlap with recorded games. Protocol:
`artifacts/doom_reference_round8_fitness64_protocol.json`, SHA256
`5a9f66836da835dcb9b6e708a9d1d3ff9c8b85055e3c430efee6dff4bac27e71`.

The guarded launcher requires the prior paired reports and independent
audit, prior process exit, preserved inputs/checkpoints, exclusive GPU lock
and fresh CUDA preflight. Its initial CPU check returned not-ready while the
pair ran. After the completed independent audit and verified process exit,
the CPU readiness check passed with no live previous processes. It requires
review rather than launching if the prior result reaches1092.
Four isolated CPU workflow fixtures passed retention/new-winner routes with
either public or trained policy as the strongest comparator. Their synthetic
scores are not experiment results. After the prior pair and audit completed,
the single CUDA search started with matrix multiplication and convolution
backward preflight passed. It completed500 generations, with best1024-dream
score953.3916 at generation300 versus initial928.3779; final CMA mean scored
901.1104. Session90086 exited0 and both search processes exited. An independent
CPU audit reconstructed0–500 history and900/100 disjoint start indices,
verified arguments, source/input fingerprints, best/final weights, optimizer
count500 and exact final mean, restored the saved NumPy RNG and recomputed the
next JAX key. Canonical incumbent and prior control remain intact. Artifact:
`artifacts/doom_reference_round8_training_cpu_audit.json`.

Fresh-seed and process-exit checks passed before the serial real supervisor
started. CUDA matrix multiplication and convolution backward preflight passed
again before its first validation. Supervisor928337 and evaluator928406 were
verified live, session10253; current stage and logs are recorded in task state.
Four80-game validations were scheduled. The copied frozen test-range metadata
upper bound was corrected before launch, and four isolated CPU routing fixtures
also checked the exact170000–170099 bounds. Dream scores establish no real-game
improvement; the completed real validations and selected test are recorded below.

The prior generation120 control completed80 games at **973.29 ±520.30**,
77 deaths/3 timeouts, with fixed-policy bootstrap95% mean interval
**[860.60,1090.03]** (50,000 whole-game resamples, seed74180). Complete
registered seeds, raw records, source/world/controller identity and preserved
incumbent passed independent CPU verification in
`artifacts/doom_reference_round8_prior_control_validation_cpu_audit.json`.
The recomputed standard deviation differs by1.14e-13, one unit in the last
place; the exact mean matches, and the strict1e-9 absolute tolerance passes.
No training or inference source changed. The supplied control completed
**922.10 ±543.40**,75 deaths/5 timeouts, mean95% **[804.71,1042.63]** on the
same80 seeds. Prior-minus-supplied paired gain is **+51.19**,95%
**[−56.91,+162.31]**,35 wins/25 losses/20 ties. This establishes no reliable
advantage. Complete raw cohorts, shared runtime/world/source fingerprints,
controller identity and canonical weights passed CPU verification in
`artifacts/doom_reference_round8_controls_validation_cpu_audit.json`.
The paired whole-game calculation uses50,000 resamples, seed74180, with fitted
policies held fixed and training/selection uncertainty excluded. The same
supervisor advanced to new best-generation300, evaluator953190 verified live
at16/80 games. These validation baselines are not reserved-test results.

New best-generation300 subsequently completed **931.41 ±523.98**,76 deaths/
4 timeouts, mean95% **[818.54,1047.91]**. Its paired differences are **−41.88**
from prior,95% **[−147.96,+63.29]**, and **+9.31** from supplied,95%
**[−93.66,+114.30]**. Complete records and shared world/runtime/source,
candidate provenance and preserved canonical weights passed CPU verification
in `artifacts/doom_reference_round8_best_validation_cpu_audit.json`.

Final-generation500 completed **1016.39 ±560.94**,73 deaths/7 timeouts,
mean95% **[895.89,1140.98]**. Its paired validation gain over prior is **+43.10**,
95% **[−86.03,+176.65]**; this is not a reliable improvement. All four complete
80-game cohorts selected the final checkpoint and prior generation120 as the
highest control. Both identities and posterior inference were frozen at
**2026-10-04T01:45:41.848942Z** before reserved testing. The independent CPU
audit reconstructed every task and parameter identity, raw scores and
control-first selection, source/world/incumbent fingerprints and freeze
timing. Artifact: `artifacts/doom_reference_round8_frozen_selection_cpu_audit.json`.
The selected policy completed100 fresh170000–170099 tests at **1026.99 ±597.70**,
90 deaths/10 timeouts, with fixed-policy bootstrap95% mean interval
**[911.33,1145.64]** (50,000 whole-game resamples, seed74140). Independent CPU
verification checked all100 actual records, registered seeds, frozen controller
and inference, source/world fingerprints, provenance, freeze timing and preserved
canonical weights. Artifact:
`artifacts/doom_reference_round8_selected_test100_cpu_audit.json`; original report
SHA256 `6cde9a0b7d29cf7980391385131d150642a1fafc1db2e3beb4a44e78e7297875`.
The observed mean is below1092. Its interval containing1092 does not prove the
goal. The frozen prior completed **1040.31 ±616.27** on the same100 seeds,
91 deaths/9 timeouts, mean95% **[920.58,1163.10]**. The selected-minus-prior
paired difference is **−13.32**,95% **[−129.75,+103.78]**,42 wins/43 losses/
15 ties. This establishes no reliable improvement. The protocol-aware CPU
auditor reconstructed selection from all validation cohorts and verified both
complete tests, source/world fingerprints, provenance and freeze timing.
Current runtime versions, raw controller identity, training-state fingerprints,
canonical preservation and actual process exit passed a separate completion
review; session10253 exited0 and no repository GPU work remained queued.
Artifacts: `artifacts/doom_reference_round8_paired_comparison.json` and
`artifacts/doom_reference_round8_completion_cpu_audit.json`.

An independent CPU comparison verified matching recorded sources, inputs,
initial parameters, dream start indices, seeds, precision and runtime versions;
only output and fitness-rollout arguments changed. However, the initial
1024-dream mean was928.3779 versus round7's928.0801, a difference of0.2979.
The cause is unresolved, and no per-rollout trace was saved. This is a recorded
repeatability limitation, not evidence of a fitness-rollout benefit or a
diagnosed backend cause. Artifact:
`artifacts/doom_reference_round8_initial_match_cpu_audit.json`.

## Prepared validation-winner initialization

A contingent next recipe was registered from training and complete validation
evidence before any round8 reserved outcomes were used. It initializes a fresh
CMA search from the completed validation winner, retaining temperature1.15,
sigma0.005, seed91, population64, fitness64, batch64,500 generations and1024
dream validations every10 generations on the same public VAE/RNN. Preserving
useful real behavior through initialization is a hypothesis, not a promised
gain. The registered decision rule resolved to the round8 final-generation500
policy only after all four validations and the independent selection audit.

The separate trainer prototype leaves original frozen numerical sources
intact. CPU checks verify actual prior/best/final initializer arrays, shared
best/final metadata resolution, incompatible vectors/flags/world rejection,
and unchanged rollout/optimizer/save code. The initial unlaunched prototype
and preparation are preserved; a bookkeeping revision fixes final-checkpoint
metadata lookup without changing the registered recipe or decision rule.
Current preparation:
`artifacts/doom_reference_round9_validation_initializer_preparation_v2.json`,
SHA256 `57f01a90231958ea9cfe14e5b3431b76b751733ccb45c7d5a70dc819016446e7`.
The current pair finished below1092 and passed independent completion review.
The materialized protocol is
`artifacts/doom_reference_round9_validation_initializer_protocol.json`, SHA256
`e1a25bf4c162a746b46961cd5a393321fddd3a9720e644353e336a5daa0eda7a`.
Its guarded launcher verifies the actual preparation, completed validation
winner, input/source fingerprints and fresh cohorts, requires the previous
completion review and current process absence, and takes an exclusive GPU
lock before CUDA preflight. Six isolated CPU routing fixtures passed, including
deduplication if the best checkpoint retains the initializer. A separate CPU
trainer-state auditor requires the completed search and cannot claim an
unstarted result. The recipe reserves80 fresh validation130260–130339 and
eligible paired100 tests180000–180099. Test scores never choose or alter its
initializer/settings. Readiness then passed with no live previous process.
The single search started in session83493, supervisor994238 and trainer994308
verified live. CUDA matrix multiplication and convolution backward checks
passed. Search arguments match the registered recipe and exact initializer;
no new real result is yet available.

Independent CPU review verified the actual initializer parameters and metadata,
frozen inputs/source, unchanged900/100 start pools and numerical runtime. The
initial1024-dream mean is **923.2109**, versus **901.1104** saved for the same
policy as round8's final CMA mean, a difference of **22.1006**. Source inspection
confirms initial validation calls one candidate (1024 engine trajectories),
while the previous final validation calls two (2048 trajectories). This is a
recorded difference, not a diagnosed numerical cause; no previous per-rollout
trace or controlled GPU comparison exists. The active search is preserved,
and no additional GPU diagnostic is launched. Artifacts:
`artifacts/doom_reference_round9_initial_match_cpu_audit.json` and
`artifacts/doom_reference_round9_initial_call_shapes_cpu_review.json`.

The prepared real supervisor now requires an independent CPU reconstruction
of all complete validation records, candidate identities, strongest control
and frozen decision before it can dispatch any reserved test. It fingerprints
the independent auditor in the frozen selection. Six isolated CPU workflow
fixtures passed this ordering, including control retention and duplicate
initializer/best identities; these synthetic fixtures are not real results.
Both independent choice and no-test closure auditors refuse missing actual
reports. At a measured generation343/500 snapshot, the live search's best
1024-dream score was958.03125 at generation290; real performance is unproven.

The search subsequently completed500 generations: best **965.65234375** at
generation380, initial923.2109375 and final CMA mean892.94140625. Independent
CPU review reconstructed0–500 history and900/100 disjoint start indices,
checked the exact validation-selected initializer, arguments, inputs/source,
best/final weights,500-iteration optimizer/final mean, NumPy RNG restoration
and next JAX key. Canonical and prior controls remain intact. Artifact:
`artifacts/doom_reference_round9_training_cpu_audit.json`. Session83493 exited0
and both training processes exited. A separate CPU dispatch check verified
all current state fingerprints, actual process absence, an exclusive lock and
fresh130260–130339/180000–180099 cohorts before one serial real supervisor
started. Artifact: `artifacts/doom_reference_round9_real_dispatch_readiness_cpu.json`.
CUDA matrix multiplication and convolution backward preflight passed again.
Real session30040, supervisor1016081 and first evaluator1016151 were verified
live; first control had completed26/80 games at the measured snapshot. All
complete real cohorts, independent frozen selection and any eligible paired
reserved tests remain pending; no1092 performance is claimed.

The unchanged initializer control completed80 games at **1031.74 ±617.01**,
69 deaths/11 timeouts, fixed-policy bootstrap95% mean **[897.89,1168.05]**
(50,000 whole-game resamples, seed74180). Independent CPU checks verified
the actual records, registered cohort, controller/inference provenance,
source/input fingerprints and preserved canonical weights. Artifact:
`artifacts/doom_reference_round9_prior_control_validation_cpu_audit.json`.
This is a validation control, not a reserved100-game result; an interval
containing1092 does not prove the goal. The same supervisor advanced to supplied
control, evaluator1019526 verified live at45/80 games. Both new candidates and
the complete selection remain pending.

The supplied control subsequently completed80 games at **982.68 ±520.17**,
77 deaths/3 timeouts, fixed-policy mean95% **[870.09,1096.81]**. Independent
CPU reconstruction verified both controls' complete registered records,
current package versions, matching world/environment/source protocol and
canonical preservation. Prior-minus-supplied is **+49.06**, paired95%
**[−73.88,+172.28]**,33 wins/34 losses/13 ties. This does not establish a
reliable gain or complete candidate selection. Artifact:
`artifacts/doom_reference_round9_controls_validation_cpu_audit.json`.

The new best-generation380 checkpoint completed80 validations at
**961.74 ±501.45**,78 deaths/2 timeouts, mean95% **[853.01,1072.85]**.
Independent CPU reconstruction verified all three complete cohorts, actual
controller parameters, source/world fingerprints and canonical preservation.
New-best-minus-prior is **−70.00**, paired95% **[−207.94,+67.68]**; minus
supplied is **−20.94**, paired95% **[−138.49,+98.60]**. Artifact:
`artifacts/doom_reference_round9_best_validation_cpu_audit.json`. The same
supervisor is evaluating the final-generation500 checkpoint. Complete
validation selection and any eligible reserved tests remain pending.

Round9's final-generation500 checkpoint completed80 games at
**940.85 ±535.48**,77 deaths/3 timeouts, mean95% **[824.32,1058.61]**.
Its difference from the prior control is **−90.89**, paired95%
**[−217.69,+33.01]**. All four complete validation cohorts retained the
unchanged prior control1031.74. The synchronous independent selection audit
passed before the supervisor closed without a reserved test. Independent
closure reconstructed all320 real records, world/source/parameter identities,
prior initialization and freeze timing, and verified no180000–180099 games
were consumed. Separate current runtime, raw controller, saved training state,
canonical preservation and actual process-exit review passed. Session30040
returned0 and the supervisor and evaluator are absent. Artifacts:
`artifacts/doom_reference_round9_validation_closure_cpu_audit.json` and
`artifacts/doom_reference_round9_completion_cpu_audit.json`. This loop provides
no new1092 result; none of its validation intervals proves that target.

Round10 repeats the same initialization and search settings with training seed
**92**, changing only the seed and output path. This random-seed change includes
CMA samples, dream streams and the900/100 start-pool permutation together.
It is a stochastic repeat, not an isolated CMA-only intervention. The
initializer remains the round8 final-generation500 policy retained by all
complete round9 validation cohorts. The imported world, temperature1.15,
sigma0.005,500 generations,population64,fitness64,candidate batch64 and1024
held-out dream evaluations every10 stay fixed. Registered protocol:
`artifacts/doom_reference_round10_validation_initializer_protocol.json`, SHA256
`4d75e8c53d9ee6fdc5203ca5cd2b4796a0035beec3bf3753fd160c1aac07e2b7`.
CPU review verified the actual unchanged initializer/numerical trainer, exact
argument differences, prior closure/current process absence, exclusive lock,
disjoint new start pools and fresh real cohorts. Readiness artifact:
`artifacts/doom_reference_round10_search_readiness_cpu_audit.json`. The
independent training-state auditor defers missing results; syntax and Ruff pass.
One search started in session49055, supervisor1050296 and trainer1050365
verified live; CUDA matrix multiplication and convolution backward preflight
passed. It preserves best/final weights and complete optimizer/RNG state.
Fresh real validation130340–130419 must compare both controls and new
best/final policies with posterior inference. Freeze a distinct updated winner
and the highest validation control before any paired100-test190000–190099.
Retaining a control consumes no reserved test. No real gain is claimed from
this newly started search, and imported world weights remain explicit.

At a measured61/500 generation snapshot, the live initial-state CPU audit
verified exact seed92 arguments, unchanged initializer parameter hash,
frozen inputs/source/runtime, reconstructed900/100 pools and canonical
preservation. Initial1024-dream mean **870.5869140625** comes from the newly
registered seed/start/noise streams and does not measure training improvement
against seed91. GPU utilization measured95% with1905MiB resident at a training
snapshot. Artifact: `artifacts/doom_reference_round10_initial_match_cpu_audit.json`.
Both actual search processes remained live; complete training and real-game
evaluation are pending.

Round10's real supervisor and independent choice/retention auditors are now
prepared with the registered130340–130419 validation and190000–190099 test
cohorts. Seven isolated CPU fixtures verify retained controls consume no tests,
eligible candidate wins freeze the strongest control, identical vectors are
deduplicated, the independent audit runs before tests, and its rejection
blocks every reserved evaluation. Synthetic scores are not experiment results.
Actual readiness/choice/closure checks correctly defer missing completed
training and real reports. The real evaluator command, inference settings and
recovery code are unchanged. Syntax and Ruff pass. Artifact:
`artifacts/doom_reference_round10_real_workflow_preparation_cpu.json`. No real
GPU job is dispatched; the existing search remains live at a measured239/500
snapshot with best1024-dream954.8711 at generation1. Real performance remains
unproven.

Round10 subsequently completed500 generations: best1024-dream **954.87109375**
at generation1, initial870.5869140625 and final CMA mean953.3369140625.
Independent CPU audit reconstructed all0–500 history and seed92 start pools,
verified exact initializer/arguments/input/source, best/final policy weights,
500-iteration optimizer/final mean, NumPy state and next JAX key. Canonical
and prior policies remain intact. Artifact:
`artifacts/doom_reference_round10_training_cpu_audit.json`. Session49055 exited0
and both training processes exited. Current complete-state fingerprints,
process absence, exclusive lock and fresh cohorts passed CPU readiness:
`artifacts/doom_reference_round10_real_dispatch_readiness_cpu.json`.
One serial real supervisor then started in session87088; supervisor1068933
and evaluator1069003 were verified live with the exact unchanged-prior,
80-game130340–130419/posterior/eight-worker arguments. CUDA matrix
multiplication and convolution backward preflight passed. Full validation,
independent choice and any eligible reserved paired test remain pending.

The unchanged prior completed80 validation games at **936.94 ±547.92**,
76 deaths/4 timeouts, fixed-policy bootstrap95% mean **[818.01,1057.30]**
(50,000 whole-game resamples, seed74180). Independent CPU checks verified
complete registered records, controller/inference/source/input provenance,
current package versions and canonical preservation. Artifact:
`artifacts/doom_reference_round10_prior_control_validation_cpu_audit.json`.
This is control validation only and does not establish1092. The same supervisor
advanced to supplied control, evaluator1072617 verified live; both new policies
and complete selection remain pending.

The supplied control completed80 validations at **936.08 ±561.21**,
76 deaths/4 timeouts, fixed-policy mean95% **[814.65,1061.19]**. Independent
CPU reconstruction verified both controls' registered raw records, actual
source/input hashes, current package versions, shared world/environment
protocol and canonical preservation. Prior minus supplied is **+0.86**,
paired95% **[−126.09,+126.04]**,35 wins/33 losses/12 ties. No reliable
advantage is established. Artifact:
`artifacts/doom_reference_round10_controls_validation_cpu_audit.json`.
Both new policies and the complete validation-only selection remain pending.

The new best-generation1 checkpoint completed80 validations at
**915.18 ±529.99**,77 deaths/3 timeouts, mean95% **[800.90,1033.56]**.
Independent CPU reconstruction verified all three complete cohorts, registered
raw records, actual FP64 controller parameters and generation, source/world,
current packages and canonical preservation. Best minus prior is **−21.76**,
paired95% **[−109.93,+69.14]**; minus supplied is **−20.90**, paired95%
**[−137.96,+94.89]**. Artifact:
`artifacts/doom_reference_round10_best_validation_cpu_audit.json`. The final
checkpoint remains running; complete validation selection is not yet frozen.

Round10's final-generation500 checkpoint completed80 validations at
**929.45 ±573.13**,72 deaths/8 timeouts, mean95% **[807.46,1057.56]**.
Final minus prior is **−7.49**, paired95% **[−133.54,+116.25]**. Complete
validation retained the unchanged prior936.94 control; neither new policy won.
The independent frozen-choice audit passed. Closure reconstructed all320
actual records, source/world/parameter identities, initializer provenance,
freeze timing and no reserved190000–190099 consumption. Separate current
runtime, checkpoint/training-state fingerprints, canonical preservation and
actual process-exit review passed. Session87088 returned0; both processes are
absent and no work is queued. Artifacts:
`artifacts/doom_reference_round10_validation_closure_cpu_audit.json` and
`artifacts/doom_reference_round10_completion_cpu_audit.json`. The goal remains
unverified; these descriptive validation intervals do not prove1092.

After round10 closed, a controlled GPU diagnostic held weights, public world,
seed91 starts, validation key100091 and temperature1.15 fixed. One policy
with1024 trajectories scored **923.2109375**; two identical copies with2048
trajectories scored **910.623046875** per copy. Repeats within each call shape
and both duplicated slots matched exactly, while783/1024 survival outcomes
differed between shapes (mean difference **−12.587890625**). Independent CPU
checks verified raw scores/array hashes, controller/source/input fingerprints,
reconstructed keys and starts, and actual session/process exit. This establishes
current batch-shape sensitivity. The two-copy result does not reproduce the
historical two-different-policy901.1103515625 mean, so the full historical
discrepancy and any particular GPU-kernel mechanism remain unresolved.
Registered protocol and results:
`artifacts/doom_reference_batch_shape_protocol.json`,
`artifacts/doom_reference_batch_shape_diagnostic.json` and
`artifacts/doom_reference_batch_shape_diagnostic_cpu_audit.json`.
Session84793 exited0; no training or real games were run.

The new `scripts/tools/train_doom_reference_fixed_validation.py` preserves
all earlier source snapshots and makes every held-out candidate use the same
single-candidate engine shape as generation0. Population fitness remains
batched as before. CPU fixtures execute the actual evaluation function and
verify baseline/update shapes, exact shared starts and keys, unchanged training
calls including padding, and unchanged optimizer/key/checkpoint code outside
the explicit evaluation change. Syntax/Ruff and diff checks pass. Artifact:
`artifacts/doom_reference_fixed_validation_calls_cpu_audit.json`. Its actual
GPU call path still needs verification before registering a new search; no
real-game improvement is inferred from this correction.

The actual corrected evaluation function then passed GPU verification on one,
two and three identical policy copies. Every call used1024 trajectories; all
six raw score vectors matched the archived single-policy baseline exactly,
mean923.2109375. Independent CPU raw-score/source/input/runtime audit passed,
session48325 returned0 and its process exited. Artifacts:
`artifacts/doom_reference_fixed_validation_gpu.json` and
`artifacts/doom_reference_fixed_validation_gpu_cpu_audit.json`. This is a
comparison verification, not a newly trained policy or real result.

Round11 is a registered same-settings seed92 replay using the corrected
single-candidate held-out call shape. All training arguments except output
path stay unchanged. Protocol:
`artifacts/doom_reference_round11_validation_initializer_protocol.json`,
SHA256`bd9345a333726a5bf2d8f9ad1049ca43dbf3598220ebc268dc6e58a49234023e`.
Fresh posterior validation130420–130499 compares both controls and any
distinct new candidates; identities already failing complete round10 validation
are excluded. Only an eligible new winner gets paired100-test200000–200099
after an independent frozen-selection audit. Actual final parameter/population
history equality against round10 must be checked, not assumed. Readiness
verified current parent closure/process absence, original weights, fixed-
validation CPU/GPU evidence, exact recipe/source/inputs and fresh cohorts.
One search started in session55459, supervisor1123810/trainer1124156 verified
live after CUDA matrix multiplication/convolution backward preflight.

The live initial-state audit verified exact arguments, initializer, frozen
inputs/source/runtime metadata and the same seed92 start partition. Initial
dream mean887.478515625 nevertheless differs by+16.8916015625 from round10's
870.5869140625. Cause remains unresolved, so a pure selection-correction
causal attribution is unproven. At83/500, all83 population mean/best statistics
match the prior run exactly; full optimizer/parameter replay remains unproven.
Best held-out dream949.2373 at generation50 establishes no real gain. Artifact:
`artifacts/doom_reference_round11_initial_match_cpu_audit.json`.

The serial real workflow is prepared with the registered fresh cohorts and
prior-failed-policy exclusion. Nine CPU routing fixtures cover retained
controls, candidate wins, strongest-control pairing, identity deduplication,
one/all archived failed identities and rejection blocking reserved tests.
Synthetic scores are not experimental results. Actual readiness/choice/closure
guards defer missing completed training/real records. Numerical inference,
recovery and independent-audit ordering match the previous workflow aside
from artifact names; syntax/Ruff pass. Artifacts:
`artifacts/doom_reference_round11_real_workflow_preparation_cpu.json` and
`artifacts/doom_reference_round11_real_numeric_guard_cpu.json`. The search
remains the only GPU job; no real evaluation has started.

Round11 subsequently completed500 generations: best fixed-shape dream
**954.5556640625** at generation470, initial887.478515625 and final mean
915.376953125. Independent CPU audit verified0–500 history, exact inputs/
arguments/source, start pools, best/final parameters and flags, optimizer500,
NumPy state and next JAX key. All500 population mean/best statistics and the
final raw parameter vector match round10 exactly. The final is the archived
failed duplicate and is excluded; generation470's best is a distinct eligible
identity. These checks do not resolve the initial held-out score discrepancy
or prove any real gain. Artifact:
`artifacts/doom_reference_round11_training_cpu_audit.json`. Session55459
returned0 and both training processes exited. Complete fingerprints, actual
process absence, exclusive lease and fresh real cohorts passed readiness:
`artifacts/doom_reference_round11_real_dispatch_readiness_cpu.json`.

The sole GPU job is now one serial real supervisor, session15768,
supervisor1149459 and first evaluator1149804 verified live. It validates the
unchanged incumbent, public control and new generation470 policy on all80
fresh posterior-inference games130420–130499, with eight workers and
unchanged single-game calculation shapes. CUDA preflight passed. Duplicate
final weights are omitted. Independent complete-validation choice and any
eligible paired100-test200000–200099 remain pending; no1092 result is claimed.

The unchanged prior controller completed all80 validations at
**953.56 ±529.86**,76 deaths/4 timeouts, fixed-policy mean95%
**[839.74,1071.74]**. Independent CPU reconstruction verified registered raw
records, actual controller/source/input hashes, current package versions,
shared environment and canonical preservation. Artifact:
`artifacts/doom_reference_round11_validation_progress_1_cpu_audit.json`.
The same live supervisor advanced to the public control; generation470 and
complete validation selection remain pending. This is a control validation
cohort, not a reserved test or evidence of reaching1092.

The public control completed80 validations at **972.89 ±546.46**,
75 deaths/5 timeouts, fixed-policy mean95% **[854.02,1094.39]**. Independent
CPU checks verified both controls' raw records, actual parameter identities,
source/input hashes, current package versions, shared world/environment and
canonical preservation. Public minus prior is **+19.33**, paired95%
**[−94.28,+130.69]**,41 wins/30 losses/9 ties; no reliable advantage is
established. Artifact:
`artifacts/doom_reference_round11_validation_progress_2_cpu_audit.json`.
The same live supervisor advanced to generation470. Complete selection
remains pending; control validation and intervals containing1092 do not
establish the target.

The generation470 candidate completed80 validations at **963.89 ±626.80**,
72 deaths/8 timeouts, mean95% **[828.96,1101.34]**. Candidate minus public is
**−9.00**, paired95% **[−136.10,+121.86]**. Complete validation retained the
unchanged public control, so no reserved test was permitted. Independent
selection/no-test closure reconstructed all240 actual records, raw controller
identities, source/world fingerprints, initializer provenance and freeze timing;
200000–200099 remain unused. Separate current-runtime, training-state/hash,
canonical preservation and actual process/lease checks passed. Session15768
returned0; both real processes and old training processes exited, with no
additional GPU work queued. Artifacts:
`artifacts/doom_reference_round11_validation_closure_cpu_audit.json` and
`artifacts/doom_reference_round11_completion_cpu_audit.json`. Fixed validation
selected a different policy but did not establish better real transfer or1092.

Round12 is preregistered as matched temperature1.15 and1.10 controller searches
from the same public initializer, selected solely by complete round11 validation.
Both use seed91, sigma0.005,500 generations,pop64,fitness64,1024 held-out
rollouts and the unchanged fixed-validation trainer. Only temperature and output
paths differ between the arms. The paper's non-monotonic temperature results
motivate this comparison;1.10 is our hypothesis. This does not support a pure
temperature claim from comparisons with older differently initialized or
validated searches. Six previously failed checkpoint identities are excluded.
Fresh validation130500–130579 and contingent paired100-test210000–210099
passed CPU seed reservation. Immutable protocol:
`artifacts/doom_reference_round12_temperature_pair_protocol.json`, SHA256
`5b500de0d64169b3329fcda2fa311c252c4d28b6d043d015601d20f79d7fb45a`.
Serial launcher/readiness and independent auditors remain to be prepared;
no new GPU job has started or real improvement been established.

The round12 serial launcher and independent arm auditor subsequently passed
CPU fixtures covering actual trainer argument parsing, a complete synthetic
public-initializer capsule and corrupt initializer/pool/RNG/history/checkpoint
rejection, serial CUDA/preflight/CPU-audit ordering, live/incomplete-run rejection,
audit-failure stopping the next arm and completed-arm recovery without dispatch.
Synthetic capsules are not experimental results. Syntax/Ruff passed and helper
sources were frozen. Actual parent/source/runtime/initializer/fresh-cohort/lease
readiness passed. One supervisor started in session42213, PID1202880; CUDA
matrix multiplication/convolution backward preflight passed, then first1.15
trainer1203226 was verified live at7/500 with96% GPU utilization measured.
Independent live initialization checks verified actual arguments, public
validation-selected initializer, source/inputs/runtime/start pools and canonical
preservation at27/500. Initial dream mean928.3779296875 exactly matches
round8's recorded initial mean; this does not prove raw-trajectory or complete
training replay, nor real gain. Artifacts:
`artifacts/doom_reference_round12_training_preparation_cpu.json`,
`artifacts/doom_reference_round12_training_readiness_cpu_audit.json` and
`artifacts/doom_reference_round12_tau115_initial_cpu_audit_correction.json`.
The original initial audit is retained; its generic unresolved-difference note
was corrected separately because the measured difference is zero. The same
supervisor will independently audit the finished first arm before launching1.10.
Real workflow preparation and real survival evidence remain pending.

The 1.15 arm subsequently completed all 500 generations, with best held-out
dream score **948.20703125 at generation 70**. Its independent training audit
verified best/final checkpoints, every history row, optimizer count 500 and
RNG state, actual source/inputs/runtime and canonical preservation. Artifact:
`artifacts/doom_reference_round12_tau115_training_cpu_audit.json`. The same
supervisor passed the next CUDA preflight and started 1.10, verified live at
generation 23 on 2026-10-04T06:18:11Z. A separate live CPU check verified equal
public initializer, start partitions, random-stream metadata, precisions and
validation shapes across arms; it does not prove all random draws or outcomes
are identical. Artifact: `artifacts/doom_reference_round12_tau110_initial_cpu_audit.json`.

The two-arm real workflow passed twelve isolated CPU fixtures and was frozen
in `artifacts/doom_reference_round12_real_workflow_preparation_cpu.json`, SHA256
`3de4fae008c4a5526bb5e7df0d33679ab9da5c95e0ce3020be188376c6847a4d`.
Its supervisor handles both arms' best/final candidates and both controls,
identity/failure exclusions and complete-validation selection. Independent
selection must pass before reserved games; retained controls consume no test.
Fixtures exercised actual independent reconstruction, audit-rejection blocking,
no-test closure and paired100 analysis. Inference/recovery/audit-dispatch AST
checks match round11. These synthetic checks are not experimental results.
Real dispatch remains gated on both complete searches, actual training exit,
fresh seed/source/runtime readiness and exclusive GPU access. No real gain or
1092 result is established.

Both searches subsequently completed. Temperature 1.10 selected generation
310 with held-out dream score **1076.3115234375**. Actual training session
42213 exited 0 (tool chunk `515ec3`); both independent arm audits, current
source/input/checkpoint/optimizer/RNG/runtime/canonical checks and actual
process/lease checks passed. Artifacts:
`artifacts/doom_reference_round12_temperature_searches_result.json` and
`artifacts/doom_reference_round12_training_completion_cpu_audit.json`.
The 1.15 final raw parameters exactly equal the preserved prior control,
while its generation-70 best is distinct. After deduplication and previous-
failure exclusions, independent reconstruction registered five policies:
public and prior controls, 1.15 best, 1.10 best and 1.10 final. This is 400
validation games, not five independent reserved test cohorts.

Fresh readiness passed in
`artifacts/doom_reference_round12_real_dispatch_readiness_cpu.json` before
one real supervisor started in session 16340, PID 1253545. CUDA preflight
passed; evaluator 1253615 was verified live on the public control at 1/80
games on 2026-10-04T06:43Z. It uses eight workers with unchanged single-game
calculation shapes and per-seed RNG. Complete validation-only selection,
independent frozen-choice audit and any eligible reserved paired100 remain
pending; no real 1092 mean is established.

The public control completed all 80 validation games at **957.56 ±599.62**,
73 deaths / 7 timeouts, fixed-policy 95% mean interval **[828.66,1092.14]**.
Independent CPU checks reconstructed all raw records and verified actual
controller/source/input fingerprints, current packages, shared environment
and canonical preservation. Artifact:
`artifacts/doom_reference_round12_validation_progress_1_cpu_audit.json`.
The same supervisor moved to the prior local controller; evaluator 1261348
was verified live. Complete five-cohort selection remains pending. Neither
this unchanged-control validation nor an interval containing 1092 proves the target.

The prior local controller completed 80 validation games at **913.01 ±502.18**,
78 deaths / 2 timeouts, fixed-policy mean95% **[803.20,1024.40]**. Its paired
difference from public is **−44.55**, 95% **[−171.36,+84.65]**, with 25 wins /
42 losses / 13 ties. Independent checks verified both complete raw cohorts,
actual control identities, source/inputs, current packages, shared environment
and canonical preservation. Artifact:
`artifacts/doom_reference_round12_validation_progress_2_cpu_audit.json`.
The same supervisor advanced to the distinct 1.15 generation-70 candidate;
evaluator 1271193 was verified live. Three candidate cohorts and complete
selection remain pending. These control validation outcomes do not prove 1092.

The distinct 1.15 generation-70 candidate completed 80 validation games at
**926.31 ±579.07**, 75 deaths / 5 timeouts, mean95% **[801.87,1055.04]**.
Its paired difference from public is **−31.25**, 95% **[−141.41,+77.48]**,
with 33 wins / 30 losses / 17 ties. Independent checks verified all three
complete raw cohorts, actual policy identity/generation/temperature and
imported-world/own-update provenance, source/inputs/current packages/shared
environment and canonical preservation. Artifact:
`artifacts/doom_reference_round12_validation_progress_3_cpu_audit.json`.
The same supervisor advanced to temperature 1.10 best, evaluator 1281065
verified live. Both 1.10 cohorts and complete selection remain pending;
this candidate did not outscore public, and no reserved test has started.

Round12 subsequently completed all five cohorts, **400 raw validation
records** on seeds 130500–130579. Temperature 1.10 best scored **904.69
±524.83**, 78 deaths / 2 timeouts, mean95% **[791.72,1020.96]**; its paired
difference from public was **−52.88**, 95% **[−181.60,+75.13]**. Temperature
1.10 final scored **922.06 ±522.69**, 76 deaths / 4 timeouts, mean95%
**[810.02,1037.90]**; its paired difference was **−35.50**, 95%
**[−174.74,+103.65]**. The independently reconstructed frozen choice retained
the unchanged public control at **957.56 ±599.62**. No new policy won, so
reserved seeds **210000–210099 remain unused**. These intervals use 50,000
whole-game bootstrap resamples, seed 74180, with policies and world fixed;
they exclude training, selection and historical runtime uncertainty.

The no-test closure audit verified unused reserved seeds and complete
validation-only selection. The final CPU review verified current raw reports,
actual source/input/runtime/control/provenance fingerprints, both searches'
checkpoints and optimizer/RNG state, and canonical preservation. Actual real
session 16340 exited 0 (tool chunk `3c3cbf`); the review at
2026-10-04T07:47:48Z found all training/evaluation PIDs absent, the exclusive
GPU lease available and no further GPU work queued. Artifacts:
`artifacts/doom_reference_round12_frozen_selection_cpu_audit.json`,
`artifacts/doom_reference_round12_validation_closure_cpu_audit.json` and
`artifacts/doom_reference_round12_completion_cpu_audit.json`. The round is
complete and the 1092-step goal remains unmet. This is controller refinement
using an imported public world, with the previously documented engine,
preprocessing and RNG differences; it does not reproduce our own world-model
training or verify the exact checkpoint/runtime behind the paper's result.
The canonical own-world **840.06 ±524.48** result and checkpoints remain intact.

Round13 is a registered direct real-survival refinement rather than another
dream-temperature search. The public controller retained by complete round12
validation is its initializer; the VAE/RNN,1088-weight bias-free tanh[z,c,h]
architecture, posterior inference, preprocessing and real-game settings stay
fixed. Real simulator rewards now supply CMA fitness, a method change from
the paper's dream-only controller training. One search uses seed94, sigma0.005,
16 generations, population16 and four shared fitness seeds per generation.
Fitness seeds500000–500063 and training holdouts510000–510015 are separate
from fresh policy validation130580–130659 and reserved tests220000–220099.
The baseline and population means at generations4,8,12,16 use the same16
training holdouts; these repeated holdouts select training checkpoints and
cannot establish fresh policy performance. Maximum training is1104 games.

`src/doom_real_training.py` passes controller weights as data through the
original one-game reference actor and keeps eight game workers open across
populations. Each actual(candidate,seed) pair receives its own reset memory
and per-seed latent RNG, independent of scheduling. Raw completed pairs,
pending populations, optimizer/RNG journals, history and best/final policies
are retained. Seven CPU tests passed, including exact interrupted/uninterrupted
CMA and next-population agreement and independent full synthetic search
reconstruction. Actual-model CPU/GPU checks matched actions, hidden states,
keys and dtypes exactly for public and perturbed policies over the tested
three-step, eight-game inputs; they do not prove historical paper equivalence
or all possible trajectories. Artifacts:
`artifacts/doom_reference_round13_cpu_tests.log`,
`artifacts/doom_reference_round13_actor_cpu_audit.json` and
`artifacts/doom_reference_round13_actor_gpu_audit.json`.

Immutable protocol `artifacts/doom_reference_round13_direct_real_protocol.json`
has SHA256 `60b9198a566ece4d1f373856a8c580e05f7cc2ed6a4cf92577f77077be6bc639`.
Fresh dispatch readiness and CUDA preflight passed before one supervisor
started in session52524, PID1323498. Trainer1323841 was independently verified
live at2026-10-04T08:10:33Z with3/16 baseline holdout games recorded. The
supervisor runs `scripts/tools/audit_doom_reference_real_training.py` on CPU
after training exits; it independently reconstructs every candidate population
from recorded fitness, all holdout choices, final optimizer and next RNG draw.
All actual training evidence remains pending. Before any real validation,
prepare and verify its independent selection workflow, with both controls,
distinct eligible best/final policies, exclusion of the nine prior failed
identities and complete80-game cohorts. Only a new validation winner may be
frozen for100 reserved games plus100 paired-control games. The goal is unmet.

The real-comparison workflow subsequently passed **13 CPU tests**, including
independent choice reconstruction before any reserved job, audit rejection
preventing testing, highest-control pairing, ties/no-test closure, complete
cohort requirements, candidate identity/failure exclusions, valid partial
recovery, world drift, nested recorded seed reuse and an incomplete100-game
control test. The fixtures use temporary synthetic reports and are not
performance evidence. The frozen preparation artifact is
`artifacts/doom_reference_round13_real_workflow_preparation_cpu.json`, SHA256
`371a78c967b9a852d0ca4ddac899cdb87d7147fffa19e391bde0fabb21db82f0`,
fingerprinting13 workflow/dependency sources and the passed test log.
`scripts/tools/run_doom_reference_comparison.py` serially evaluates complete
fresh cohorts, then invokes `scripts/tools/audit_doom_reference_comparison.py`
on CPU before any eligible reserved pair. The auditor independently rebuilds
eligible identities and chooses from raw validation means; final analysis
requires both complete100-game reports and matching world/runtime/input/source
evidence. Startup requires actual training supervisor/session exit0 and the
current completed training audit. Actual check-only readiness deferred dispatch
while the existing trainer remained live; no real validation has started.

At2026-10-04T08:38:18Z, three CMA generations were complete; generation4 had
16/64 fitness games recorded. Generation3's population mean was860.96875 and
its best four-game fitness was1092.0. The training-holdout best was still the
public baseline768.5625, with its first new checkpoint comparison scheduled
after generation4. These training measurements do not establish the reserved
goal. Supervisor1323498 and trainer1323841 were independently verified live.

`src/doom_reference_selection.py` verifies matching validation cohorts and
distinct policies, prefers controls on ties, and permits reserved testing only
when an updated candidate wins validation. The CPU completion auditor accepts
`--protocol` to verify the preregistered80-game validation cohort,100-game test
cohort and fixed or validation-selected paired control. For round7 it
independently recomputes the highest validation control and rejects a
substituted frozen control, even with matching report hashes. Fixtures reject duplicate policies,
missing or mismatched games, wrong cohorts and substituted controls. An
isolated supervisor fixture also verified that retaining a control starts no
reserved test; its synthetic scores are not experiment results.

Previous commit validation: all49 tests passed on CPU, including dream rollout timing,
CMA/RNG restoration, captured-failure recovery and result-selection audits.
Ruff lint/format checks and `git diff --check` passed. The existing reserved
real-game evaluation continued independently; these checks launched no GPU work.
