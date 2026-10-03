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
These are dream scores; independent training-state review and real validation
remain pending. The search supervisor and child have exited.

The frozen protocol reserves80 fresh validation seeds130100–130179 for the
new best/final policies, supplied control and frozen1.15 generation270 control.
Round6's retained supplied policy adds no distinct control. Identical parameter
sets are deduplicated, and controls win ties. Only a distinct new candidate
winning validation gets100 reserved games160000–160099, paired with the
highest validation control whose identity is frozen before test outcomes.
Retaining a control starts no reserved test. No round7 real evaluation has
started at this documentation snapshot.

Protocol: `artifacts/doom_reference_round7_fine_search_protocol.json`, SHA256
`9f9cb63038ea87a0926d770a4ed6ca5aad95693840fefe01c41ab383e867f5e8`.
Search result: `artifacts/doom_reference_round7_fine_search_result.json`.
This remains own-controller training on imported public VAE/RNN weights,
with modern runtime differences; the1092-step goal is unproven.

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
