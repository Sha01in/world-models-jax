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
not resolve the full real-game evaluation failure described above.
