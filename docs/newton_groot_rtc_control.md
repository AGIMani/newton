# Newton GR00T RTC control

`tools/run_newton_groot_rtc_control.py` extends the dual Nero + Linker L10
scene in `debug/import_dual_nero_linker_l10.py`. It loads the local
`checkpoint-200000`, produces 32-step action chunks, and replans with the RTC
action-head path used by `probe_l10_rtc_trimmed_chunks.py`.

## Local assets

The runner uses these project-local paths by default:

- `checkpoints/groot/checkpoint-200000`
- `checkpoints/nvidia/Cosmos-Reason2-2B`
- `local_data/groot/smooth`
- `logs/groot_newton_rtc/trace.jsonl`

The Isaac-GR00T Python source is resolved from the Newton sibling directory
`../Isaac-GR00T` by default, or from `ISAAC_GROOT_ROOT` when that environment
variable is set.

The checkpoint, VLM, smooth dataset, and generated traces are ignored by Git.

Use the existing Newton conda environment. Isaac-GR00T pins the image
processor packages below; newer incompatible versions fail while constructing
the checkpoint processor.

```bash
conda activate newton
python -m pip install albumentations==1.4.18 albucore==0.0.17
```

Validate the copied assets, checkpoint modality contract, and one smooth frame
without loading Newton or the model:

```bash
python tools/run_newton_groot_rtc_control.py --validate-only --episode-index 8
```

## Live simulator images

This mode uses the current Newton D455 and D405 RGB buffers together with the
current simulated arm, hand, and end-effector state:

```bash
python tools/run_newton_groot_rtc_control.py \
  --viewer gl \
  --device cuda:0 \
  --policy-device cuda:0 \
  --image-source sim \
  --state-source sim \
  --start-policy
```

The simulator defaults match node0's live policy inputs: raw `320x180`
`ego_view` RGB and raw `640x480` `wrist_view` RGB. The runner does not crop,
resize, pad, or letterbox either image before the checkpoint processor. Use
`--sim-ego-roi` only for the older 2x scene-camera crop during framing
diagnostics. Smooth images are always passed through unchanged, and the
checkpoint processor performs its own resize and crop.

## Smooth episode images

Use recorded images while retaining the current Newton robot state:

```bash
python tools/run_newton_groot_rtc_control.py \
  --viewer gl \
  --image-source smooth \
  --state-source sim \
  --episode-index 8 \
  --smooth-frame-offset 0 \
  --start-policy
```

For a fully recorded observation, also set `--state-source smooth`. The image
and state sources are intentionally independent so recorded perception can be
tested against either recorded or simulated proprioception. An empty
`--instruction` uses the selected episode task; pass `--instruction TEXT` to
override it.

## RTC and execution

Defaults match the validated L10 deployment settings: 10 Hz actions, replan
every 8 executed actions, at most 24 overlap steps, 4 frozen steps, and an RTC
ramp rate of 3.0. Use `--no-rtc` for ordinary chunk replanning. Use
`--dry-run-policy` to test the Newton control loop without loading the model.

The default `--arm-control-mode eef_ik` treats decoded `eef_9d` as an absolute
target in the checkpoint's policy-state frame. It uses node0's fixed
`A * T_policy * B` transform to produce a Newton world
`/right_revo2_flange` target: `A` has translation `(0, 0.059, 0.918)` and
quaternion `(0.5, 0.5, 0.5, -0.5)` in XYZW order; `B` has translation
`(0.032, 0, -0.0235)` and quaternion `(-0.5, 0.5, -0.5, 0.5)`. The result is
then mapped from Genesis world to Newton world using the current
`/right_base_link` pose. This explicit scene alignment preserves node0's
transform while accounting for the Newton assembly's different base height.
The target is applied through `NewtonLinkKinematicsModel` and the full-pose
differential IK controller. `arm_joint_target` is used only if EEF IK fails. Use
`--arm-control-mode joint_target` to select the old direct joint-alias path, or
`--no-arm-joint-fallback` to make an IK failure stop execution.

The Nero/L10 `rot6d` layout now exactly follows node0's live GR00T bridge: the
first two rotation-matrix rows in row-major order,
`[R00, R01, R02, R10, R11, R12]`. State encoding and decoded EEF actions use
the same convention.

This runner also selects the checkpoint/Harness right-arm initial pose instead
of the generic debug scene pose. The left arm continues to use the URDF initial
state. Pass the inherited `--initial-right-arm-q q1,...,q7` option only when
deliberately evaluating a different starting configuration.

The right L10 hand is initialized from the Harness checkpoint command pose so
its simulated reported state and wrist image begin inside the training
distribution. Override it with `--groot-initial-hand-q q1,...,q10` only for an
intentional state-distribution test.

For the Newton pinhole wrist camera, the runner uses a `72` degree vertical FOV
and a small connector-frame optical-axis correction. Together these reproduce
the D405's wider horizontal field and keep the nearby bottle in the lower-right
region seen in training. The generic scene's D405 body mount remains unchanged.

The node0 transform is fixed and does not depend on the first simulator
observation. Use `--eef-transform-mode initial_calibration` to reproduce the
older dynamic alignment; in that compatibility mode, `--eef-frame-update replan`
recalibrates at every chunk. Per-tick arm and hand changes are bounded
by `--max-arm-joint-step` and `--max-hand-joint-step`. The trace records
policy/world EEF targets, current world TCP, IK status, position/orientation
error, and the actual arm control source for every executed action.

Policy execution is disabled until `--start-policy` is supplied. Every replan
and executed target is written to the JSONL trace unless `--no-policy-trace`
is set. For a bounded smoke test, add `--max-policy-steps 9`; step 8 performs
the first RTC replan with a previous action chunk.

Use `--dump-first-observation-dir PATH` to save the exact current `ego_view`
and `wrist_view` RGB arrays passed to the checkpoint processor. This is useful
for checking simulator/training camera alignment without changing image
preprocessing.

## Docker runtime

Build the GR00T inference layer on top of the existing direct-GPU image:

```bash
docker/build_groot_rtc.sh
```

On RTX 5090 hosts, `docker/run_groot_rtc.sh` automatically prefers the mounted
`conda_envs/newton` Python runtime because its CUDA 12.8 PyTorch build includes
`sm_120`. Set `NEWTON_GROOT_PYTHON` only when intentionally selecting another
Python runtime inside the container.

Run with live Newton images on GPU 0:

```bash
NEWTON_GROOT_GPU=0 docker/run_groot_rtc.sh \
  --viewer gl \
  --image-source sim \
  --state-source sim \
  --start-policy
```

For `--viewer gl`, run from the node3 desktop terminal so its current
`DISPLAY` and `.Xauthority` are available. The wrapper forwards both into the
container; do not hard-code a display number unless that X socket exists.

Run a recorded episode without opening a viewer:

```bash
NEWTON_GROOT_GPU=0 docker/run_groot_rtc.sh \
  --viewer null \
  --image-source smooth \
  --state-source smooth \
  --episode-index 8 \
  --start-policy
```
