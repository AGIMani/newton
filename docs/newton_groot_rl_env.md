<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Newton GR00T reinforcement-learning environment

`GrootNewtonEnv` is a headless, batched PickBottle environment for the dual
Nero robot, right Linker L10 hand, table scene, and dynamic bottle. It loads
`assets/generated/dual_nero_linker_l10_combined.urdf` and replicates the scene
as independent Newton worlds inside one CUDA model.

The interface follows the parts of ManiSkill that are useful to GPU robot
learning:

- all values retain a leading `num_envs` batch dimension;
- `reset()` and `step()` return CUDA Torch tensors;
- `step()` returns the Gymnasium five-tuple
  `(obs, reward, terminated, truncated, info)`;
- `obs_mode`, `control_mode`, and `reward_mode` select the corresponding
  interface behavior;
- `reset(options={"env_idx": indices})` performs a partial GPU reset;
- `single_action_space`, `action_space`, `single_observation_space`, and
  `observation_space` describe the unbatched and batched values when
  Gymnasium is installed.

This is based on ManiSkill's [GPU simulation
contract](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/gpu_simulation.html),
[observation organization](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html),
and [Gymnasium quickstart](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html).
It is an API-compatible design, not a dependency on ManiSkill or SAPIEN.

The bottle currently starts at the same fixed pose in every world. Scene and
object randomization are intentionally left for a later task.

## Files

- `teleop_stack/envs/groot_newton_env.py`: environment, control, task reward,
  and GPU kernels.
- `teleop_stack/envs/groot_diffusion_policy_env.py`: observation-history and
  action-chunk adapter.
- `teleop_stack/envs/groot_newton_vector_env.py`: PPO auto-reset and terminal
  transition wrapper.
- `tools/run_newton_groot_rl_env.py`: bounded rollout and throughput runner.
- `docker/run_groot_rl_env.sh`: headless Docker launcher.
- `debug/import_dual_nero_linker_l10.py`: shared scene construction.

## Starting the environment

Run a headless rollout in the GR00T Docker image:

```bash
docker/run_groot_rl_env.sh \
  --num-envs 8 \
  --steps 100 \
  --obs-mode state_dict+rgb \
  --control-mode pd_joint_delta_pos \
  --reward-mode normalized_dense
```

Select a host GPU with `NEWTON_GROOT_RL_GPU`. The device is exposed as
`cuda:0` inside the container:

```bash
NEWTON_GROOT_RL_GPU=1 docker/run_groot_rl_env.sh --num-envs 8 --steps 100
```

For a state-only interface check:

```bash
docker/run_groot_rl_env.sh \
  --num-envs 2 \
  --steps 10 \
  --obs-mode state \
  --no-images \
  --no-scene-visuals \
  --no-hydroelastic
```

## ManiSkill-style Python interface

The constructor accepts either a `GrootNewtonEnvConfig` or ManiSkill-like
keyword options:

```python
import torch

from teleop_stack.envs import GrootNewtonEnv

env = GrootNewtonEnv(
    num_envs=64,
    device="cuda:0",
    obs_mode="state_dict+rgb",
    control_mode="pd_joint_delta_pos",
    reward_mode="normalized_dense",
    max_episode_steps=100,
)

obs, info = env.reset(seed=0)
action = torch.zeros((env.num_envs, 17), device="cuda:0")
obs, reward, terminated, truncated, info = env.step(action)
done = terminated | truncated

# ManiSkill-style partial reset. torch.where remains on CUDA.
env_idx = torch.where(done)[0]
obs, info = env.reset(options={"env_idx": env_idx})
env.close()
```

`step()` and `reset()` are the network-facing Torch API. `step_warp()`,
`reset_warp()`, and `observation_warp()` expose the underlying Warp arrays for
custom CUDA code. Torch views are created with Warp interoperability and do
not copy observation storage.

The base environment does not automatically reset completed worlds or clone a
`final_observation`. This keeps the Diffusion Policy path lean. PPO can either
store the terminal transition and call the partial reset shown above, or use
the vector wrapper described below.

## Action and controller interface

Every action is `float32`, has shape `[num_envs, 17]`, and is normalized to
`[-1, 1]`. This matches ManiSkill controllers and the assertion made by its
Diffusion Policy baseline. Components are ordered as follows:

1. `right_joint1` through `right_joint7`;
2. `thumb_cmc_pitch`, `thumb_cmc_yaw`;
3. `index_mcp_pitch`, `middle_mcp_pitch`, `ring_mcp_pitch`,
   `pinky_mcp_pitch`;
4. `index_mcp_roll`, `ring_mcp_roll`, `pinky_mcp_roll`;
5. `thumb_cmc_roll`.

Supported control modes are:

| Mode | Meaning |
| --- | --- |
| `pd_joint_delta_pos` | Add normalized deltas to the current joint positions. Arm and hand deltas default to `0.1 rad` at `|action|=1`. This is the recommended Diffusion Policy and PPO mode. |
| `pd_joint_pos` | Map `[-1, 1]` over each imported URDF joint-position limit. |

Targets are clipped to the URDF limits and target velocity is set to zero.
`hold_action_torch()` returns zero for the delta controller and the normalized
current position for the absolute controller.

One `step()` is a 10 Hz control interval by default. The action is held for
six 60 Hz frames, each containing sixteen physics substeps, for 96 physics
substeps per policy action.

## Observation modes

All observations are CUDA Torch tensors with a leading batch dimension.

### `state_dict`

This follows ManiSkill's `agent` and task-specific `extra` split:

```text
agent
  qpos                 [N, 17]  float32
  qvel                 [N, 17]  float32
  qfrc_actuator        [N, 40]  float32
  arm_joint_pos        [N, 7]   float32
  hand_joint_pos       [N, 10]  float32
extra
  tcp_pose              [N, 7]   float32
  obj_pose              [N, 7]   float32
  goal_pos             [N, 3]   float32
  tcp_to_obj_pos       [N, 3]   float32
  obj_to_goal_pos      [N, 3]   float32
  is_grasped           [N]      bool
  eef_9d               [N, 9]   float32
```

`qfrc_actuator` contains the latest MuJoCo actuator generalized forces for all
40 DOFs of the current combined URDF. It is not a direct contact force or
contact-point wrench.

### `state`

Returns the compact `[N, 66]` policy state used before this ManiSkill adapter:

```text
[eef_9d (9), hand_joint_pos (10), arm_joint_pos (7), qfrc_actuator (40)]
```

### `rgb` and `state_dict+rgb`

`rgb` returns real-machine-available agent state, TCP/goal data, and
`sensor_data`. `state_dict+rgb` additionally includes the privileged bottle
pose and relative task vectors from `state_dict`.

```text
sensor_data
  ego_view/rgb         [N, 180, 320, 3]  uint8
  wrist_view/rgb       [N, 480, 640, 3]  uint8
```

RGB is intentionally left as `uint8`, as in ManiSkill, to reduce observation
memory and bandwidth. Normalize it in the network encoder.

### `policy`

This compact mode is convenient without a wrapper:

```text
state                    [N, 66]
rgb/ego_view             [N, 180, 320, 3]
rgb/wrist_view           [N, 480, 640, 3]
```

## Diffusion Policy adapter

ManiSkill's official Diffusion Policy predicts a sequence shaped
`[batch, prediction_horizon, action_dim]`, selects an action horizon, and
calls the base environment once for each selected action. Its common defaults
are observation horizon 2, action horizon 8, and prediction horizon 16. See
the official [training
code](https://github.com/mani-skill/ManiSkill/blob/main/examples/baselines/diffusion_policy/train_rgbd.py)
and [evaluation loop](https://github.com/mani-skill/ManiSkill/blob/main/examples/baselines/diffusion_policy/diffusion_policy/evaluate.py).

`GrootDiffusionPolicyEnv` implements that contract and also accepts a complete
action chunk directly:

```python
import torch

from teleop_stack.envs import GrootDiffusionPolicyEnv, GrootNewtonEnv

base_env = GrootNewtonEnv(
    num_envs=64,
    obs_mode="policy",
    control_mode="pd_joint_delta_pos",
    reward_mode="sparse",
)
env = GrootDiffusionPolicyEnv(base_env, obs_horizon=2, action_horizon=8)

obs, info = env.reset()
# obs["state"]: [64, 2, 66]
# obs["rgb"]["ego_view"]: [64, 2, 180, 320, 3]

with torch.no_grad():
    action_chunk = policy(obs)  # [64, T, 17], 1 <= T <= 8
obs, reward, terminated, truncated, info = env.step(action_chunk)
```

The history uses two preallocated CUDA buffers, so shifting the observation
window does not allocate a new history tensor every step. For action chunks,
the adapter sums rewards until each world first reports done and returns
`info["action_chunk"]["executed_steps"]`.

The adapter also exposes `single_action_space`, `action_space`,
`single_observation_space`, and `observation_space`. The observation spaces
include the state-history dimension expected by ManiSkill's Diffusion Policy
`Agent`, while scalar image bounds avoid allocating dense CPU bound arrays.
The two cameras remain nested because their resolutions differ; the visual
encoder should encode each view separately and fuse the resulting features.

Avoid crossing an episode boundary inside a chunk when the exact terminal
observation is needed. PPO should use the base environment one action at a
time; action chunks are intended for Diffusion Policy inference.

## PickBottle reward and PPO

The reward follows ManiSkill's [PickCube
task](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/pick_cube.py):

1. Reaching: `1 - tanh(5 * ||tcp - bottle||)`.
2. Grasping: add `1` when at least two distinct L10 fingers contact the
   bottle.
3. Placing: while grasped, add
   `1 - tanh(5 * ||bottle - goal||)`.
4. Static: once placed, add
   `1 - tanh(5 * velocity_norm)`.
5. Success: replace the dense reward with `5`.

The fixed goal is the reset bottle position plus `0.1 m` in Z. The bottle is
placed when it is within `0.025 m` of that goal. Success additionally requires
the controlled robot and bottle linear velocity norm to be below `0.2`.
`is_grasped` is a cheap GPU contact-topology heuristic, not a force-closure
proof and not a threshold on `qfrc_actuator`.

Reward modes are:

| Mode | Reward |
| --- | --- |
| `normalized_dense` | Dense reward divided by 5; recommended for PPO. |
| `dense` | Unnormalized reward in the PickCube-style scale. |
| `sparse` | `1` on success, otherwise `0`; useful for Diffusion Policy evaluation. |
| `none` | Always zero. |

`terminated` is success when `terminate_on_success=True`. `truncated` is the
time limit, which defaults to 100 policy steps. The `info` dictionary includes
`success`, `fail`, `is_grasped`, `is_obj_placed`, `is_robot_static`, reward
components, and episode return/length/success metrics.

ManiSkill's PPO baseline uses normalized dense rewards, batched CUDA rollout
buffers, partial resets, and bootstraps time-limit truncations from the final
observation. See the official [PPO
implementation](https://github.com/mani-skill/ManiSkill/blob/main/examples/baselines/ppo/ppo.py)
and [baseline guide](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/baselines.html).

For this environment, a PPO runner should use `obs_mode="state"` for the
fastest baseline or an image encoder with `obs_mode="policy"`, store rollout
tensors on CUDA, distinguish `terminated` from `truncated` during bootstrap,
and partial-reset completed world indices after saving their terminal values.

`GrootNewtonVectorEnv` provides the subset of ManiSkill's vector wrapper used
by its PPO code:

```python
import torch

from teleop_stack.envs import GrootNewtonEnv, GrootNewtonVectorEnv

base_env = GrootNewtonEnv(
    num_envs=512,
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    reward_mode="normalized_dense",
)
env = GrootNewtonVectorEnv(
    base_env,
    auto_reset=True,
    ignore_terminations=False,
    record_metrics=True,
)

obs, info = env.reset(seed=0)
action = torch.zeros((env.num_envs, 17), device="cuda:0")
obs, reward, terminated, truncated, info = env.step(action)

# These keys match the official PPO rollout expectations.
done_mask = info["_final_info"]
final_obs = info["final_observation"]
episode_metrics = info["final_info"]["episode"]
```

The wrapper clones the small reward/done arrays before reset and lazily
allocates one terminal-observation buffer. For image PPO this buffer is large;
use the base environment with explicit partial resets if the trainer already
owns terminal image storage.

## GPU and CPU boundary

Steady-state numerical work remains on CUDA:

- Newton collision and hydroelastic contact generation;
- MJWarp dynamics and `qfrc_actuator`;
- normalized action decoding and PD target writes;
- state/TCP/bottle extraction, multi-finger contact classification, reward,
  success, and episode metrics;
- partial reset, BVH updates, tiled camera ray tracing, and RGB unpacking;
- Diffusion Policy observation-history buffers and action chunks.

There is still light CPU scheduling for Python, CUDA launches, the six-frame
control loop, and dictionary construction. Startup also performs URDF/GLB/JSON
parsing, MuJoCo model compilation, Warp JIT, and small metadata reads on the
CPU. Neither `step()` nor `reset()` calls `.numpy()` or copies observations to
host memory.

The standalone runner calls `wp.synchronize_device()` before and after its
timed loop. These runner-only synchronizations are not part of the environment
API. Training code should avoid `.numpy()`, `.cpu()`, per-step synchronization,
and image logging.

At the default resolutions, packed and RGB camera buffers use roughly
2.44 MiB per world before ray-tracer workspace and physics state.
`mujoco_njmax`, `mujoco_nconmax`, `rigid_contacts_per_env`, image resolution,
and Diffusion Policy observation horizon all affect capacity. The `state` and
`state_dict` modes skip visual asset loading, camera allocation, and ray
tracing. In image observation modes, `--no-images` skips sensor creation and
ray tracing but retains zero-filled RGB buffers so the schema remains stable.

The environment does not write rollout, image, contact, or policy logs to
disk.
