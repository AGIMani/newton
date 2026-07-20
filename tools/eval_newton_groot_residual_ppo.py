#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministically evaluate and visualize frozen Groot DP or residual PPO."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from teleop_stack.envs import GrootDiffusionPolicyEnv, GrootNewtonEnv, GrootNewtonEnvConfig
from teleop_stack.policies import (
    GROOT_RESIDUAL_PPO_CHECKPOINT_FORMAT,
    GrootResidualActorCritic,
    GrootResidualActorCriticConfig,
)
from tools.train_newton_groot_residual_ppo import (
    _PHASE_NAMES,
    _evaluate,
    _file_sha256,
    _load_frozen_dp,
    _validate_frozen_dp_training_contract,
    _validate_resume_training_contract,
)

_NOISE_SEED_OFFSET = 10_003
_VIDEO_WIDTH = 1_280
_VIDEO_HEIGHT = 540


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Frozen Diffusion Policy checkpoint")
    parser.add_argument(
        "residual_checkpoint",
        type=Path,
        help="Residual PPO policy, or the matched evaluation contract when --policy-mode=pure-dp",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--record-rollouts", type=int, default=6)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--policy-mode", choices=("residual", "pure-dp"), default="residual")
    parser.add_argument(
        "--selected-episodes-json",
        type=Path,
        help="Optional selected_episodes.json whose episode IDs should be recorded for paired comparison",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if args.record_rollouts < 0:
        raise ValueError("record_rollouts cannot be negative")
    if args.video_fps < 1:
        raise ValueError("video_fps must be positive")


def _episode_locations(episodes: int, num_envs: int) -> list[tuple[int, int]]:
    """Return each episode's ``(wave, lane)`` under :class:`_EvaluationQuota`."""
    if episodes < 1 or num_envs < 1:
        raise ValueError("episodes and num_envs must be positive")
    locations: list[tuple[int, int]] = []
    completed = 0
    wave = 0
    while completed < episodes:
        count = min(episodes - completed, num_envs)
        offset = (wave * count) % num_envs
        lanes = sorted((index + offset) % num_envs for index in range(count))
        locations.extend((wave, lane) for lane in lanes)
        completed += count
        wave += 1
    return locations


def _build_episode_noise_contract(
    *,
    episodes: int,
    num_envs: int,
    pred_horizon: int,
    action_dim: int,
    seed: int,
    device: Any,
) -> tuple[Any, list[Any], list[tuple[int, int]]]:
    """Build independently reproducible per-wave DP noise and scheduler states."""
    import torch

    locations = _episode_locations(episodes, num_envs)
    bank = torch.empty((episodes, pred_horizon, action_dim), dtype=torch.float32, device=device)
    wave_states: list[Any] = []
    episode_start = 0
    wave_count = math.ceil(episodes / num_envs)
    for wave in range(wave_count):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + _NOISE_SEED_OFFSET + wave)
        full_wave_noise = torch.randn(
            (num_envs, pred_horizon, action_dim),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        wave_states.append(generator.get_state().cpu().clone())
        episode_end = min(episode_start + num_envs, episodes)
        wave_locations = locations[episode_start:episode_end]
        for episode_id, (_, lane) in enumerate(wave_locations, start=episode_start):
            bank[episode_id].copy_(full_wave_noise[lane])
        episode_start = episode_end
    return bank, wave_states, locations


def _candidate(
    records: list[dict[str, Any]],
    predicate: Any,
    key: Any,
) -> dict[str, Any] | None:
    candidates = [record for record in records if predicate(record)]
    return None if not candidates else max(candidates, key=lambda record: (*key(record), -record["episode_id"]))


def _select_representative_episodes(
    records: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Select stable, deduplicated rollouts spanning the observed outcomes."""
    if limit <= 0 or not records:
        return []
    ordered = sorted(records, key=lambda record: record["episode_id"])
    selected: dict[int, dict[str, Any]] = {}

    def add(tag: str, record: dict[str, Any] | None) -> None:
        if record is None:
            return
        episode_id = int(record["episode_id"])
        if episode_id in selected:
            selected[episode_id]["tags"].append(tag)
        elif len(selected) < limit:
            selected[episode_id] = {**record, "tags": [tag]}

    def lift_key(record: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(record["max_contacted_carry_lift_height_m"]),
            float(record["max_physical_lift_height_m"]),
            float(record["return"]),
        )

    def physical_lift_key(record: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(record["max_physical_lift_height_m"]),
            float(record["max_contacted_carry_lift_height_m"]),
            float(record["return"]),
        )

    def return_key(record: dict[str, Any]) -> tuple[float, float]:
        return (float(record["return"]), float(record["max_physical_lift_height_m"]))

    add("best_success", _candidate(ordered, lambda record: record["success"], return_key))
    add("top_contacted_lift", _candidate(ordered, lambda _record: True, lift_key))
    add("top_physical_lift", _candidate(ordered, lambda _record: True, physical_lift_key))
    add("top_return", _candidate(ordered, lambda _record: True, return_key))
    add("best_grasp", _candidate(ordered, lambda record: record.get("event_grasp_ever", False), lift_key))
    add(
        "early_release",
        _candidate(ordered, lambda record: record.get("event_early_release_ever", False), lift_key),
    )
    add(
        "contact_without_grasp",
        _candidate(
            ordered,
            lambda record: record.get("event_contact_ever", False) and not record.get("event_grasp_ever", False),
            return_key,
        ),
    )
    median_lift = float(np.median([record["max_physical_lift_height_m"] for record in ordered]))
    median_record = min(
        ordered,
        key=lambda record: (abs(float(record["max_physical_lift_height_m"]) - median_lift), record["episode_id"]),
    )
    add("median_lift", median_record)
    add("lowest_return", min(ordered, key=lambda record: (float(record["return"]), record["episode_id"])))

    for record in sorted(ordered, key=lambda item: (-float(item["return"]), item["episode_id"])):
        if len(selected) >= limit:
            break
        add("return_rank_fill", record)
    return list(selected.values())


def _select_referenced_episodes(
    records: list[dict[str, Any]],
    references: Any,
    limit: int,
) -> list[dict[str, Any]]:
    """Select the referenced episode IDs while retaining current-run metrics."""
    if limit <= 0:
        return []
    if not isinstance(references, list):
        raise ValueError("selected episode reference must contain a JSON list")
    records_by_id = {int(record["episode_id"]): record for record in records}
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for reference in references:
        if not isinstance(reference, dict) or "episode_id" not in reference:
            raise ValueError("each selected episode reference must contain episode_id")
        episode_id = int(reference["episode_id"])
        if episode_id in seen:
            raise ValueError(f"selected episode reference duplicates episode_id {episode_id}")
        if episode_id not in records_by_id:
            raise ValueError(f"selected episode reference contains unavailable episode_id {episode_id}")
        tags = reference.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) and tag for tag in tags):
            raise ValueError(f"selected episode {episode_id} tags must be a list of non-empty strings")
        selected.append({**records_by_id[episode_id], "tags": ["matched_rl", *tags]})
        seen.add(episode_id)
        if len(selected) >= limit:
            break
    return selected


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _record_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for output_name, record_name in (
        ("return", "return"),
        ("contacted_carry_lift_m", "max_contacted_carry_lift_height_m"),
        ("physical_lift_m", "max_physical_lift_height_m"),
    ):
        values = np.asarray([record[record_name] for record in records], dtype=np.float64)
        result[output_name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "standard_error": float(values.std() / math.sqrt(len(values))),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "max": float(values.max()),
        }
    return result


def _wilson_interval(successes: int, episodes: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (proportion + z * z / (2.0 * episodes)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / episodes + z * z / (4.0 * episodes**2))
    radius /= denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _put_text(image: np.ndarray, text: str, origin: tuple[int, int], *, scale: float = 0.55) -> None:
    import cv2  # noqa: PLC0415

    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _compose_video_frame(
    observation: dict[str, Any],
    info: dict[str, Any],
    *,
    lane: int,
    episode_id: int,
    tags: list[str],
    step: int,
    row_index: int,
    policy_mode: str,
) -> np.ndarray:
    import cv2  # noqa: PLC0415

    ego = observation["observation.images.ego_view"][lane, -1].detach().cpu().numpy()
    wrist = observation["observation.images.wrist_view"][lane, -1].detach().cpu().numpy()
    if ego.ndim != 3 or ego.shape[-1] != 3 or wrist.ndim != 3 or wrist.shape[-1] != 3:
        raise ValueError(f"Expected RGB camera tensors, got ego={ego.shape}, wrist={wrist.shape}")
    ego = cv2.resize(ego, (640, 360), interpolation=cv2.INTER_AREA)
    wrist = cv2.resize(wrist, (640, 480), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((_VIDEO_HEIGHT, _VIDEO_WIDTH, 3), dtype=np.uint8)
    canvas[60:540, :640] = wrist
    canvas[120:480, 640:] = ego

    phase_index = int(info["task_phase"][lane])
    phase = _PHASE_NAMES[phase_index] if 0 <= phase_index < len(_PHASE_NAMES) else str(phase_index)
    episode_return = float(info["episode"]["return"][lane])
    physical_lift = float(info["physical_max_lift_height"][lane])
    contacted_lift = float(info["max_contacted_carry_lift_height"][lane])
    header = (
        f"policy {policy_mode}  episode {episode_id:03d}  step {step + 1:03d}  row {row_index}  phase {phase}  "
        f"return {episode_return:.2f}  tags {','.join(tags)}"
    )
    flags = (
        f"contact={int(info['has_hand_contact'][lane])} grasp={int(info['is_grasped'][lane])} "
        f"transport={int(info['transport_started'][lane])} lift={int(info['is_lifted'][lane])} "
        f"early_release={int(info['early_release'][lane])} success={int(info['success'][lane])} "
        f"max_lift={physical_lift:.4f}m contacted_lift={contacted_lift:.4f}m"
    )
    _put_text(canvas, header, (12, 24), scale=0.50)
    _put_text(canvas, flags, (12, 49), scale=0.50)
    _put_text(canvas, "wrist_view", (12, 84))
    _put_text(canvas, "ego_view", (652, 144))
    return canvas


class _ProgressCallback:
    def __init__(self, episodes: int) -> None:
        self.episodes = episodes
        self.completed = 0

    def __call__(self, *, active: Any, done: Any, **_kwargs: Any) -> None:
        accepted = int((active & done).sum())
        if accepted:
            self.completed += accepted
            print(f"evaluation progress: {self.completed}/{self.episodes} episodes", flush=True)


class _VideoCallback:
    def __init__(self, output_dir: Path, selected: list[dict[str, Any]], fps: int, policy_mode: str) -> None:
        import imageio.v2 as imageio  # noqa: PLC0415

        self._imageio = imageio
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.selected = {int(record["episode_id"]): record for record in selected}
        self._writers: dict[int, Any] = {}
        self._temporary_paths: dict[int, Path] = {}
        self._final_paths: dict[int, Path] = {}
        self._frame_counts: dict[int, int] = dict.fromkeys(self.selected, 0)
        self._fps = fps
        self._policy_mode = policy_mode
        self._closed = False

    @staticmethod
    def _slug(tags: list[str]) -> str:
        return re.sub(r"[^a-z0-9_-]+", "-", "_".join(tags).lower()).strip("-") or "representative"

    def _writer(self, episode_id: int) -> Any:
        writer = self._writers.get(episode_id)
        if writer is not None:
            return writer
        tags = self.selected[episode_id]["tags"]
        final_path = self.output_dir / f"episode_{episode_id:03d}_{self._slug(tags)}.mp4"
        temporary_path = final_path.with_name(f"{final_path.stem}.tmp.mp4")
        writer = self._imageio.get_writer(
            temporary_path,
            fps=self._fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
            ffmpeg_log_level="error",
        )
        self._writers[episode_id] = writer
        self._temporary_paths[episode_id] = temporary_path
        self._final_paths[episode_id] = final_path
        return writer

    def __call__(
        self,
        *,
        wave: int,
        step: int,
        episode_ids: Any,
        active: Any,
        observation: dict[str, Any],
        info: dict[str, Any],
        row_index: Any,
        **_kwargs: Any,
    ) -> None:
        for episode_id, record in self.selected.items():
            if int(record["wave"]) != wave:
                continue
            lane = int(record["lane"])
            if not bool(active[lane]) or int(episode_ids[lane]) != episode_id:
                continue
            frame = _compose_video_frame(
                observation,
                info,
                lane=lane,
                episode_id=episode_id,
                tags=record["tags"],
                step=step,
                row_index=int(row_index[lane]),
                policy_mode=self._policy_mode,
            )
            self._writer(episode_id).append_data(frame)
            self._frame_counts[episode_id] += 1

    def close(self) -> list[dict[str, Any]]:
        if self._closed:
            return self.manifest()
        self._closed = True
        for episode_id, writer in self._writers.items():
            writer.close()
            self._temporary_paths[episode_id].replace(self._final_paths[episode_id])
        return self.manifest()

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "episode_id": episode_id,
                "tags": record["tags"],
                "frames": self._frame_counts[episode_id],
                "fps": self._fps,
                "relative_path": str(
                    self._final_paths.get(episode_id, Path("missing.mp4")).relative_to(self.output_dir.parent)
                ),
            }
            for episode_id, record in self.selected.items()
        ]

    def abort(self) -> None:
        for writer in self._writers.values():
            writer.close()
        for path in self._temporary_paths.values():
            path.unlink(missing_ok=True)
        self._closed = True


def _replay_matches(primary: dict[str, Any], replay: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    differences = {
        name: abs(float(primary[name]) - float(replay[name]))
        for name in (
            "return",
            "max_contacted_carry_lift_height_m",
            "max_physical_lift_height_m",
            "final_current_lift_height_m",
        )
    }
    differences["length"] = abs(int(primary["length"]) - int(replay["length"]))
    matches = (
        primary["length"] == replay["length"]
        and primary["success"] == replay["success"]
        and primary["fail"] == replay["fail"]
        and differences["return"] <= 1.0e-4
        and max(value for name, value in differences.items() if name != "length") <= 1.0e-5
    )
    return matches, differences


def main() -> None:
    args = create_parser().parse_args()
    _validate_args(args)
    import torch

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Residual PPO evaluation requires a CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    apply_residual = args.policy_mode == "residual"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for protected_name in ("summary.json", "episodes.jsonl", "selected_episodes.json"):
        if (args.output_dir / protected_name).exists():
            raise FileExistsError(f"Refusing to overwrite {args.output_dir / protected_name}")

    started = time.perf_counter()
    print(f"loading frozen DP: {args.checkpoint}", flush=True)
    frozen_dp, dp_config, scheduler, stats, dp_sha256 = _load_frozen_dp(args.checkpoint, device)
    _validate_frozen_dp_training_contract(dp_config)
    checkpoint_role = "action_policy" if apply_residual else "evaluation_contract_only"
    print(f"loading residual checkpoint ({checkpoint_role}): {args.residual_checkpoint}", flush=True)
    residual = torch.load(args.residual_checkpoint, map_location="cpu", weights_only=False)
    if residual.get("format") != GROOT_RESIDUAL_PPO_CHECKPOINT_FORMAT:
        raise ValueError("Residual checkpoint format is incompatible")
    _validate_resume_training_contract(residual)
    if residual.get("frozen_dp_sha256") != dp_sha256:
        raise ValueError("Residual checkpoint was trained with a different frozen DP checkpoint")

    policy_config = GrootResidualActorCriticConfig(**residual["policy_config"])
    expected_condition_dim = dp_config.obs_horizon * (2 * dp_config.camera_feature_dim + dp_config.state_feature_dim)
    if policy_config.condition_dim != expected_condition_dim:
        raise ValueError(
            f"Residual condition width {policy_config.condition_dim} does not match frozen DP {expected_condition_dim}"
        )
    actor_critic = None
    if apply_residual:
        actor_critic = GrootResidualActorCritic(policy_config).to(device)
        actor_critic.load_state_dict(residual["actor_critic"])
        actor_critic.eval().requires_grad_(False)
    else:
        print("pure DP mode: residual actor weights will not be loaded or executed", flush=True)

    saved_env_config = dict(residual["env_config"])
    saved_env_config.update(num_envs=args.num_envs, device=args.device)
    env_config = GrootNewtonEnvConfig(**saved_env_config)
    train_args = residual["train_args"]
    evaluation_args = SimpleNamespace(
        device=args.device,
        seed=args.seed,
        num_envs=args.num_envs,
        inference_steps=int(train_args["inference_steps"]),
        bfloat16=bool(train_args["bfloat16"]),
        position_residual_scale_m=float(train_args["position_residual_scale_m"]),
        vertical_residual_scale_m=float(train_args["vertical_residual_scale_m"]),
        rotation_residual_scale_deg=float(train_args["rotation_residual_scale_deg"]),
        hand_residual_scale_normalized=float(train_args["hand_residual_scale_normalized"]),
        max_episode_steps=env_config.max_episode_steps,
    )
    hand_scale_values = residual["hand_residual_scale"]["effective_scale_normalized"]
    hand_residual_scale = torch.tensor(hand_scale_values, dtype=torch.float32, device=device)
    action_min = torch.as_tensor(stats["action_min"], dtype=torch.float32, device=device)
    action_max = torch.as_tensor(stats["action_max"], dtype=torch.float32, device=device)
    state_min = torch.as_tensor(stats["state_min"], dtype=torch.float32, device=device)
    state_max = torch.as_tensor(stats["state_max"], dtype=torch.float32, device=device)
    noise_bank, wave_generator_states, locations = _build_episode_noise_contract(
        episodes=args.episodes,
        num_envs=args.num_envs,
        pred_horizon=dp_config.pred_horizon,
        action_dim=dp_config.action_dim,
        seed=args.seed,
        device=device,
    )

    base_env: GrootNewtonEnv | None = None
    try:
        print(f"constructing {args.num_envs} Newton environments", flush=True)
        base_env = GrootNewtonEnv(env_config)
        if residual.get("finger_root_load") != base_env.finger_root_load_metadata:
            raise ValueError("Residual checkpoint finger-root load calibration does not match the environment")
        if residual.get("hand_target") != base_env.hand_target_metadata:
            raise ValueError("Residual checkpoint hand-target metadata does not match the environment")
        env = GrootDiffusionPolicyEnv(base_env, obs_horizon=dp_config.obs_horizon, action_horizon=1)
        episode_records: list[dict[str, Any]] = []
        print(
            f"starting deterministic evaluation: episodes={args.episodes}, num_envs={args.num_envs}, "
            f"waves={len(wave_generator_states)}",
            flush=True,
        )
        evaluation_started = time.perf_counter()
        metrics, _, _ = _evaluate(
            env,
            frozen_dp,
            scheduler,
            actor_critic,
            action_min,
            action_max,
            state_min,
            state_max,
            hand_residual_scale,
            evaluation_args,
            episodes=args.episodes,
            apply_residual=apply_residual,
            episode_initial_noise=noise_bank,
            wave_generator_states=wave_generator_states,
            episode_records=episode_records,
            step_callback=_ProgressCallback(args.episodes),
        )
        evaluation_seconds = time.perf_counter() - evaluation_started
        episode_records.sort(key=lambda record: record["episode_id"])
        expected_ids = list(range(args.episodes))
        if [record["episode_id"] for record in episode_records] != expected_ids:
            raise RuntimeError("Evaluation episode records are missing, duplicated, or out of order")
        for episode_id, (wave, lane) in enumerate(locations):
            record = episode_records[episode_id]
            if (record["wave"], record["lane"], record["noise_index"]) != (wave, lane, episode_id):
                raise RuntimeError(f"Episode {episode_id} noise/wave/lane mapping is inconsistent")

        selection_source = "current_run_representatives"
        if args.selected_episodes_json is None:
            selected = _select_representative_episodes(episode_records, min(args.record_rollouts, args.episodes))
        else:
            references = json.loads(args.selected_episodes_json.read_text())
            selected = _select_referenced_episodes(
                episode_records,
                references,
                min(args.record_rollouts, args.episodes),
            )
            selection_source = str(args.selected_episodes_json)
        _atomic_write_jsonl(args.output_dir / "episodes.jsonl", episode_records)
        _atomic_write_json(args.output_dir / "selected_episodes.json", selected)
        summary = {
            "status": "evaluation_complete",
            "deterministic_contract": {
                "policy_mode": args.policy_mode,
                "residual_applied": apply_residual,
                "executed_action": "residual_composition" if apply_residual else "cached_dp_row_exact",
                "actor_action": "mean" if apply_residual else "not_applicable",
                "base_action_mode": residual["base_action_mode"],
                "base_action_horizon": residual["base_action_horizon"],
                "episode_initial_noise": "unique_per_episode_and_reused_at_each_replan",
                "scheduler_rng": "independent_seeded_wave_state_after_full_lane_initial_noise",
                "wave_seed_formula": f"seed + {_NOISE_SEED_OFFSET} + wave",
                "first_wave_legacy_compatibility": "matches legacy 32-lane initial-noise and scheduler RNG stream",
                "environment_reset": "fixed task reset from checkpoint environment configuration",
            },
            "request": {
                "episodes": args.episodes,
                "num_envs": args.num_envs,
                "seed": args.seed,
                "device": args.device,
                "record_rollouts": len(selected),
                "video_fps": args.video_fps,
                "selected_episode_source": selection_source,
            },
            "checkpoint": {
                "residual_path": str(args.residual_checkpoint),
                "residual_sha256": _file_sha256(args.residual_checkpoint),
                "residual_checkpoint_role": checkpoint_role,
                "frozen_dp_path": str(args.checkpoint),
                "frozen_dp_sha256": dp_sha256,
                "training_contract_version": residual["training_contract_version"],
                "reward_contract_version": residual["reward_contract_version"],
                "update": int(residual["update"]),
                "global_step": int(residual["global_step"]),
            },
            "runtime": {
                "evaluation_seconds": evaluation_seconds,
                "episodes_per_second": args.episodes / evaluation_seconds,
            },
            "metrics": metrics,
            "record_statistics": _record_statistics(episode_records),
            "success_rate_wilson_95": _wilson_interval(int(metrics["success_count"]), args.episodes),
            "visualization": {"status": "pending" if selected else "not_requested", "rollouts": []},
            "env_config": asdict(env_config),
            "policy_config": asdict(policy_config),
        }
        _atomic_write_json(args.output_dir / "summary.json", summary)
        print(
            f"headless evaluation complete in {evaluation_seconds:.1f}s; "
            f"success={metrics['success_rate']:.3%}, mean_return={metrics['mean_return']:.3f}, "
            f"max_physical_lift={metrics['max_episode_physical_max_lift_height_m']:.4f}m",
            flush=True,
        )

        if selected:
            print(f"replaying all episodes and recording {len(selected)} representative rollouts", flush=True)
            video_callback = _VideoCallback(
                args.output_dir / "rollouts",
                selected,
                args.video_fps,
                args.policy_mode,
            )
            replay_records: list[dict[str, Any]] = []
            replay_started = time.perf_counter()
            try:
                replay_metrics, _, _ = _evaluate(
                    env,
                    frozen_dp,
                    scheduler,
                    actor_critic,
                    action_min,
                    action_max,
                    state_min,
                    state_max,
                    hand_residual_scale,
                    evaluation_args,
                    episodes=args.episodes,
                    apply_residual=apply_residual,
                    episode_initial_noise=noise_bank,
                    wave_generator_states=wave_generator_states,
                    episode_records=replay_records,
                    step_callback=video_callback,
                )
                rollout_manifest = video_callback.close()
            except BaseException:
                video_callback.abort()
                raise
            replay_seconds = time.perf_counter() - replay_started
            replay_records.sort(key=lambda record: record["episode_id"])
            _atomic_write_jsonl(args.output_dir / "replay_episodes.jsonl", replay_records)
            replay_by_id = {record["episode_id"]: record for record in replay_records}
            all_match = True
            for rollout in rollout_manifest:
                episode_id = rollout["episode_id"]
                matches, differences = _replay_matches(episode_records[episode_id], replay_by_id[episode_id])
                rollout["deterministic_replay_match"] = matches
                rollout["replay_absolute_differences"] = differences
                rollout["headless_episode"] = episode_records[episode_id]
                rollout["visual_replay_episode"] = replay_by_id[episode_id]
                all_match &= matches
            _atomic_write_json(
                args.output_dir / "rollouts" / "manifest.json",
                {
                    "policy_mode": args.policy_mode,
                    "replay_seconds": replay_seconds,
                    "all_selected_rollouts_match": all_match,
                    "aggregate_metrics_match": replay_metrics == metrics,
                    "rollouts": rollout_manifest,
                },
            )
            summary["visualization"] = {
                "status": "complete" if all_match else "complete_with_gpu_replay_drift",
                "replay_seconds": replay_seconds,
                "all_selected_rollouts_match": all_match,
                "aggregate_metrics_match": replay_metrics == metrics,
                "rollouts": rollout_manifest,
            }
            summary["runtime"]["total_seconds"] = time.perf_counter() - started
            _atomic_write_json(args.output_dir / "summary.json", summary)
            if not all_match:
                print(
                    "warning: visual capture changed one or more GPU contact rollouts; "
                    "headless and visual replay results are both preserved in the manifest",
                    flush=True,
                )
            print(f"visualization complete in {replay_seconds:.1f}s", flush=True)
        else:
            summary["runtime"]["total_seconds"] = time.perf_counter() - started
            _atomic_write_json(args.output_dir / "summary.json", summary)
    finally:
        if base_env is not None:
            base_env.close()


if __name__ == "__main__":
    main()
