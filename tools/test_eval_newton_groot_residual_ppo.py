#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic residual PPO evaluation helpers."""

from __future__ import annotations

import unittest

from tools.eval_newton_groot_residual_ppo import (
    _episode_locations,
    _select_referenced_episodes,
    _select_representative_episodes,
)


class TestResidualPPOEvaluation(unittest.TestCase):
    def test_episode_locations_match_rotated_partial_evaluation_waves(self) -> None:
        self.assertEqual(
            _episode_locations(7, 3),
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 2)],
        )
        self.assertEqual(_episode_locations(2, 4), [(0, 0), (0, 1)])
        with self.assertRaisesRegex(ValueError, "positive"):
            _episode_locations(0, 4)

    def test_representative_selection_is_stable_and_merges_tags(self) -> None:
        records = [
            {
                "episode_id": 0,
                "return": 100.0,
                "success": True,
                "max_contacted_carry_lift_height_m": 0.20,
                "max_physical_lift_height_m": 0.20,
                "event_contact_ever": True,
                "event_grasp_ever": True,
                "event_early_release_ever": False,
            },
            {
                "episode_id": 1,
                "return": 10.0,
                "success": False,
                "max_contacted_carry_lift_height_m": 0.08,
                "max_physical_lift_height_m": 0.09,
                "event_contact_ever": True,
                "event_grasp_ever": True,
                "event_early_release_ever": True,
            },
            {
                "episode_id": 2,
                "return": 20.0,
                "success": False,
                "max_contacted_carry_lift_height_m": 0.0,
                "max_physical_lift_height_m": 0.01,
                "event_contact_ever": True,
                "event_grasp_ever": False,
                "event_early_release_ever": False,
            },
            {
                "episode_id": 3,
                "return": -1.0,
                "success": False,
                "max_contacted_carry_lift_height_m": 0.0,
                "max_physical_lift_height_m": 0.0,
                "event_contact_ever": False,
                "event_grasp_ever": False,
                "event_early_release_ever": False,
            },
        ]

        selected = _select_representative_episodes(records, 3)
        reversed_selected = _select_representative_episodes(list(reversed(records)), 3)
        self.assertEqual(selected, reversed_selected)
        self.assertEqual([record["episode_id"] for record in selected], [0, 1, 2])
        self.assertIn("best_success", selected[0]["tags"])
        self.assertIn("top_return", selected[0]["tags"])
        self.assertIn("early_release", selected[1]["tags"])
        self.assertIn("contact_without_grasp", selected[2]["tags"])
        self.assertEqual(_select_representative_episodes(records, 0), [])

    def test_referenced_selection_keeps_current_metrics_and_reference_ids(self) -> None:
        records = [
            {"episode_id": 0, "return": 1.0},
            {"episode_id": 1, "return": 2.0},
            {"episode_id": 2, "return": 3.0},
        ]
        references = [
            {"episode_id": 2, "return": 99.0, "tags": ["top_return"]},
            {"episode_id": 0, "tags": ["lowest_return"]},
        ]

        selected = _select_referenced_episodes(records, references, 2)

        self.assertEqual([record["episode_id"] for record in selected], [2, 0])
        self.assertEqual([record["return"] for record in selected], [3.0, 1.0])
        self.assertEqual(selected[0]["tags"], ["matched_rl", "top_return"])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _select_referenced_episodes(records, [references[0], references[0]], 2)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            _select_referenced_episodes(records, [{"episode_id": 4}], 1)


if __name__ == "__main__":
    unittest.main()
