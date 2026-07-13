#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the node0 GR00T conventions used by Newton."""

# ruff: noqa: SLF001

from __future__ import annotations

import unittest

import numpy as np

from tools import run_newton_groot_rtc_control as groot_runtime


class TestNode0GrootAlignment(unittest.TestCase):
    def test_rot6d_uses_first_two_rows(self) -> None:
        rotation = np.asarray(
            (
                (0.0, -1.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )

        rot6d = groot_runtime._rotmat_to_rot6d(rotation)

        np.testing.assert_array_equal(rot6d, (0.0, -1.0, 0.0, 1.0, 0.0, 0.0))
        np.testing.assert_allclose(groot_runtime._rot6d_to_rotmat(rot6d), rotation, atol=1.0e-7)

    def test_node0_eef_transform_is_scene_a_policy_b(self) -> None:
        controller = groot_runtime.NewtonPolicyController.__new__(groot_runtime.NewtonPolicyController)
        controller.eef_transform_mode = "node0_fixed"
        controller._eef_frame_calibrated = True
        controller._state_to_genesis_transform = groot_runtime._rigid_transform_matrix(
            groot_runtime.NODE0_STATE_TO_GENESIS_TRANSLATION_XYZ,
            groot_runtime.NODE0_STATE_TO_GENESIS_QUATERNION_XYZW,
        )
        controller._eef_offset_transform = groot_runtime._rigid_transform_matrix(
            groot_runtime.NODE0_STATE_TO_GENESIS_EEF_OFFSET_TRANSLATION_XYZ,
            groot_runtime.NODE0_STATE_TO_GENESIS_EEF_OFFSET_QUATERNION_XYZW,
        )
        controller._genesis_to_world_transform = groot_runtime._rigid_transform_matrix(
            (-0.003, 0.003, 0.16),
            (0.0, 0.0, 0.0, 1.0),
        )
        policy_pose = np.eye(4, dtype=np.float64)
        policy_pose[:3, 3] = (0.41, -0.12, 0.73)
        policy_pose[:3, :3] = np.asarray(
            groot_runtime.quat_xyzw_to_matrix((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)))
        )

        world_pose = controller._policy_pose_to_world_pose(policy_pose)
        expected = (
            controller._genesis_to_world_transform
            @ controller._state_to_genesis_transform
            @ policy_pose
            @ controller._eef_offset_transform
        )

        np.testing.assert_allclose(world_pose, expected, atol=1.0e-12)
        np.testing.assert_allclose(
            controller._world_pose_to_policy_pose(world_pose),
            policy_pose,
            atol=1.0e-12,
        )

    def test_sim_image_defaults_match_node0(self) -> None:
        args = groot_runtime.create_parser().parse_args([])

        self.assertEqual((args.d455_render_width, args.d455_render_height), (320, 180))
        self.assertEqual((args.d405_width, args.d405_height), (640, 480))
        self.assertFalse(args.sim_ego_roi)
        self.assertEqual(args.eef_transform_mode, "node0_fixed")


if __name__ == "__main__":
    unittest.main()
