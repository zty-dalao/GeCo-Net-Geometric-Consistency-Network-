"""Per-view quaternion geometry calibration."""

import math

import torch
import torch.nn as nn

from .pose import apply_pose_delta


class PoseRefiner(nn.Module):
    """Learn small, per-view geometric corrections.

    A parameter table fits the task because a scan has a fixed, ordered set of
    views.  Predicting a bounded rotation vector instead of an arbitrary
    quaternion keeps the correction near the nominal geometry, which is the
    regime where cone-beam miscalibration lives.  Zero initialization makes
    ``calibration`` start exactly at the original geometry.
    """

    def __init__(self, max_views=256, max_angle_deg=1.0, max_translation=1.0,
                 learn_rotation=True, learn_translation=True):
        super().__init__()
        self.max_angle = float(max_angle_deg) * math.pi / 180.0
        self.max_translation = float(max_translation)
        self.learn_rotation = bool(learn_rotation)
        self.learn_translation = bool(learn_translation)
        self.delta_rotvec = nn.Parameter(torch.zeros(int(max_views), 3))
        self.delta_translation = nn.Parameter(torch.zeros(int(max_views), 3))

    def forward(self, poses):
        views = poses.shape[0]
        if views > self.delta_rotvec.shape[0]:
            raise ValueError(
                f"PoseRefiner supports {self.delta_rotvec.shape[0]} views, got {views}"
            )
        if self.learn_rotation:
            rotvec = self.max_angle * torch.tanh(self.delta_rotvec[:views])
        else:
            rotvec = poses.new_zeros((views, 3))
        if self.learn_translation:
            translation = self.max_translation * torch.tanh(
                self.delta_translation[:views]
            )
        else:
            translation = poses.new_zeros((views, 3))

        corrected = apply_pose_delta(poses, rotvec, translation)
        # |log(delta q)|^2 is exactly |rotvec|^2 because delta q is built with
        # axis-angle_to_quaternion; translations use the same quadratic penalty.
        regularizer = rotvec.square().mean() + translation.square().mean()
        return corrected, regularizer, rotvec, translation
