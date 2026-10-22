"""Quaternion-aware view fusion and per-view geometry calibration."""

from .calibration import PoseRefiner
from .fusion import QuaternionAwareAggregator
from .pose import (
    apply_pose_delta,
    axis_angle_to_quaternion,
    pose_vec_to_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_matrix,
    quaternion_to_rotation_vector,
    relative_quaternion,
    rotation_angle_degrees,
    rotation_matrix_to_quaternion,
    vec12_to_rotation,
)

# Diagnostic values written to the train/val/test streams.  Keys that the
# active mode does not produce are reported as zero so every split always
# logs the same set.
METRIC_KEYS = (
    "pose_delta_regularizer",
    "pose_delta_angle_mean",
    "pose_delta_translation_mean",
    "relative_pose_angle_deg_mean",
    "view_weight_entropy_norm",
    "view_weight_max_mean",
)

__all__ = [
    "METRIC_KEYS",
    "PoseRefiner",
    "QuaternionAwareAggregator",
    "apply_pose_delta",
    "axis_angle_to_quaternion",
    "pose_vec_to_quaternion",
    "quaternion_inverse",
    "quaternion_multiply",
    "quaternion_to_matrix",
    "quaternion_to_rotation_vector",
    "relative_quaternion",
    "rotation_angle_degrees",
    "rotation_matrix_to_quaternion",
    "vec12_to_rotation",
]
