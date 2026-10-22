"""Differentiable quaternion utilities for the CBCT view geometry.

The 12-vector geometry used by the projection code is

    [source, detector, u_vector, v_vector]

where ``u_vector`` and ``v_vector`` span the detector plane.  This module keeps
that parameterization as the projection geometry and only derives rotations
from it, so the quaternion representation is an additional view relation rather
than a replacement for the physical camera parameters.
"""

import math

import torch

_EPS = 1e-8


def _normalize(vector, eps=_EPS):
    return vector / (torch.linalg.norm(vector, dim=-1, keepdim=True) + eps)


def _sqrt_positive(part):
    return torch.sqrt(torch.clamp(part, min=_EPS))


def vec12_to_rotation(poses):
    """Return (..., 3, 3) detector frames from [source, detector, u, v]."""
    u = _normalize(poses[..., 6:9])
    v = _normalize(poses[..., 9:12])
    normal = _normalize(torch.cross(u, v, dim=-1))
    return torch.stack((u, v, normal), dim=-1)


def rotation_matrix_to_quaternion(matrix):
    """Convert (..., 3, 3) rotation matrices to canonical ``(w, x, y, z)``.

    Uses the branchless Shepperd construction so that rotations close to 180
    degrees remain numerically stable; the trace-only formula divides by
    ``4 * w`` and degrades exactly in that case.
    """
    m = matrix
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    q_abs = _sqrt_positive(torch.stack((
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ), dim=-1))

    candidates = torch.stack((
        torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
        torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
        torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
        torch.stack((m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2), dim=-1),
    ), dim=-2)

    denominator = 2.0 * q_abs.max(dim=-1, keepdim=True).values
    candidates = candidates / denominator.unsqueeze(-1)

    largest = q_abs.argmax(dim=-1)
    index = largest[..., None, None].expand(*largest.shape, 1, 4)
    quaternion = torch.gather(candidates, dim=-2, index=index).squeeze(-2)
    # q and -q encode the same rotation.  Picking w >= 0 makes the encoding
    # unique, which matters because the quaternion is fed to an MLP.
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    return _normalize(quaternion)


def quaternion_to_matrix(quaternion):
    """Convert (..., 4) ``(w, x, y, z)`` quaternions to (..., 3, 3) matrices."""
    q = _normalize(quaternion)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(q.shape[:-1] + (3, 3))


def quaternion_inverse(quaternion):
    q = _normalize(quaternion)
    return q * torch.tensor(
        (1.0, -1.0, -1.0, -1.0), device=q.device, dtype=q.dtype,
    )


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), dim=-1)


def pose_vec_to_quaternion(poses):
    """Return camera-frame quaternions from [source, detector, u, v] vectors."""
    return rotation_matrix_to_quaternion(vec12_to_rotation(poses))


def relative_quaternion(poses, reference_index=0):
    """Return ``q_ref^-1 * q_v``, the rotation from the reference view to v."""
    quaternion = pose_vec_to_quaternion(poses)
    reference = quaternion[reference_index:reference_index + 1]
    return _normalize(quaternion_multiply(quaternion_inverse(reference), quaternion))


def axis_angle_to_quaternion(rotvec):
    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(
        angle > _EPS,
        torch.sin(half) / angle.clamp_min(_EPS),
        0.5 - angle * angle / 48.0,
    )
    return _normalize(torch.cat((torch.cos(half), rotvec * scale), dim=-1))


def quaternion_to_rotation_vector(quaternion):
    """Return the axis-angle vector (rotation logarithm) of a quaternion."""
    q = _normalize(quaternion)
    q = torch.where(q[..., :1] < 0, -q, q)
    xyz = q[..., 1:]
    norm = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, q[..., :1].clamp_min(0.0))
    scale = torch.where(
        norm > 1e-7,
        angle / norm.clamp_min(1e-7),
        torch.full_like(norm, 2.0),
    )
    return xyz * scale


def apply_pose_delta(poses, rotation_vector, translation):
    """Apply ``(delta q, delta t)`` to [source, detector, u, v] geometry.

    The source is translated, while the source-to-detector direction and the
    detector basis vectors are rotated.  This preserves the cone-beam layout
    and is exactly the identity for zero deltas.
    """
    rotation = quaternion_to_matrix(axis_angle_to_quaternion(rotation_vector))
    source, detector = poses[..., :3], poses[..., 3:6]
    u_vector, v_vector = poses[..., 6:9], poses[..., 9:12]

    new_source = source + translation
    relative_detector = (detector - source).unsqueeze(-1)
    new_detector = new_source + torch.matmul(rotation, relative_detector).squeeze(-1)
    new_u = torch.matmul(rotation, u_vector.unsqueeze(-1)).squeeze(-1)
    new_v = torch.matmul(rotation, v_vector.unsqueeze(-1)).squeeze(-1)
    return torch.cat((new_source, new_detector, new_u, new_v), dim=-1)


def rotation_angle_degrees(quaternion):
    """Return rotation angles in degrees for (..., 4) quaternions."""
    rotvec = quaternion_to_rotation_vector(quaternion)
    return torch.linalg.norm(rotvec, dim=-1) * (180.0 / math.pi)
