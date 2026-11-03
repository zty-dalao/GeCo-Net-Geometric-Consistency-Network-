"""Cone-beam geometry helpers for quaternion-conditioned view synthesis.

The project stores one view as a 12-vector

    [source(3), detector(3), u_vector(3), v_vector(3)]

where ``u_vector`` / ``v_vector`` are the detector basis vectors *with the pixel
spacing already folded in* (see ``models/render.py:angle2vec``).  This module
derives, from that vector only:

* the camera frame ``R = [u_hat, v_hat, n_hat]`` (camera -> world),
* the relative rotation / translation between two views,
* the plane-induced homography that warps one view onto another,
* the quaternion pair to feed the quaternion network.

Nothing here replaces the 12-vector: a quaternion cannot encode source
position, SID/SAD, pixel spacing or half-fan offsets, so the physical geometry
stays in the 12-vector and the quaternion is only the rotation condition.
"""

import math

import torch
import torch.nn.functional as F

_EPS = 1e-8


def _normalize(vector, eps=_EPS):
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


def pose_to_frame(vecs):
    """Return ``(source, detector, R)`` with ``R = [u_hat, v_hat, n_hat]``."""
    u = _normalize(vecs[..., 6:9])
    v = _normalize(vecs[..., 9:12])
    n = _normalize(torch.cross(u, v, dim=-1))
    frame = torch.stack((u, v, n), dim=-1)  # [..., 3, 3], columns
    return vecs[..., :3], vecs[..., 3:6], frame


def pixel_spacing(vecs):
    """Return detector pixel spacing ``(du, dv)`` in millimetres."""
    return vecs[..., 6:9].norm(dim=-1), vecs[..., 9:12].norm(dim=-1)


def intrinsics(du, dv, height, width, focal):
    """Pixel matrix for a flat detector.

    A ray through the camera point ``X_cam = (X, Y, Z)`` hits the detector plane
    ``Z = SID`` at ``(X, Y) * SID / Z``, so the focal length in pixels is
    ``SID / du``.  ``K^-1 @ (col, row, 1)`` therefore returns the normalised
    ray ``(X/Z, Y/Z, 1)`` that the plane-induced homography needs.
    """
    zero = torch.zeros_like(du)
    one = torch.ones_like(du)
    centre_u = torch.full_like(du, (width - 1) * 0.5)
    centre_v = torch.full_like(dv, (height - 1) * 0.5)
    # [..., 3, 3]
    row0 = torch.stack((focal / du, zero, centre_u), dim=-1)
    row1 = torch.stack((zero, focal / dv, centre_v), dim=-1)
    row2 = torch.stack((zero, zero, one), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def relative_motion(vecs_s, vecs_t, isocenter=None):
    """Return ``(R_rel, t_rel, R_s, C_s)`` such that ``X_t = R_rel X_s + t_rel``."""
    source_s, _, frame_s = pose_to_frame(vecs_s)
    source_t, _, frame_t = pose_to_frame(vecs_t)
    frame_t_inv = frame_t.transpose(-1, -2)
    rot_rel = frame_t_inv @ frame_s
    trans_rel = (frame_t_inv @ (source_s - source_t).unsqueeze(-1)).squeeze(-1)
    return rot_rel, trans_rel, frame_s, source_s


def isocenter_depth(vecs_s, isocenter=None):
    """Signed distance from the source to the rotation-axis plane, along n."""
    source_s, _, frame_s = pose_to_frame(vecs_s)
    if isocenter is None:
        isocenter = torch.zeros_like(source_s)
    normal = frame_s[..., :, 2]
    return ((isocenter - source_s) * normal).sum(dim=-1)


def homography_source_to_target(vecs_s, vecs_t, height, width, isocenter=None):
    """Plane-induced homography mapping *source* pixels to *target* pixels.

    The plane is the one through the isocenter perpendicular to the optical
    axis of the source view, i.e. ``z_cam = depth`` in the source camera frame.
    For a pure circular C-arm orbit the detector size is tiny compared with
    SAD, so this homography already explains the dominant part of the warp and
    the network only has to model the residual (cone-beam parallax, occlusion
    in/out, heel effect, scatter).
    """
    rot_rel, trans_rel, frame_s, _ = relative_motion(vecs_s, vecs_t, isocenter)
    depth = isocenter_depth(vecs_s, isocenter)
    e_z = torch.zeros_like(trans_rel)
    e_z[..., 2] = 1.0
    plane = rot_rel + trans_rel.unsqueeze(-1) * e_z.unsqueeze(-2) / depth.view(-1, 1, 1)

    du_s, dv_s = pixel_spacing(vecs_s)
    du_t, dv_t = pixel_spacing(vecs_t)
    source_s, detector_s, _ = pose_to_frame(vecs_s)
    source_t, detector_t, _ = pose_to_frame(vecs_t)
    sid_s = (detector_s - source_s).norm(dim=-1)
    sid_t = (detector_t - source_t).norm(dim=-1)
    k_s = intrinsics(du_s, dv_s, height, width, sid_s)
    k_t = intrinsics(du_t, dv_t, height, width, sid_t)
    return k_t @ plane @ torch.linalg.inv(k_s)


def _pixel_grid(height, width, device, dtype):
    rows = torch.arange(height, device=device, dtype=dtype)
    cols = torch.arange(width, device=device, dtype=dtype)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    ones = torch.ones_like(grid_r)
    return torch.stack((grid_c, grid_r, ones), dim=-1)  # [H, W, 3]


def warp_with_homography(image, homography, out_height=None, out_width=None):
    """Warp ``image`` [B, C, H, W] by a single 3x3 homography."""
    if homography.ndim == 2:
        homography = homography.unsqueeze(0)
    height, width = out_height or image.shape[-2], out_width or image.shape[-1]
    grid = _pixel_grid(height, width, image.device, image.dtype)
    grid = grid.unsqueeze(0).expand(homography.shape[0], -1, -1, -1)

    inverse = torch.linalg.inv(homography)
    src = torch.einsum("bij,bhwj->bhwi", inverse, grid)
    src = src[..., :2] / src[..., 2:3].clamp_min(_EPS)

    col = src[..., 0]
    row = src[..., 1]
    if width > 1:
        col = 2.0 * col / (width - 1) - 1.0
    if height > 1:
        row = 2.0 * row / (height - 1) - 1.0
    sample_grid = torch.stack((col, row), dim=-1)
    return F.grid_sample(
        image, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def warp_view(image, vec_s, vec_t, isocenter=None):
    """Warp a single [1, H, W] projection from ``vec_s`` geometry to ``vec_t``."""
    homography = homography_source_to_target(
        vec_s.view(-1, 12), vec_t.view(-1, 12), image.shape[-2], image.shape[-1], isocenter
    )
    batched = image.view(1, 1, *image.shape[-2:])
    return warp_with_homography(batched, homography).view(image.shape)


def signed_angle_degrees(quaternion):
    """Rotation angle in degrees of a unit quaternion, in ``[0, 180]``."""
    from .qconv import quaternion_normalize

    q = quaternion_normalize(quaternion)
    return 2.0 * torch.acos(q[..., 0].abs().clamp(max=1.0)) * (180.0 / math.pi)


def residual_gate(quaternion):
    """``g = sin(Delta/2)``: exactly 0 at identity, 1 at 180 degrees.

    Multiplying the learned residual by this gate makes the synthesiser an
    exact identity when the requested angle equals the input angle, so the 20
    real measurements are never rewritten by the network.
    """
    from .qconv import quaternion_normalize

    q = quaternion_normalize(quaternion)
    half = torch.acos(q[..., 0].abs().clamp(max=1.0))
    return torch.sin(half)
