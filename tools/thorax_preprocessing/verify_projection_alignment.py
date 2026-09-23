"""诊断 thorax 病例的 DRR 是否与实测投影对齐。

纯新增文件，不修改任何现有模块。几何**不做任何重新推导**：脚本把
``transforms.json["frames"][i]["vec"]`` 原样送进仓库自己的
``models.render.get_rays`` / ``models.render.composite``，因此它评估的就是训练时
真正使用的几何。

每个变体只改动**一个**假设，其余完全相同：

======================  ==========================================================
变体                     改动了什么
======================  ==========================================================
``declared``            当前行为：``volume_origin`` 取自 transforms.json，体数据按
                        ``sitk.GetArrayFromImage`` 的 (Z, Y, X) 顺序直接喂入
``true-origin``         ``volume_origin`` 改为 NIfTI 头里的真实值
``v-flip``              ``declared`` + 渲染结果行方向翻转
``no-half-fan``         ``declared`` + 去掉每帧探测器偏移（即 ``--angle_sampling
                        random`` 会产生的几何）
``z-only``              仅 z 用真实 origin，x/y 保持声明值
``tight-phy``           真实 origin，且 ``volume_phy = (size - 1) * spacing``
``axis-xyz``            体数据转置成 (X, Y, Z)，origin 用声明值
``axis-xyz+true``       体数据转置成 (X, Y, Z)，origin 用真实值
``u-flip``              ``declared`` + 渲染结果列方向翻转
======================  ==========================================================

``axis-xyz`` 这一组检验的是一个**独立于 origin 的假设**：``models/model.py`` 的
``forward`` 里 ``self.decoder(latent)[0,0,:,:,:].transpose(0,2)`` 让 3D 体数据以
(Z, Y, X) 存放（与 SimpleITK 一致），但 ``models/render.py`` 的 ``volume_sampling``
用 ``grid_sample``，其最后一维 (x, y, z) 会去索引张量的 (dim0, dim1, dim2)，也就是
把**第 0 维当作世界 x**。对 256^3 的 dental 体数据这不可见；对 248x248x120 的
thorax 体数据则会把世界 x 的坐标拿去索引数组的 Z 维。

用法（在项目根目录执行）：

    /autdl-tmp/conda_env/GeoAware/bin/python \
        tools/thorax_preprocessing/verify_projection_alignment.py \
        --cases 2026-06-04_065713 2026-06-04_081709 \
        --view-indices 0 90 180 270

判读：哪个变体的 ``r_body`` 明显最高，就说明那套几何假设是对的。若某个变体的
``best_shift`` 远离 (0, 0)，说明该变体仍有残余错位，``r_body`` 是靠搜索补偿来的，
不能作为"几何正确"的证据。

先跑 ``--mode correspondence``
------------------------------
在比较变体之前**必须先确认 ``frames[i]`` 与 ``proj[i]`` 描述的是同一个视角**。
``--mode correspondence`` 对若干帧渲染 DRR，然后在**全部**实测视角里找最佳匹配：

    ... verify_projection_alignment.py --mode correspondence

如果 ``frames[i]`` 的最佳匹配不是 ``i``，那么 origin / half-fan 这些毫米级误差都是
次要问题——连视角都对不上时，变体之间的差异没有意义（实测中 frame 0 的最佳匹配是
视角 90，r=0.856，而自身只有 0.320）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.render import composite, get_rays  # noqa: E402


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def load_case(data_root: Path, case: str) -> dict:
    case_dir = data_root / case
    if not case_dir.is_dir():
        raise FileNotFoundError(f"case directory not found: {case_dir}")

    transforms_path = case_dir / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"missing transforms.json: {transforms_path}")
    with transforms_path.open(encoding="utf-8") as handle:
        paras = json.load(handle)

    volume_path = case_dir / "gt_volume.nii.gz"
    image = sitk.ReadImage(str(volume_path))
    # 与 data/Dataset.py 完全一致：不做任何 permute，直接取 (Z, Y, X) 数组
    volume_zyx = sitk.GetArrayFromImage(image)

    proj_path = case_dir / "proj.nii.gz"
    projections = sitk.GetArrayFromImage(sitk.ReadImage(str(proj_path)))

    return {
        "case": case,
        "paras": paras,
        "volume_zyx": np.clip(volume_zyx, 0, volume_zyx.max()).astype(np.float32),
        "projections": np.clip(projections, 0, projections.max()).astype(np.float32),
        "nifti_size": tuple(int(v) for v in image.GetSize()),
        "nifti_spacing": tuple(float(v) for v in image.GetSpacing()),
        "nifti_origin": tuple(float(v) for v in image.GetOrigin()),
    }


# --------------------------------------------------------------------------- #
# 变体构造
# --------------------------------------------------------------------------- #
def make_volume(volume_zyx: np.ndarray, axis: str) -> np.ndarray:
    """``zyx`` 保持原样（= data/Dataset.py 的行为）；``xyz`` 转成 (X, Y, Z)。"""
    if axis == "zyx":
        return volume_zyx
    if axis == "xyz":
        return np.transpose(volume_zyx, (2, 1, 0))
    raise ValueError(f"unknown axis order: {axis}")


def strip_detector_offset(vec: np.ndarray, sid: float, sad: float) -> np.ndarray:
    """把 vec 的探测器中心退回光轴，复现 ``--angle_sampling random`` 的几何。

    ``models/render.angle2vec`` 把探测器中心写死为 ``isocenter - (sid - sad) *
    (cos a, sin a, 0)``，没有偏移参数；这里做同样的替换。
    """
    stripped = np.array(vec, dtype=np.float64, copy=True)
    cam = stripped[:3]
    angle = np.arctan2(cam[1], cam[0])
    stripped[3:6] = (
        -(sid - sad) * np.cos(angle),
        -(sid - sad) * np.sin(angle),
        0.0,
    )
    return stripped


def build_variants(case: dict) -> dict[str, dict]:
    paras = case["paras"]
    resolution = np.asarray(paras["volume_resolution"], dtype=float)
    spacing = np.asarray(paras["volume_spacing"], dtype=float)
    declared_origin = np.asarray(paras["volume_origin"], dtype=float)
    declared_phy = np.asarray(paras["volume_phy"], dtype=float)
    true_origin = np.asarray(case["nifti_origin"], dtype=float)

    # NIfTI 头用的是 (x, y, z) 顺序的 size/spacing，与 volume_resolution/phy 同序
    tight_phy = (resolution - 1.0) * spacing

    z_only_origin = declared_origin.copy()
    z_only_origin[2] = true_origin[2]

    base = dict(use_offset=True, flip_u=False, flip_v=False)

    variants: dict[str, dict] = {}
    for axis in ("zyx", "xyz"):
        suffix = "" if axis == "zyx" else f" [{axis}]"
        variants[f"declared{suffix}"] = {
            **base, "axis": axis, "origin": declared_origin, "phy": declared_phy,
        }
        variants[f"true-origin{suffix}"] = {
            **base, "axis": axis, "origin": true_origin, "phy": declared_phy,
        }
        variants[f"z-only{suffix}"] = {
            **base, "axis": axis, "origin": z_only_origin, "phy": declared_phy,
        }
        variants[f"tight-phy{suffix}"] = {
            **base, "axis": axis, "origin": true_origin, "phy": tight_phy,
        }
    # 只对"当前行为"这一支做单因子扰动，避免组合爆炸
    variants["v-flip"] = {**base, "axis": "zyx", "origin": declared_origin,
                          "phy": declared_phy, "flip_v": True}
    variants["u-flip"] = {**base, "axis": "zyx", "origin": declared_origin,
                          "phy": declared_phy, "flip_u": True}
    variants["no-half-fan"] = {**base, "axis": "zyx", "origin": declared_origin,
                               "phy": declared_phy, "use_offset": False}
    return variants


# --------------------------------------------------------------------------- #
# 渲染与打分
# --------------------------------------------------------------------------- #
def render_drr(volume: torch.Tensor, vec: torch.Tensor, origin, phy, step_mm,
               height: int, width: int, device: torch.device) -> torch.Tensor:
    rays = get_rays(vec[None, :], height, width).reshape(-1, 6)
    projection = composite(
        rays,
        volume,
        torch.as_tensor(origin, dtype=torch.float32, device=device),
        torch.as_tensor(phy, dtype=torch.float32, device=device),
        step_mm,
    )
    return projection.reshape(height, width)


def pearson(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    a = pred[mask]
    b = target[mask]
    if a.numel() < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(a.norm() * b.norm())
    if denom <= 0:
        return float("nan")
    return float((a * b).sum() / denom)


def shifted(pred: torch.Tensor, dv: int, du: int) -> torch.Tensor:
    """把 pred 平移 (dv, du)，越界补零。"""
    out = torch.zeros_like(pred)
    h, w = pred.shape
    vs = slice(max(0, dv), h + min(0, dv))
    us = slice(max(0, du), w + min(0, du))
    vs_src = slice(max(0, -dv), h + min(0, -dv))
    us_src = slice(max(0, -du), w + min(0, -du))
    out[vs, us] = pred[vs_src, us_src]
    return out


def best_shift(drr: torch.Tensor, measured: torch.Tensor, mask: torch.Tensor,
               search: int) -> tuple[int, int, float]:
    """粗搜 + 细化的整数平移，返回 (dv, du, r)。"""
    def score(dv: int, du: int) -> float:
        return pearson(shifted(drr, dv, du), measured, mask)

    coarse = 4
    best = (0, 0, score(0, 0))
    for dv in range(-search, search + 1, coarse):
        for du in range(-search, search + 1, coarse):
            value = score(dv, du)
            if value == value and value > best[2]:
                best = (dv, du, value)
    dv0, du0 = best[0], best[1]
    for dv in range(dv0 - coarse, dv0 + coarse + 1):
        for du in range(du0 - coarse, du0 + coarse + 1):
            value = score(dv, du)
            if value == value and value > best[2]:
                best = (dv, du, value)
    return best


def evaluate_variant(case: dict, spec: dict, view_index: int, step_mm: float,
                     device: torch.device, search: int,
                     angle_triple: tuple[float, float, float] = (1.0, 0.0, 1.0)) -> dict:
    paras = case["paras"]
    resolution = tuple(int(v) for v in paras["volume_resolution"])
    height, width = (int(v) for v in paras["proj_resolution"])

    volume = make_volume(case["volume_zyx"], spec["axis"])
    volume_tensor = torch.as_tensor(volume, dtype=torch.float32, device=device)

    frame = paras["frames"][view_index]
    vec = apply_angle_convention(np.asarray(frame["vec"], dtype=np.float64), paras,
                                 *angle_triple)
    if not spec["use_offset"]:
        vec = strip_detector_offset(vec, paras["sid"], paras["sad"])
    vec_tensor = torch.as_tensor(vec, dtype=torch.float32, device=device)

    drr = render_drr(volume_tensor, vec_tensor, spec["origin"], spec["phy"],
                     step_mm, height, width, device)
    if spec.get("flip_v"):
        drr = torch.flip(drr, dims=(0,))
    if spec.get("flip_u"):
        drr = torch.flip(drr, dims=(1,))

    measured = torch.as_tensor(case["projections"][view_index],
                               dtype=torch.float32, device=device)

    body = measured > 0.15 * float(measured.max())
    hit = drr > 0
    mask = body & hit
    if int(mask.sum()) < 100:
        mask = body

    r_body = pearson(drr, measured, mask)
    r_all = pearson(drr, measured, torch.ones_like(measured, dtype=torch.bool))
    dv, du, r_shift = best_shift(drr, measured, mask, search)

    # 最优尺度下的相对误差，给相关系数一个绝对量纲的补充
    a = drr[mask]
    b = measured[mask]
    scale = float((a * b).sum() / (a * a).sum().clamp_min(1e-12))
    nrmse = float((scale * drr - measured)[mask].pow(2).mean().sqrt()
                  / b.pow(2).mean().sqrt().clamp_min(1e-12))

    return {
        "axis": spec["axis"],
        "origin": [round(float(v), 4) for v in spec["origin"]],
        "phy": [round(float(v), 4) for v in spec["phy"]],
        "use_offset": spec["use_offset"],
        "flip_u": spec["flip_u"],
        "flip_v": spec["flip_v"],
        "volume_input_shape": list(volume.shape),
        "volume_expected_shape": list(resolution),
        "r_body": round(r_body, 5),
        "r_all": round(r_all, 5),
        "best_shift_dv": int(dv),
        "best_shift_du": int(du),
        "r_body_at_best_shift": round(r_shift, 5),
        "nrmse_at_best_scale": round(nrmse, 5),
        "body_pixels": int(body.sum()),
    }


def run_correspondence(case: dict, frame_indices: list[int], step_mm: float,
                       device: torch.device,
                       angle_triple: tuple[float, float, float] = (1.0, 0.0, 1.0)) -> dict:
    """逐帧渲染 DRR，在全部实测视角里找相关系数最高的那个。"""
    paras = case["paras"]
    height, width = (int(v) for v in paras["proj_resolution"])
    volume = torch.as_tensor(make_volume(case["volume_zyx"], "zyx"),
                             dtype=torch.float32, device=device)
    measured = torch.as_tensor(case["projections"], dtype=torch.float32, device=device)
    centered = measured - measured.mean(dim=(1, 2), keepdim=True)
    norms = centered.flatten(1).norm(dim=1)

    origin = torch.as_tensor(paras["volume_origin"], dtype=torch.float32, device=device)
    phy = torch.as_tensor(paras["volume_phy"], dtype=torch.float32, device=device)

    print(f"  {'frame':>5s} {'nominal':>8s} {'angle':>8s} | {'最佳视角':>8s} "
          f"{'r':>6s} | {'自身r':>6s}  {'top-3':>18s}")
    results: dict[str, dict] = {}
    for frame_index in frame_indices:
        if frame_index >= len(paras["frames"]):
            continue
        frame = paras["frames"][frame_index]
        vec = torch.as_tensor(
            apply_angle_convention(np.asarray(frame["vec"], dtype=np.float64),
                                   paras, *angle_triple),
            dtype=torch.float32, device=device)
        rays = get_rays(vec[None, :], height, width).reshape(-1, 6)
        drr = composite(rays, volume, origin, phy, step_mm).reshape(height, width).float()
        flat = drr.flatten() - drr.mean()
        r = (flat[None] * centered.flatten(1)).sum(dim=1) / (flat.norm() * norms + 1e-9)
        top = torch.topk(r, 3)
        best_index = int(top.indices[0])
        top3 = ", ".join(f"{i}:{v:.3f}" for i, v in
                          zip(top.indices.tolist(), top.values.tolist()))
        print(f"  {frame_index:5d} {frame['nominal_angle_degrees']:8.1f} "
              f"{frame['angle_degrees']:8.2f} | {best_index:8d} "
              f"{float(r[best_index]):6.3f} | {float(r[frame_index]):6.3f}  {top3:>18s}")
        results[str(frame_index)] = {
            "nominal_angle_degrees": frame["nominal_angle_degrees"],
            "angle_degrees": frame["angle_degrees"],
            "best_view": best_index,
            "r_at_best_view": round(float(r[best_index]), 5),
            "r_at_same_view": round(float(r[frame_index]), 5),
            "top3": [
                {"view": int(i), "r": round(float(v), 5)}
                for i, v in zip(top.indices.tolist(), top.values.tolist())
            ],
        }
    return results


def decompose_vec(vec: np.ndarray, sid: float, sad: float) -> tuple[float, float, float]:
    """从 vec 拆出 (角度, u 偏移, v 偏移)。

    ``angle_to_vec`` 的探测器中心是 ``-(sid-sad)*(cos a, sin a, 0) + off_u*u + off_v*v``，
    这里做同样的逆运算，不依赖 transforms.json 之外的信息。
    """
    cam = np.asarray(vec[:3], dtype=np.float64)
    det = np.asarray(vec[3:6], dtype=np.float64)
    u_vec = np.asarray(vec[6:9], dtype=np.float64)
    v_vec = np.asarray(vec[9:12], dtype=np.float64)
    angle = float(np.degrees(np.arctan2(cam[1], cam[0])) % 360.0)
    u_unit = u_vec / np.linalg.norm(u_vec)
    v_unit = v_vec / np.linalg.norm(v_vec)
    base = np.array([-(sid - sad) * np.cos(np.radians(angle)),
                     -(sid - sad) * np.sin(np.radians(angle)), 0.0])
    delta = det - base
    return angle, float(delta @ u_unit), float(delta @ v_unit)


def compose_vec(angle_deg: float, paras: dict, offset_u: float, offset_v: float,
                u_sign: float) -> np.ndarray:
    """按 ``angle_to_vec`` 的公式重建 vec，可指定 u 轴朝向。

    ``angle_to_vec`` 用的是 ``source = sad*(cos a, sin a, 0)``（数学逆时针）。真实
    Varian gantry 角是否满足这个约定，正是本模式要检验的对象。
    """
    sid, sad = float(paras["sid"]), float(paras["sad"])
    spacing = np.asarray(paras["proj_spacing"], dtype=np.float64)
    angle = np.radians(angle_deg)
    source = sad * np.array([np.cos(angle), np.sin(angle), 0.0])
    detector = -(sid - sad) * np.array([np.cos(angle), np.sin(angle), 0.0])
    u_unit = u_sign * np.array([-np.sin(angle), np.cos(angle), 0.0])
    v_unit = np.array([0.0, 0.0, -1.0])
    detector = detector + offset_u * u_unit + offset_v * v_unit
    return np.concatenate((source, detector, spacing[0] * u_unit,
                           spacing[1] * v_unit))


def apply_angle_convention(vec: np.ndarray, paras: dict, sign: float = 1.0,
                           offset: float = 0.0, u_sign: float = 1.0) -> np.ndarray:
    """按 ``angle_eff = sign * angle + offset`` 重建 vec。

    ``sign=+1, offset=0, u_sign=+1`` 时原样返回，即当前 ``angle_to_vec`` 的行为。
    探测器偏移在 u_sign 不变时保持不变（它是平板的物理安装量）。
    """
    vec = np.asarray(vec, dtype=np.float64)
    if sign == 1.0 and offset == 0.0 and u_sign == 1.0:
        return vec
    angle, off_u, off_v = decompose_vec(vec, float(paras["sid"]), float(paras["sad"]))
    effective = (sign * angle + offset) % 360.0
    return compose_vec(effective, paras, off_u, off_v, u_sign)


def run_angle_convention(case: dict, frame_indices: list[int], step_mm: float,
                         device: torch.device, sign_candidates,
                         offset_candidates, u_sign_candidates) -> dict:
    """网格搜索角度约定：``angle_eff = s * angle + c``，并试 u 轴两种朝向。

    判据是**同一视角下的相关系数**——不做任何平移搜索，因为平移搜索会掩盖错误。
    约定正确时 DRR 应当直接匹配它自己那个视角。
    """
    paras = case["paras"]
    height, width = (int(v) for v in paras["proj_resolution"])
    sid, sad = float(paras["sid"]), float(paras["sad"])
    volume = torch.as_tensor(make_volume(case["volume_zyx"], "zyx"),
                             dtype=torch.float32, device=device)
    measured = torch.as_tensor(case["projections"], dtype=torch.float32, device=device)
    centered = measured - measured.mean(dim=(1, 2), keepdim=True)
    norms = centered.flatten(1).norm(dim=1)
    origin = torch.as_tensor(paras["volume_origin"], dtype=torch.float32, device=device)
    phy = torch.as_tensor(paras["volume_phy"], dtype=torch.float32, device=device)

    usable = [i for i in frame_indices if i < len(paras["frames"])]
    decomposed = {i: decompose_vec(np.asarray(paras["frames"][i]["vec"], dtype=np.float64),
                                   sid, sad) for i in usable}

    print(f"  从 vec 反解（帧 0）: angle={decomposed[usable[0]][0]:.3f} "
          f"offset_u={decomposed[usable[0]][1]:.3f} offset_v={decomposed[usable[0]][2]:.3f}")
    print(f"\n  {'s':>3s} {'c':>5s} {'u_sign':>7s} | {'mean r(自身)':>13s} "
          f"{'mean r(最佳)':>13s} {'mean|best-i|':>13s} {'命中±3':>8s}")

    results: dict[str, dict] = {}
    for sign in sign_candidates:
        for offset in offset_candidates:
            for u_sign in u_sign_candidates:
                labels = []
                for frame_index in usable:
                    angle, off_u, off_v = decomposed[frame_index]
                    effective = (sign * angle + offset) % 360.0
                    vec = compose_vec(effective, paras, off_u, off_v, u_sign)
                    rays = get_rays(torch.as_tensor(vec, dtype=torch.float32,
                                                    device=device)[None, :],
                                    height, width).reshape(-1, 6)
                    drr = composite(rays, volume, origin, phy,
                                    step_mm).reshape(height, width).float()
                    flat = drr.flatten() - drr.mean()
                    r = (flat[None] * centered.flatten(1)).sum(dim=1) / (
                        flat.norm() * norms + 1e-9)
                    labels.append((frame_index, float(r[frame_index]),
                                   int(torch.argmax(r)), float(r.max())))
                mean_same = float(np.mean([item[1] for item in labels]))
                mean_best = float(np.mean([item[3] for item in labels]))
                mean_gap = float(np.mean([abs(item[2] - item[0]) for item in labels]))
                hits = sum(1 for item in labels if abs(item[2] - item[0]) <= 3)
                key = f"s={sign:+.0f} c={offset:.0f} u={u_sign:+.0f}"
                results[key] = {
                    "sign": sign, "offset_degrees": offset, "u_sign": u_sign,
                    "mean_r_at_same_view": round(mean_same, 5),
                    "mean_r_at_best_view": round(mean_best, 5),
                    "mean_abs_best_minus_frame": round(mean_gap, 2),
                    "hits_within_3_views": hits, "frames": len(labels),
                    "per_frame": [
                        {"frame": f, "r_same": round(rs, 4), "best_view": bv,
                         "r_best": round(rb, 4)}
                        for f, rs, bv, rb in labels
                    ],
                }
                print(f"  {sign:+3.0f} {offset:5.0f} {u_sign:+7.0f} | "
                      f"{mean_same:13.4f} {mean_best:13.4f} {mean_gap:13.2f} "
                      f"{hits:5d}/{len(labels)}")
    best = max(results.items(), key=lambda item: item[1]["mean_r_at_same_view"])
    print(f"\n  >>> 同一视角相关系数最高: {best[0]}  "
          f"r={best[1]['mean_r_at_same_view']:.4f}")
    return results


def run_convention_equivalence(case: dict, frame_stride: int = 1) -> dict:
    """校验方案 A（调用点变换）与方案 B（函数内参数）产生完全相同的 vec。

    ``angle_to_vec`` 内部只读取 ``geometry.sad`` 与 ``geometry.sid``，其余字段不参与
    计算，所以这里用一个只带这两个属性的轻量对象即可，不需要重新解析 Scan.xml。
    """
    from types import SimpleNamespace

    from tools.thorax_preprocessing.projection_io import (
        angle_to_vec, convention_angle,
    )

    paras = case["paras"]
    geometry = SimpleNamespace(sad=float(paras["sad"]), sid=float(paras["sid"]))
    spacing = tuple(float(v) for v in paras["proj_spacing"])
    offset_u, offset_v = (float(v) for v in paras["detector_offset_mm"])

    frames = paras["frames"][::frame_stride]
    max_ab = 0.0
    max_vs_stored = 0.0
    for frame in frames:
        angle = float(frame["angle_degrees"])
        vec_a = angle_to_vec(convention_angle(angle, "varian"), geometry, spacing,
                             offset_u, offset_v)
        vec_b = angle_to_vec(angle, geometry, spacing, offset_u, offset_v,
                             angle_convention="varian")
        max_ab = max(max_ab, float(np.abs(vec_a - vec_b).max()))
        stored = np.asarray(frame["vec"], dtype=np.float64)
        max_vs_stored = max(max_vs_stored, float(np.abs(vec_a - stored).max()))

    same = max_ab < 1e-12
    print(f"  比较帧数            : {len(frames)}")
    print(f"  A vs B 最大逐元素差 : {max_ab:.3e}   -> {'完全一致' if same else '不一致!'}")
    print(f"  varian vs 已存 vec  : 最大差 {max_vs_stored:.3f}"
          f"（这是修复带来的几何改变量）")
    return {
        "frames_compared": len(frames),
        "max_abs_diff_a_vs_b": max_ab,
        "identical": same,
        "max_abs_diff_varian_vs_stored": round(max_vs_stored, 6),
    }


def run_residual_map(case: dict, frame_indices: list[int], step_mm: float,
                     device: torch.device, search: int,
                     angle_triple: tuple[float, float, float] = (1.0, 0.0, 1.0),
                     origin: str = "declared",
                     phy: str = "declared") -> dict:
    """逐帧测残余平移 (dv, du)，看它随 gantry 角如何变化。

    判据：

    * 世界坐标系里的平移误差 δ 投影到探测器上的位移正比于 ``δ · u_unit(g)``，
      而 ``u_unit`` 随角度旋转，所以 **du 会随角度呈正弦变化**。
    * 探测器侧的固定偏移（面板原点、offset 约定）给出的 du **近似恒定**。
    * 因此「正弦 vs 常数」可以把残余误差归类到 origin 类或探测器侧。
    """
    paras = case["paras"]
    height, width = (int(v) for v in paras["proj_resolution"])
    volume = torch.as_tensor(make_volume(case["volume_zyx"], "zyx"),
                             dtype=torch.float32, device=device)
    measured_all = torch.as_tensor(case["projections"], dtype=torch.float32,
                                   device=device)

    true_origin = np.asarray(case["nifti_origin"], dtype=float)
    declared_origin = np.asarray(paras["volume_origin"], dtype=float)
    declared_phy = np.asarray(paras["volume_phy"], dtype=float)
    resolution = np.asarray(paras["volume_resolution"], dtype=float)
    spacing = np.asarray(paras["volume_spacing"], dtype=float)
    origin_vec = true_origin if origin == "true" else declared_origin
    phy_vec = (resolution - 1.0) * spacing if phy == "tight" else declared_phy
    origin_t = torch.as_tensor(origin_vec, dtype=torch.float32, device=device)
    phy_t = torch.as_tensor(phy_vec, dtype=torch.float32, device=device)

    print(f"  origin={origin}  phy={phy}  search=±{search} px")
    print(f"  {'frame':>5s} {'angle':>8s} | {'dv':>5s} {'du':>5s} {'r@0':>7s} "
          f"{'r@best':>7s}")
    rows: list[dict] = []
    for frame_index in frame_indices:
        if frame_index >= len(paras["frames"]):
            continue
        frame = paras["frames"][frame_index]
        vec = apply_angle_convention(np.asarray(frame["vec"], dtype=np.float64),
                                     paras, *angle_triple)
        rays = get_rays(torch.as_tensor(vec, dtype=torch.float32,
                                        device=device)[None, :],
                        height, width).reshape(-1, 6)
        drr = composite(rays, volume, origin_t, phy_t,
                        step_mm).reshape(height, width).float()
        measured = measured_all[frame_index]
        # 与 evaluate_variant 保持完全相同的掩膜定义，否则两个模式的 r 不可比
        body = measured > 0.15 * float(measured.max())
        hit = drr > 0
        mask = body & hit
        if int(mask.sum()) < 100:
            mask = body
        r0 = pearson(drr, measured, mask)
        dv, du, r_best = best_shift(drr, measured, mask, search)
        rows.append({"frame": frame_index, "angle": float(frame["angle_degrees"]),
                     "dv": int(dv), "du": int(du),
                     "r_at_zero": round(r0, 4), "r_at_best": round(r_best, 4)})
        print(f"  {frame_index:5d} {frame['angle_degrees']:8.2f} | {dv:5d} {du:5d} "
              f"{r0:7.4f} {r_best:7.4f}")

    # 分类：du 对角度做 sin/cos 最小二乘拟合，比较「正弦模型」和「常数模型」的残差
    du = np.array([row["du"] for row in rows], dtype=float)
    ang = np.radians([row["angle"] for row in rows])
    dv = np.array([row["dv"] for row in rows], dtype=float)
    design = np.stack([np.cos(ang), np.sin(ang), np.ones_like(ang)], axis=1)
    fit, *_ = np.linalg.lstsq(design, du, rcond=None)
    residual_sinusoid = float(np.sqrt(np.mean((design @ fit - du) ** 2)))
    residual_constant = float(np.sqrt(np.mean((du - du.mean()) ** 2)))
    verdict = ("正弦型 -> 世界坐标平移类（origin 误差）"
               if residual_sinusoid < 0.7 * residual_constant
               else "常数型 -> 探测器侧偏移类")
    print(f"\n  du 均值 {du.mean():+.2f}  标准差 {du.std():.2f}")
    print(f"  拟合 du = {fit[0]:+.3f}*cos(a) {fit[1]:+.3f}*sin(a) {fit[2]:+.3f}")
    print(f"  正弦模型 残差RMS = {residual_sinusoid:.2f} px")
    print(f"  常数模型 残差RMS = {residual_constant:.2f} px")
    print(f"  >>> 判定: {verdict}")
    return {"rows": rows,
            "du_mean": round(float(du.mean()), 4),
            "du_std": round(float(du.std()), 4),
            "sinusoid_fit": [round(float(v), 4) for v in fit],
            "residual_rms_sinusoid": round(residual_sinusoid, 4),
            "residual_rms_constant": round(residual_constant, 4),
            "verdict": verdict}


def _ranks(values: np.ndarray) -> np.ndarray:
    """无并列处理的秩（浮点图像不需要并列处理）。"""
    return np.argsort(np.argsort(values)).astype(np.float64)


def monotone_and_polynomial_scores(drr: np.ndarray, measured: np.ndarray,
                                   mask: np.ndarray) -> dict:
    """用单调非线性重映射检验"实测 vs DRR 的关系是否已经线性"。

    判据：

    * ``r_pearson`` 低而 ``r_spearman`` 高 -> 关系是*单调但非线性*的，与束硬化/散射
      一致（它们给出的是线积分的单调变形）。
    * 两者都低 -> 单调重映射救不回来，说明是**空间错位**而不是值域非线性。
    * ``r_poly3`` 用三次多项式拟合 ``measured = f(drr)``，给一个比秩变换更保守的
      "非线性可解释程度"。
    """
    a = drr[mask]
    b = measured[mask]
    if a.size < 10:
        return {"r_pearson": float("nan"), "r_spearman": float("nan"),
                "r_poly3": float("nan"), "poly3_coef": []}

    def corr(x, y):
        x = x - x.mean()
        y = y - y.mean()
        denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
        return float(x @ y / denominator) if denominator > 0 else float("nan")

    r_pearson = corr(a, b)
    r_spearman = corr(_ranks(a), _ranks(b))
    coefficients = np.polyfit(a, b, 3)
    r_poly3 = corr(np.polyval(coefficients, a), b)
    return {"r_pearson": round(r_pearson, 4),
            "r_spearman": round(r_spearman, 4),
            "r_poly3": round(r_poly3, 4),
            "poly3_coef": [round(float(c), 6) for c in coefficients]}


def run_nonlinearity(case: dict, frame_indices: list[int], step_mm: float,
                     device: torch.device,
                     angle_triple: tuple[float, float, float] = (1.0, 0.0, 1.0),
                     origin: str = "declared", phy: str = "declared") -> dict:
    """逐帧比较「线性相关」与「单调/多项式重映射后相关」，并按角度分段汇总。"""
    paras = case["paras"]
    height, width = (int(v) for v in paras["proj_resolution"])
    volume = torch.as_tensor(make_volume(case["volume_zyx"], "zyx"),
                             dtype=torch.float32, device=device)
    measured_all = torch.as_tensor(case["projections"], dtype=torch.float32,
                                   device=device)

    true_origin = np.asarray(case["nifti_origin"], dtype=float)
    declared_origin = np.asarray(paras["volume_origin"], dtype=float)
    declared_phy = np.asarray(paras["volume_phy"], dtype=float)
    resolution = np.asarray(paras["volume_resolution"], dtype=float)
    spacing = np.asarray(paras["volume_spacing"], dtype=float)
    origin_vec = true_origin if origin == "true" else declared_origin
    phy_vec = (resolution - 1.0) * spacing if phy == "tight" else declared_phy
    origin_t = torch.as_tensor(origin_vec, dtype=torch.float32, device=device)
    phy_t = torch.as_tensor(phy_vec, dtype=torch.float32, device=device)

    print(f"  origin={origin}  phy={phy}")
    print(f"  {'frame':>5s} {'angle':>8s} | {'r_pearson':>9s} {'r_spearman':>10s} "
          f"{'r_poly3':>8s} | {'spearman增益':>11s} {'poly3增益':>9s}")
    rows: list[dict] = []
    for frame_index in frame_indices:
        if frame_index >= len(paras["frames"]):
            continue
        frame = paras["frames"][frame_index]
        vec = apply_angle_convention(np.asarray(frame["vec"], dtype=np.float64),
                                     paras, *angle_triple)
        rays = get_rays(torch.as_tensor(vec, dtype=torch.float32,
                                        device=device)[None, :],
                        height, width).reshape(-1, 6)
        drr = composite(rays, volume, origin_t, phy_t,
                        step_mm).reshape(height, width).float()
        measured = measured_all[frame_index]
        body = measured > 0.15 * float(measured.max())
        mask = (body & (drr > 0))
        if int(mask.sum()) < 100:
            mask = body
        scores = monotone_and_polynomial_scores(
            drr.detach().cpu().numpy().astype(np.float64),
            measured.detach().cpu().numpy().astype(np.float64),
            mask.detach().cpu().numpy())
        scores.update({"frame": frame_index,
                       "angle": float(frame["angle_degrees"])})
        rows.append(scores)
        print(f"  {frame_index:5d} {frame['angle_degrees']:8.2f} | "
              f"{scores['r_pearson']:9.4f} {scores['r_spearman']:10.4f} "
              f"{scores['r_poly3']:8.4f} | "
              f"{scores['r_spearman'] - scores['r_pearson']:+11.4f} "
              f"{scores['r_poly3'] - scores['r_pearson']:+9.4f}")

    # 按角度分段汇总：束硬化预测"AP 方向(0/180)增益大、侧位(90/270)增益小"
    bands = {"AP near 0": (315.0, 360.0), "AP near 0 (wrap)": (0.0, 45.0),
             "oblique 45-135": (45.0, 135.0), "AP near 180": (135.0, 225.0),
             "oblique 225-315": (225.0, 315.0)}
    summary: dict[str, dict] = {}
    print(f"\n  {'角度分段':>20s} {'n':>3s} {'r_pearson':>10s} {'r_spearman':>11s} "
          f"{'r_poly3':>9s} {'spearman增益':>12s}")
    for label, (low, high) in bands.items():
        selected = [row for row in rows
                    if low <= row["angle"] < high
                    and row["r_pearson"] == row["r_pearson"]]
        if not selected:
            continue
        mean_p = float(np.mean([row["r_pearson"] for row in selected]))
        mean_s = float(np.mean([row["r_spearman"] for row in selected]))
        mean_3 = float(np.mean([row["r_poly3"] for row in selected]))
        summary[label] = {"n": len(selected), "r_pearson": round(mean_p, 4),
                          "r_spearman": round(mean_s, 4),
                          "r_poly3": round(mean_3, 4),
                          "spearman_gain": round(mean_s - mean_p, 4)}
        print(f"  {label:>20s} {len(selected):3d} {mean_p:10.4f} {mean_s:11.4f} "
              f"{mean_3:9.4f} {mean_s - mean_p:+12.4f}")

    ap_gain = np.mean([summary[k]["spearman_gain"] for k in summary
                       if "AP" in k] or [float("nan")])
    oblique_gain = np.mean([summary[k]["spearman_gain"] for k in summary
                            if "oblique" in k] or [float("nan")])
    if ap_gain == ap_gain and oblique_gain == oblique_gain:
        verdict = ("AP 方向增益明显更大 -> 支持束硬化/散射（路径越长非线性越强）"
                   if ap_gain > oblique_gain + 0.05
                   else "两个方向增益接近 -> 非线性不是角度依赖的，束硬化解释不足，"
                        "应回到几何/截断方向排查")
        print(f"\n  AP 平均增益 {ap_gain:+.4f}   斜位平均增益 {oblique_gain:+.4f}")
        print(f"  >>> 判定: {verdict}")
        summary["verdict"] = verdict
    return {"rows": rows, "bands": summary}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("variants", "correspondence",
                                           "angle-convention",
                                           "convention-equivalence",
                                           "residual", "nonlinearity"),
                        default="variants",
                        help="correspondence 核对视角对应；angle-convention 搜索约定；"
                             "convention-equivalence 校验方案 A/B 等价；"
                             "residual 测残余平移随角度的变化；"
                             "nonlinearity 用单调重映射判别束硬化/散射")
    parser.add_argument("--residual-origin", choices=("declared", "true"),
                        default="true")
    parser.add_argument("--residual-phy", choices=("declared", "tight"),
                        default="tight")
    parser.add_argument("--correspondence-frames", nargs="+", type=int,
                        default=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
                        help="correspondence / angle-convention 模式使用的帧号")
    parser.add_argument("--sign-candidates", nargs="+", type=float, default=[1.0, -1.0])
    parser.add_argument("--offset-candidates", nargs="+", type=float,
                        default=[0.0, 90.0, 180.0, 270.0])
    parser.add_argument("--u-sign-candidates", nargs="+", type=float, default=[1.0, -1.0])
    parser.add_argument("--angle-sign", type=float, default=1.0,
                        help="correspondence / variants 模式使用的角度符号")
    parser.add_argument("--angle-offset", type=float, default=0.0,
                        help="correspondence / variants 模式使用的角度偏移（度）")
    parser.add_argument("--angle-u-sign", type=float, default=1.0,
                        help="correspondence / variants 模式使用的 u 轴朝向")
    parser.add_argument("--data-root", default="dataset/thorax/syn_data_cbct_gt")
    parser.add_argument("--cases", nargs="+",
                        default=["2026-06-04_065713", "2026-06-04_081709"],
                        help="z 偏差一小一大各取一例，便于区分误差源")
    parser.add_argument("--view-indices", nargs="+", type=int, default=[0, 90, 180, 270])
    parser.add_argument("--step-mm", type=float, default=1.0,
                        help="渲染步长；训练里是 min(volume_spacing) * render.factor")
    parser.add_argument("--shift-search", type=int, default=40,
                        help="整数平移搜索半径（像素）")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or
                          args.device == "cpu" else "cpu")
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    report: dict = {"data_root": str(data_root), "step_mm": args.step_mm,
                    "mode": args.mode,
                    "angle_convention": {"sign": args.angle_sign,
                                         "offset_degrees": args.angle_offset,
                                         "u_sign": args.angle_u_sign},
                    "cases": {}}

    for case_name in args.cases:
        case = load_case(data_root, case_name)
        paras = case["paras"]
        print("=" * 100)
        print(f"病例 {case_name}")
        print(f"  transforms.volume_resolution : {paras['volume_resolution']}")
        print(f"  transforms.volume_origin     : "
              f"{[round(float(v), 3) for v in paras['volume_origin']]}")
        print(f"  transforms.volume_phy        : "
              f"{[round(float(v), 3) for v in paras['volume_phy']]}")
        print(f"  NIfTI 头真实 origin          : "
              f"{[round(float(v), 3) for v in case['nifti_origin']]}")
        print(f"  sitk 数组 shape (Z,Y,X)      : {case['volume_zyx'].shape}")
        print(f"  投影 shape (N,H,W)           : {case['projections'].shape}")
        print(f"  SID/SAD = {paras['sid']}/{paras['sad']} "
              f"-> m = {paras['sid'] / paras['sad']:.4f}")

        case_report: dict = {"geometry": {
            "volume_resolution": paras["volume_resolution"],
            "declared_origin": [float(v) for v in paras["volume_origin"]],
            "true_origin": [float(v) for v in case["nifti_origin"]],
            "volume_phy": [float(v) for v in paras["volume_phy"]],
            "array_shape_zyx": list(case["volume_zyx"].shape),
            "sid": paras["sid"], "sad": paras["sad"],
        }, "views": {}}

        if args.mode == "nonlinearity":
            print("\n  --- 单调非线性重映射判别（束硬化/散射）---")
            case_report["nonlinearity"] = run_nonlinearity(
                case, args.correspondence_frames, args.step_mm, device,
                (args.angle_sign, args.angle_offset, args.angle_u_sign),
                args.residual_origin, args.residual_phy)
            report["cases"][case_name] = case_report
            del case
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if args.mode == "residual":
            print("\n  --- 残余平移 vs gantry 角 ---")
            case_report["residual"] = run_residual_map(
                case, args.correspondence_frames, args.step_mm, device,
                args.shift_search,
                (args.angle_sign, args.angle_offset, args.angle_u_sign),
                args.residual_origin, args.residual_phy)
            report["cases"][case_name] = case_report
            del case
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if args.mode == "convention-equivalence":
            print("\n  --- 方案 A / 方案 B 等价性校验 ---")
            case_report["convention_equivalence"] = run_convention_equivalence(case)
            report["cases"][case_name] = case_report
            del case
            continue

        variants = build_variants(case)

        if args.mode == "correspondence":
            print("\n  --- frames 与 proj 的视角对应关系 ---")
            case_report["correspondence"] = run_correspondence(
                case, args.correspondence_frames, args.step_mm, device,
                (args.angle_sign, args.angle_offset, args.angle_u_sign))
            report["cases"][case_name] = case_report
            del case
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if args.mode == "angle-convention":
            print("\n  --- 角度约定网格搜索 ---")
            case_report["angle_convention"] = run_angle_convention(
                case, args.correspondence_frames, args.step_mm, device,
                args.sign_candidates, args.offset_candidates,
                args.u_sign_candidates)
            report["cases"][case_name] = case_report
            del case
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        for view_index in args.view_indices:
            if view_index >= len(paras["frames"]):
                print(f"  [SKIP] view {view_index} 超出 frames 范围")
                continue
            measured = case["projections"][view_index]
            measured_max = float(measured.max())
            print(f"\n  --- 视角 {view_index}（实测投影 max={measured_max:.4f}）---")
            print(f"  {'变体':22s} {'r_body':>9s} {'r_all':>9s} "
                  f"{'best(dv,du)':>14s} {'r@shift':>9s} {'nrmse':>9s}")
            view_report: dict = {}
            for name, spec in variants.items():
                result = evaluate_variant(case, spec, view_index, args.step_mm,
                                          device, args.shift_search,
                                          (args.angle_sign, args.angle_offset,
                                           args.angle_u_sign))
                view_report[name] = result
                print(f"  {name:22s} {result['r_body']:9.4f} {result['r_all']:9.4f} "
                      f"{str((result['best_shift_dv'], result['best_shift_du'])):>14s} "
                      f"{result['r_body_at_best_shift']:9.4f} "
                      f"{result['nrmse_at_best_scale']:9.4f}")
            case_report["views"][str(view_index)] = view_report
            del measured

        # 汇总：按 r_body 排序，给出每个变体的跨视角均值
        print(f"\n  === {case_name} 跨视角平均 ===")
        totals: dict[str, list[float]] = {name: [] for name in variants}
        for view_report in case_report["views"].values():
            for name, result in view_report.items():
                if result["r_body"] == result["r_body"]:
                    totals[name].append(result["r_body"])
        ranking = sorted(
            ((name, float(np.mean(values))) for name, values in totals.items() if values),
            key=lambda item: -item[1],
        )
        for name, mean_r in ranking:
            print(f"    {name:22s} 平均 r_body = {mean_r:.4f}")
        case_report["mean_r_body"] = {name: round(value, 5) for name, value in ranking}
        report["cases"][case_name] = case_report
        del case
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.output_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"\n报告已写入 {output_path}")


if __name__ == "__main__":
    main()
