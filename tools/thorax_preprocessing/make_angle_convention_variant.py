"""生成 ``syn_data_cbct_gt_v2``：只修 gantry 角度约定，其余几何保持不变。

纯新增文件，不改动任何现有模块，也不覆盖任何已有数据。每个病例的新目录里
``proj.nii.gz`` / ``gt_volume.nii.gz`` 等体数据用符号链接复用，只重写
``transforms.json`` 的 ``frames[].vec``。

为什么可以只重写 ``vec``
------------------------
``proj.nii.gz`` 的像素与几何无关——``convert_projections`` 只是把挑选出的帧按顺序
堆进 ``stack``。几何信息全部编码在 ``frames[].vec`` 里，而 ``vec`` 是
``angle_to_vec(angle, geometry, spacing, offset_u, offset_v)`` 的确定性函数。因此把
``angle`` 映射成 ``90 - angle``（实测确认的 Varian 约定）后重算 ``vec`` 即可，不需要
重读 XIM，也不需要重新挑帧。

自校验
------
脚本会先用 **原约定**（``simulation``）从 ``transforms.json`` 反解的
``(angle, offset_u, offset_v)`` 重建 ``vec``，与文件里存的值逐元素比较。只有该复原
误差接近 0（说明反解-重建闭环精确）时，才认为 ``varian`` 版本可信。这一条会在每个
病例上执行，任何一例不达标都会记入汇总。

用法（在项目根目录执行）：

    /autdl-tmp/conda_env/GeoAware/bin/python \
        tools/thorax_preprocessing/make_angle_convention_variant.py \
        --source dataset/thorax/syn_data_cbct_gt \
        --output dataset/thorax/syn_data_cbct_gt_v2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.thorax_preprocessing.projection_io import (  # noqa: E402
    angle_to_vec, convention_angle,
)
from tools.thorax_preprocessing.verify_projection_alignment import (  # noqa: E402
    decompose_vec,
)

MARKER = ".thorax_angle_convention_variant"
DEFAULT_SOURCE = "dataset/thorax/syn_data_cbct_gt"
# 体数据文件原样复用；proj.nii.gz 与几何无关所以也能复用
SHARED_FILES = (
    "proj.nii.gz",
    "gt_volume.nii.gz",
    "cbct_volume.nii.gz",
    "ct_volume.nii.gz",
)
TO_CONVENTION = "varian"
FROM_CONVENTION = "simulation"


def link_file(source: Path, destination: Path, mode: str) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "symlink":
        os.symlink(os.path.relpath(source, destination.parent), destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def convert_case(source_dir: Path, destination_dir: Path, args) -> dict:
    transforms_path = source_dir / "transforms.json"
    with transforms_path.open(encoding="utf-8") as handle:
        params = json.load(handle)

    frames = params.get("frames")
    if not frames:
        raise ValueError("transforms.json has no frames")

    from_probe = params.get("angle_convention", FROM_CONVENTION)
    if from_probe == TO_CONVENTION:
        raise ValueError(
            f"{source_dir} 已经记录了 angle_convention={TO_CONVENTION}，"
            "不需要再转换（否则会二次施加）"
        )

    sid, sad = float(params["sid"]), float(params["sad"])
    spacing = tuple(float(v) for v in params["proj_spacing"])

    # 复原自校验：用原约定反解再重建，必须与存储值一致
    max_recover = 0.0
    new_frames = []
    for frame in frames:
        stored = np.asarray(frame["vec"], dtype=np.float64)
        angle, offset_u, offset_v = decompose_vec(stored, sid, sad)
        recovered = angle_to_vec(convention_angle(angle, FROM_CONVENTION),
                                 type("G", (), {"sad": sad, "sid": sid}), spacing,
                                 offset_u, offset_v)
        max_recover = max(max_recover, float(np.abs(recovered - stored).max()))
        # 目标：同一组 (angle, offset_u, offset_v)，换成 varian 约定
        new_vec = angle_to_vec(convention_angle(angle, TO_CONVENTION),
                               type("G", (), {"sad": sad, "sid": sid}), spacing,
                               offset_u, offset_v)
        updated = dict(frame)
        updated["vec"] = new_vec.tolist()
        updated["angle_degrees"] = angle
        new_frames.append(updated)

    if max_recover > 1e-6:
        raise ValueError(
            f"{source_dir}: 复原自校验失败（max diff {max_recover:.3e}），"
            "说明反解-重建闭环不精确，拒绝生成"
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in SHARED_FILES:
        candidate = source_dir / name
        if candidate.exists():
            link_file(candidate, destination_dir / name, args.link_mode)

    params["frames"] = new_frames
    params["angle_convention"] = TO_CONVENTION
    params["angle_convention_source"] = str(source_dir)
    params["angle_convention_note"] = (
        f"frames[].vec recomputed from {source_dir}'s transforms.json by mapping "
        f"angle -> 90 - angle (Varian gantry convention, verified against real "
        f"projections). All other geometry, the projections and the label volume "
        f"are reused unchanged."
    )
    with (destination_dir / "transforms.json").open("w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2, ensure_ascii=False)
    # 标明这是一个"几何已转换"的目录，避免被 prepare_thorax 当作原始输出覆盖
    (destination_dir / MARKER).write_text(f"{TO_CONVENTION}\n", encoding="utf-8")

    return {
        "case": source_dir.name,
        "frames": len(new_frames),
        "recover_max_diff": max_recover,
        "gt_source": params.get("gt_source"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default="dataset/thorax/syn_data_cbct_gt_v2")
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"),
                        default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source)
    if not source_root.is_absolute():
        source_root = REPO_ROOT / source_root
    output_root = Path(args.output)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    if output_root.exists():
        if not args.overwrite and not (output_root / MARKER).exists():
            raise SystemExit(
                f"{output_root} 已存在且不是本脚本生成的目录，拒绝覆盖。"
                "确认无误后加 --overwrite。"
            )
    output_root.mkdir(parents=True, exist_ok=True)

    cases = sorted(p.name for p in source_root.iterdir()
                   if p.is_dir() and (p / "transforms.json").is_file())
    if args.cases:
        wanted = set(args.cases)
        cases = [c for c in cases if c in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit(f"{source_root} 下没有可用病例")

    print(f"源目录  : {source_root}")
    print(f"输出目录: {output_root}")
    print(f"病例数  : {len(cases)}  角度约定: {FROM_CONVENTION} -> {TO_CONVENTION}")
    print(f"链接方式: {args.link_mode}\n")

    summary = {"source": str(source_root), "output": str(output_root),
               "from_convention": FROM_CONVENTION, "to_convention": TO_CONVENTION,
               "link_mode": args.link_mode, "cases": {}}
    failed: list[tuple[str, str]] = []
    for index, case in enumerate(cases, 1):
        try:
            record = convert_case(source_root / case, output_root / case, args)
        except Exception as error:  # noqa: BLE001 - 逐病例记录，中断无意义
            failed.append((case, str(error)))
            print(f"[{index}/{len(cases)}] {case}: 失败 - {error}", flush=True)
            continue
        summary["cases"][case] = record
        if index % 25 == 0 or index == len(cases):
            print(f"[{index}/{len(cases)}] {case}: ok "
                  f"({record['frames']} 帧, 复原误差 {record['recover_max_diff']:.2e})",
                  flush=True)

    summary["succeeded"] = len(summary["cases"])
    summary["failed"] = len(failed)
    summary["failed_detail"] = [{"case": c, "reason": r} for c, r in failed]
    worst = max((r["recover_max_diff"] for r in summary["cases"].values()),
                default=0.0)
    summary["worst_recover_diff"] = worst
    with (output_root / "batch_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\n成功 {len(summary['cases'])} / 失败 {len(failed)}")
    print(f"全部病例的最大复原误差 = {worst:.3e}（应为 ~0，否则数据不可信）")
    if failed:
        print("失败列表见 batch_summary.json 的 failed_detail")
    print(f"\n下一步验证：\n"
          f"  python tools/thorax_preprocessing/verify_projection_alignment.py \\\n"
          f"    --mode correspondence --data-root {args.output} \\\n"
          f"    --cases <任意病例>")


if __name__ == "__main__":
    main()
