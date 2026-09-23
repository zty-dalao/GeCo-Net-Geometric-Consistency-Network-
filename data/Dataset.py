import json
import os
import torch
import SimpleITK as sitk
import numpy as np
from models.render import angle2vec, get_rays, composite
from tqdm import tqdm

class CBCTDataset(torch.utils.data.Dataset):
    """
    Dataset from CBCT projection
    """
    def __init__(
        self, args, stage="train"
    ):
        """
        :param args
        :param stage train | val | test | visual
        """
        super().__init__()
        self.args = args
        self.stage = stage
        self.angle_sampling = args.angle_sampling
        self.device = args.device
        self.datadir = args.datadir
        dataset_split_json = os.path.join('./data/dataset_split', args.datatype+'_split.json')
        with open(dataset_split_json, 'r') as file:
            json_data = json.load(file)
        dataset_split = json_data[stage]
        self.dataset_split = dataset_split
        print("Loading CBCT dataset", self.datadir, "stage:",self.stage)    
    
    def __len__(self):
        return len(self.dataset_split)
    
    def __getitem__(self, index):

        # load paras
        paras_json = os.path.join(self.datadir, self.dataset_split[index],'transforms.json')
        with open(paras_json, 'r') as file:
            paras = json.load(file)

        # load volume
        volume_path = os.path.join(self.datadir, self.dataset_split[index], "gt_volume.nii.gz")
        volume = sitk.GetArrayFromImage(sitk.ReadImage(volume_path))
        volume = np.clip(volume, 0, volume.max())
        volume_tensor = torch.tensor(volume, dtype=torch.float32, device=self.device)

        # basic information
        start, end, nviews = self.args.start, self.args.end, self.args.nviews

        if self.angle_sampling == "uniform":
            # load images
            img_path = os.path.join(self.datadir, self.dataset_split[index], 'proj.nii.gz')
            proj = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            proj = np.clip(proj, 0, proj.max())
            angle_per_view = paras['angle_per_view']
            start_index = int(np.round(start/angle_per_view))
            end_index = int(np.round(end/angle_per_view))
            indices = np.linspace(start_index, end_index, nviews, endpoint=False, dtype=int)
            all_imgs = torch.tensor(proj[indices], dtype=torch.float32, device=self.device).unsqueeze(1)

            # load poses
            vecs = []
            for i in indices:
                frame = paras['frames'][i]
                vec = torch.tensor(frame['vec'], dtype=torch.float32, device=self.device)
                vecs.append(vec)
            vecs = torch.stack(vecs)

        elif self.angle_sampling == "random":  
            # we only recommend random sampling during evaluation because X-ray simulation is really slow, 
            # only for dental/spine dataset
            # basic information
            # ---- half-fan / 角度约定安全门 ----
            # 本分支用 models/render.angle2vec 在线合成 DRR。该函数有两条与真实扫描数据
            # 不兼容的地方，且都会“静默跑通不报错”：
            #   1) 它的探测器中心写死为 -(sid-sad)*(cos a, sin a, 0)，无法表达 half-fan
            #      的横向偏移（thorax 实测 ImagerLat = -175.5 mm ≈ 104 像素，占图幅 41%）；
            #   2) 它用的是“数学逆时针”角度约定，而真实 Varian 数据实测是
            #      angle -> 90 - angle（见 transforms.json 的 angle_convention 字段）。
            # 因此对本类数据必须走 uniform 分支（它直接读取 frames[].vec）。
            detector_offset = paras.get("detector_offset_mm") or [0.0, 0.0]
            recorded_convention = paras.get("angle_convention", "simulation")
            offset_magnitude = max(abs(float(v)) for v in detector_offset)
            if offset_magnitude > 1.0 or recorded_convention != "simulation":
                raise ValueError(
                    f"--angle_sampling random 无法用于 {self.dataset_split[index]}："
                    f"该病例的 detector_offset_mm={list(detector_offset)}"
                    f"（half-fan 偏移），angle_convention={recorded_convention!r}。"
                    "models/render.angle2vec 既不能表达探测器偏移，也用的是与真实扫描"
                    "不同的角度约定，在线合成 DRR 会与实测投影整体错位（实测约 104 像素 "
                    "= 图幅 41%）。请改用 --angle_sampling uniform：它直接读取 "
                    "transforms.json 的 frames[].vec，几何与实测一致。"
                )
            isocenter = [0, 0, 0]
            sad = paras['sad']
            sid = paras['sid']
            proj_spacing = paras['proj_spacing']
            W, H = paras['proj_resolution']
            factor = 0.5
            chunksize = 65536
            volume_phy = torch.tensor(paras['volume_phy']).to(self.device)
            volume_origin = torch.tensor(paras['volume_origin']).to(self.device)
            volume_spacing = torch.min(torch.tensor(paras['volume_spacing'])).to(self.device).to(torch.float32)
            render_step_size = volume_spacing * factor

            # pose generation
            angles = np.random.uniform(start, end, nviews)
            vecs = []
            for angle in tqdm(angles, desc='Projection Geometry Production'):
                angle *= np.pi / 180
                vec = angle2vec(angle, 0, isocenter, sid, sad, proj_spacing[0], proj_spacing[1])
                vec = torch.tensor(vec, dtype=torch.float32, device=self.device)
                vecs.append(vec)
            vecs = torch.stack(vecs)
            cam_rays = get_rays(vecs, H, W)

            # projection generation
            all_imgs = []
            for i in tqdm(range(nviews), desc='Projection Generation'):
                rays = cam_rays[i, ...]
                rays = rays.reshape(-1, rays.shape[-1])
                projection = composite(rays, volume_tensor, volume_origin, volume_phy, render_step_size, chunksize=chunksize)
                projection = projection.reshape(H, W)
                all_imgs.append(projection)
            all_imgs = torch.stack(all_imgs).unsqueeze(1)

        result = {
            "paras": paras,
            "3Dvolume": volume_tensor,
            "images": all_imgs,
            "poses": vecs, 
            "obj_index": paras['obj_index']
        }

        return result


def describe_gt_source(args, stages=("train", "val", "test", "visual")):
    """Report, and optionally enforce, where the 3-D training labels come from.

    ``CBCTDataset`` always reads ``<datadir>/<case>/gt_volume.nii.gz``; there is no
    command-line switch that picks the label volume.  The provenance of that file is
    recorded in each case's ``transforms.json`` by ``tools.thorax_preprocessing.
    prepare_thorax`` (``"gt_source": "registered-ct"`` means the label is the
    registered pCT, while ``"cbct"`` means it is the unregistered CBCT).

    Call this before building the dataloaders so a run cannot silently train against
    the wrong volume.  ``--require-gt-source`` turns the report into a hard check.
    """
    split_path = os.path.join('./data/dataset_split', args.datatype + '_split.json')
    with open(split_path, 'r') as handle:
        split = json.load(handle)
    cases = []
    for stage in stages:
        for name in split.get(stage, []):
            if name not in cases:
                cases.append(name)

    sources, missing_field, missing_transforms, missing_volume = {}, [], [], []
    grids, non_divisible = {}, []
    for name in cases:
        case_dir = os.path.join(args.datadir, name)
        transforms_path = os.path.join(case_dir, 'transforms.json')
        if not os.path.isfile(transforms_path):
            missing_transforms.append(name)
            continue
        if not os.path.isfile(os.path.join(case_dir, 'gt_volume.nii.gz')):
            missing_volume.append(name)
            continue
        with open(transforms_path, 'r') as handle:
            transforms = json.load(handle)
        source = transforms.get('gt_source')
        if source is None:
            missing_field.append(name)
        else:
            sources[source] = sources.get(source, 0) + 1
        resolution = transforms.get('volume_resolution')
        if resolution:
            key = tuple(int(value) for value in resolution)
            grids[key] = grids.get(key, 0) + 1
            if any(value % 4 for value in key):
                non_divisible.append((name, key))

    report = {
        "datadir": args.datadir,
        "cases": len(cases),
        "label_file": "gt_volume.nii.gz",
        "gt_source_counts": sources,
        "grids": {"x".join(str(v) for v in key): count for key, count in grids.items()},
        "grid_not_divisible_by_4": non_divisible,
        "missing_transforms": missing_transforms,
        "missing_gt_volume": missing_volume,
        "transforms_without_gt_source": missing_field,
    }

    print("=" * 72)
    print(f"3D 标签体数据（gt_volume.nii.gz）来源: {args.datadir}")
    if not cases:
        print("  [WARN] split 内没有任何病例")
    for source, count in sorted(sources.items()):
        label = {
            "registered-ct": "配准后的 pCT（推荐）",
            "ct": "未配准的计划 CT",
            "cbct": "CBCT（非配准，仅用于链路自检）",
            "cbct-fixed": "重采样到训练网格的 CBCT（与投影同源配对，248x248xN @2mm）",
        }.get(source, "未知")
        print(f"  transforms.json gt_source={source}: {count} 例  → {label}")
    if missing_field:
        print(
            f"  [WARN] {len(missing_field)} 例的 transforms.json 没有 gt_source 字段，"
            f"无法确认标签来源（例如 {missing_field[0]}）"
        )
    for grid, count in sorted(report["grids"].items()):
        print(f"  体积网格 {grid}: {count} 例")
    if non_divisible:
        print(
            f"  [WARN] {len(non_divisible)} 例的体尺寸不能被 4 整除，训练无法进行："
            f"例如 {non_divisible[0][0]} {non_divisible[0][1]}"
        )
    if missing_transforms:
        print(f"  [WARN] {len(missing_transforms)} 例缺少 transforms.json：例如 {missing_transforms[0]}")
    if missing_volume:
        print(f"  [WARN] {len(missing_volume)} 例缺少 gt_volume.nii.gz：例如 {missing_volume[0]}")

    required = getattr(args, "require_gt_source", None)
    if required is not None:
        problems = []
        if missing_transforms:
            problems.append(f"{len(missing_transforms)} 例缺少 transforms.json")
        if missing_volume:
            problems.append(f"{len(missing_volume)} 例缺少 gt_volume.nii.gz")
        if missing_field:
            problems.append(f"{len(missing_field)} 例的 transforms.json 没有 gt_source 字段")
        unexpected = {source: count for source, count in sources.items() if source != required}
        if unexpected:
            problems.append(f"gt_source 不是 {required} 的病例：{unexpected}")
        if problems:
            raise RuntimeError(
                f"--require-gt-source {required} 校验失败：" + "；".join(problems)
                + "。请用 tools.thorax_preprocessing.prepare_thorax 的 "
                f"--gt-source {required} 重新生成 syn_data，或去掉该参数。"
            )
        print(f"  [OK] --require-gt-source {required} 校验通过（{len(cases)} 例）")
    print("=" * 72)
    return report
