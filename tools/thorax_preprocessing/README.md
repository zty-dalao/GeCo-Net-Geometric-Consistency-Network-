# Thorax 数据预处理

这组脚本把 `dataset/thorax` 中混合的 DICOM 与 Varian XIM 数据转换为本项目
`data/Dataset.py` 可直接读取的病例目录：

```text
dataset/thorax/syn_data/<case>/
├── gt_volume.nii.gz       # 训练标签；正式训练用配准后的 pCT（--gt-source registered-ct）
├── proj.nii.gz            # (N, H, W) 的非负线积分投影
├── transforms.json        # 每帧 12 维扫描几何和体数据几何
├── cbct_volume.nii.gz     # CBCT，HU -> 线性衰减系数 μ
└── ct_volume.nii.gz       # 裁剪后的计划 CT，HU -> μ
```

当前原始病例按同一个病例目录名配对。例如：

```text
dataset/thorax/image/2026-06-04_065713/       # 该病例的CT、CBCT DICOM
dataset/thorax/projection/2026-06-04_065713/  # 该病例的XIM、Scan.xml
                         │
                         └── 合并为一个训练病例
dataset/thorax/syn_data/2026-06-04_065713/
```

因此不会在 `syn_data` 下分别创建“CT文件夹”和“投影文件夹”。两类数据必须进入同一个病例目录，
才能由数据加载器按一次索引同时取到 `gt_volume.nii.gz`、`proj.nii.gz` 和 `transforms.json`。
输出病例名一律取自病人文件夹名（`image/<case>` 与 `projection/<case>` 必须同名），脚本按病人逐个转换。
配准中间结果单独保存在 `dataset/thorax/registration/current`，不作为第二个训练病例。

## 当前数据识别结果与处理原则

- DICOM 不是靠文件名区分。脚本按 `SeriesInstanceUID` 分组，忽略 RTSTRUCT，再以
  Manufacturer/SeriesDescription 区分计划 CT 与 Varian CBCT；也可以显式传 UID。
- 当前病例中计划 CT 和 CBCT 的 `FrameOfReferenceUID` 不同。中心裁剪只能统一轴向覆盖范围，
  **不能完成空间配准**。正式训练应使用 `--gt-source registered-ct` 并指向
  `registration/current`；`--gt-source cbct` 只用于无配准时检查投影转换和数据链路。
- 只读取 `Acquisitions/<id>/Proj_*.xim` 作为患者投影；`Calibrations` 下的 XIM 只用于空气/弓形
  滤板校正，不会混入训练帧。
- `projection-mode=log` 先按源角度为患者帧匹配最近的空气/弓形滤板校准帧，用每帧
  `KVNormChamber` 对患者帧和空气帧做曝光归一化，再执行
  `p=-log(max(I,eps)/max(I0,eps))` 并裁剪至非负值；这是从探测器强度
  转为网络所需线积分的基础校正。数据未提供暗场时无法做 dark-field 校正，若后续拿到暗场，应改为
  `-log((I-dark)/(I0-dark))`。
- 几何参数 SAD、SID、探测器像素间距、半扇扫描 lateral offset 和 source angle offset 均从
  `Scan.xml` 读取。优先从每个 XIM 的 `KVSourceRtn` 读取射线源角度，并读取逐帧探测器 offset；
  缺失时才按扫描起止角均匀插值。原始 492 帧含加减速段，默认按实测角度抽取为 360 个均匀视角。

### `proj.nii.gz` 保存了哪些投影

本项目的 dental/spine 数据加载器原生约定每个病例使用一个 `proj.nii.gz`，其数组形状为
`(N, H, W)`；这里第一维是投影视角，不是人体的空间 z 轴。对应每一帧的源位置、探测器中心、
u/v 像素向量、实测角度和原始 XIM 文件名写在 `transforms.json` 的 `frames` 中。因此不需要为
每个 view 再创建一个 Python 专用的 pickle 文件。

当前 thorax 原始患者扫描有492帧。转换阶段默认按实测角度选择360个覆盖 `[0, 360)` 的均匀
视角，完成空气校正、曝光归一化、负对数和探测器重采样后写入 `proj.nii.gz`。它不是原始 XIM
所有字段的无损副本；原始计数、XIM 私有属性以及未入选的加减速帧仍保留在原始 XIM 中。
训练或推理使用 `--nviews=20 --angle_sampling=uniform` 时，`Dataset` 才从这360帧中按角度范围
确定性取20帧，并同步读取 `transforms.json` 中相同索引的几何参数。也就是说，20-view 是加载阶段
的稀疏采样，不是转换阶段只保存20张投影。

## 环境

在项目根目录运行：

```powershell
conda activate deeplearning
pip install pydicom
```

项目自带的 `dataset/thorax/ximreader` 是 Varian 开源参考实现，但依赖旧版 Python。这里的
`xim_io.py` 沿用同一 HND 格式算法，并移除了 `docutils/matplotlib` 等与批处理无关的依赖。

## 1. 先检查原始数据

```powershell
python -m tools.thorax_preprocessing.inspect_thorax --root dataset/thorax
```

输出包括 CT/CBCT 的 UID、层数、spacing、Frame of Reference，以及投影数量、空气校准帧数量、
SAD/SID 和 XIM 属性名。新增病例后应先执行一次。

## 2. 转换

### 将 Varian XIM 转为 `proj.nii.gz`

投影转换集成在 `prepare_thorax.py` 中，它按**病人文件夹**逐个转换：对每个 `image/<case>`
单独做 `discover_series` + `select_series`，读取该病人自己的
`projection/<case>/Acquisitions/<id>/Proj_*.xim`，完成空气校正、曝光归一化、负对数变换、
均匀角度选择和探测器重采样，然后把体数据、投影和 `transforms.json` 写入
`dataset/thorax/syn_data/<case>/`。正式训练的命令是：

```powershell
conda activate deeplearning

python -m tools.thorax_preprocessing.prepare_thorax `
  --root dataset/thorax `
  --output dataset/thorax/syn_data `
  --gt-source registered-ct `
  --registered-ct-root dataset/thorax/registration/current `
  --ct-crop match-cbct-center `
  --projection-bin 1 `
  --projection-resolution 256 `
  --output-views 360
```

`--registered-ct-root` 是批处理形式，脚本按
`<root>/<病人文件夹名>/registered_ct_mu.nii.gz` 取每个病人自己的配准 CT；单病例场景仍可用
`--registered-ct-mu` 直接指定一个文件。先用 `--limit N` 小批量验证参数与耗时：

```powershell
  --limit 3
```

只检查投影转换和数据链路、尚无配准时，可改用 `--gt-source cbct`（此时 GT 是原生 CBCT 网格，
深度往往不能被 4 整除，不能直接训练）。

`--gt-source` 只选择 `gt_volume.nii.gz` 的来源，不参与XIM投影像素的生成。在其余投影参数相同
时，`cbct` 与 `registered-ct` 两种模式得到的 `proj.nii.gz` 相同；不同的是GT内容以及
`transforms.json` 中的体积网格。原生CBCT网格的深度（如118）不能被模型的4倍上采样尺度整除，
因此正式训练用 `registered-ct`：配准网格是 2 mm 等方且各维可被 4 整除（例如
248×248×120），`batch_summary.json` 的 `gt_not_divisible_by_4` 会列出不满足该条件的病例。

单个病人出错只会记录该病人并继续处理后面的病人，不会中断整批。会被跳过的情形包括：缺
`Scan.xml` / `Acquisitions/<id>` / `Proj_*.xim` 的投影目录、缺 CT 或 CBCT 序列、缺
`Calibrations/AIR-*` 空气校准帧（`--projection-mode log` 必需，`raw` 可绕过），以及
`--gt-source registered-ct` 时缺少对应的 `registered_ct_mu.nii.gz`。每个病人的结论写入输出根目录的
`batch_summary.json`（`succeeded` / `skipped_detail` / `failed_detail` / `cases`）。

原始探测器是 1280×320，spacing 为 0.336×1.344 mm，物理视野约为
430.08×430.08 mm。默认按物理空间线性插值为 256×256、1.68×1.68 mm，避免二维 CNN 将
各向异性像素误认为普通方形像素。`--projection-bin` 可在插值前做 block mean；推荐保持为1，
避免先把垂直方向从320降到160再放大。

CT 默认按 CBCT 的物理 z 覆盖长度中心裁剪。若已经人工确定正确范围，显式范围更可靠：

```powershell
python -m tools.thorax_preprocessing.prepare_thorax `
  --ct-slices 60:140 `
  --projection-bin 1 `
  --projection-resolution 256 `
  --overwrite
```

切片范围采用按 `ImageOrientationPatient` 法向排序后的 Python 半开区间 `[start, end)`，不是文件名
顺序。若自动分类不正确，用检查脚本输出的 UID 指定：

```powershell
python -m tools.thorax_preprocessing.prepare_thorax `
  --ct-series-uid <planning-CT-SeriesInstanceUID> `
  --cbct-series-uid <Varian-CBCT-SeriesInstanceUID>
```

主要可调参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--gt-source` | `cbct` | `gt_volume.nii.gz` 使用 CBCT、未配准 CT 或 `registered-ct`；正式训练用 `registered-ct` |
| `--registered-ct-root` | 无 | `--gt-source registered-ct` 的批处理形式：含 `<病人>/registered_ct_mu.nii.gz` 的父目录 |
| `--registered-ct-mu` | 无 | 单病例形式，直接指定一个已配准 CT 文件 |
| `--ct-crop` | `match-cbct-center` | CT 中心裁到与 CBCT 相同的 z 物理长度；`none` 保留完整 CT |
| `--ct-slices` | 无 | 覆盖自动裁剪，显式指定 `START:END` |
| `--projection-mode` | `log` | 空气校正并取负对数；`raw` 仅用于诊断 |
| `--projection-bin` | `1` | 插值前的探测器 block mean 因子 |
| `--projection-resolution` | `256` | 投影统一后的正方形分辨率；0 表示保留 bin 后的形状 |
| `--output-views` | `360` | 根据实测源角度就近抽取的均匀输出视角数 |
| `--limit` | `0` | 只处理前 N 个病人文件夹（0 表示全部），便于小批量试跑 |
| `--detector-offset-u-mm` | Scan.xml | 覆盖 ImagerLat；只在核对几何后修改 |
| `--max-line-integral` | `20` | 投影异常值上限 |

重新生成已有输出必须显式加 `--overwrite`，避免误覆盖。

## CT 到 CBCT 自动配准

独立配准入口以 CBCT 为 fixed image、计划 CT 为 moving image，执行身体掩膜提取、轴向粗搜索、
多分辨率刚性配准和仿射配准，最后直接把完整 CT 重采样到 CBCT 网格：

```powershell
python -m tools.thorax_preprocessing.registration `
  --root dataset/thorax `
  --output dataset/thorax/registration/current `
  --target-spacing-mm 2 `
  --size-multiple 4
```

### 方案A：使用身体掩膜 `MOMENTS` 重心初始化

初始化逻辑独立在 `registration_initialization.py` 中，并通过 `--initializer` 接入主配准流程：

| 参数值 | 行为 |
|---|---|
| `geometry` | 默认值；保持原流程，按CT和CBCT图像网格的几何中心初始化 |
| `moments` | 按二值身体掩膜的物理质心初始化，再执行原有z粗搜索、6自由度刚性和仿射配准 |

这里使用身体掩膜而不是原始灰度强度计算MOMENTS，减少CT/CBCT强度差异、CBCT散射和床板对重心的
影响。开启参数后会作用于本次批处理的**全部病人**，与质控是否通过无关。建议写到新的输出根目录，
保留旧结果用于逐病例比较：

```bash
python -m tools.thorax_preprocessing.registration \
  --root dataset/thorax \
  --output dataset/thorax/registration/moments \
  --initializer moments \
  --target-spacing-mm 2 \
  --size-multiple 4
```

如果确定要覆盖原来的 `registration/current/<病人>/`，使用：

```bash
python -m tools.thorax_preprocessing.registration \
  --root dataset/thorax \
  --output dataset/thorax/registration/current \
  --initializer moments \
  --target-spacing-mm 2 \
  --size-multiple 4 \
  --overwrite
```

每例 `registration_metrics.json` 会增加：

```json
"initialization": {
  "method": "moments",
  "fixed_center_mm": [0.0, 0.0, 0.0],
  "moving_center_mm": [0.0, 0.0, 0.0]
}
```

实际数值为该病例CBCT与pCT身体掩膜在DICOM物理坐标中的质心。程序日志也会打印两个质心，便于
确认MOMENTS是否产生了明显的x/y/z初始平移。呼吸导致的肺、膈肌局部形变不能由重心初始化、刚性
或仿射完全消除；本参数改善的是全局初值，仍必须结合 `registration_qa.png` 和质控指标检查结果。

### 提高z方向粗搜索采样数

原配置 `--coarse-z-range-mm 240 --coarse-z-step-mm 30` 会产生17个候选位置。新增
`--coarse-z-samples` 后，可以直接指定在完整 `[-range,+range]` 上均匀评估的候选数量；该参数非0
时优先于 `--coarse-z-step-mm`。对全部病例使用MOMENTS初始化和356点z搜索：

```bash
python -m tools.thorax_preprocessing.registration \
  --root dataset/thorax \
  --output dataset/thorax/registration/moments_z356 \
  --initializer moments \
  --coarse-z-range-mm 240 \
  --coarse-z-samples 356 \
  --target-spacing-mm 2 \
  --size-multiple 4
```

在±240 mm范围内，356点对应约1.352 mm间隔。建议先用 `--limit 2` 测试时间和结果；356点的
粗搜索评估次数约为原17点的20.9倍。粗搜索是有限网格枚举，不是梯度迭代，没有可靠的“收敛后
停止”条件；按扫描顺序提前停止可能漏掉尚未评估的更优位置，所以没有启用不安全的early stop。
后续刚性和仿射优化器本身已有收敛检测，会在满足条件时提前停止。

更密的z搜索只提高z初值分辨率，不能解决残余x/y错位、旋转、呼吸形变或掩膜/FOV差异。Dice是
身体轮廓质控指标，也不是粗搜索直接优化的目标；粗搜索仍使用跨模态更稳健的互信息。因此采样从
17增至356不保证每例Dice都上升。应保留旧输出，以 `batch_summary.json`、逐例指标和QA叠加图
比较后再选择训练GT，不能只按Dice单一排序。

### 批处理：`image` 下一个文件夹就是一个病人

`registration.py` 会遍历 `--root/image` 下的**每一个病人文件夹**，逐个独立完成配准：

1. 只在当前病人文件夹内按 `SeriesInstanceUID` 分组识别序列，因此不会把甲病人的 CT 和乙病人的
   CBCT 配到一起；扫描量也从整棵树降到单个病例的几百个文件，几秒即可完成；
2. 每完成一个阶段都打印进度（读取体数据、身体掩膜、轴向粗搜索、刚性、仿射、重采样写盘），
   长阶段每 20 秒报告一次当前 metric，因此不会出现"看着像卡死"的情况；
3. 单个病人出错（缺 CT、缺 CBCT、读取失败、配准异常）只记录该病人并继续处理后面的病人；
4. `--output` 现在是**父目录**，每个病人写入自己的子目录 `--output/<病人文件夹名>/`；
5. 结束时打印汇总，并明确列出：
   - 缺失数据的文件夹（无 DICOM、缺计划 CT 序列、缺 Varian CBCT 序列）；
   - 处理失败的文件夹及原因；
   - 已有输出被跳过的文件夹（未加 `--overwrite` 时不会重跑，也不会中断批处理）；
   - 配准质控不通过的文件夹及其触发项；
   - `projection` 有目录但 `image` 缺文件夹，以及 `image` 有文件夹但 `projection` 缺失。
6. 同一份汇总同时写入 `--output/batch_summary.json`，便于脚本筛选。

正式跑 184 个病人前，建议先用 `--limit 2` 只跑前两个病人验证参数与耗时：

```powershell
python -m tools.thorax_preprocessing.registration `
  --root dataset/thorax `
  --output dataset/thorax/registration/current `
  --target-spacing-mm 2 --size-multiple 4 `
  --limit 2
```

`--ct-series-uid` / `--cbct-series-uid` 只在需要强制指定单个病人的序列时使用；批处理下会对每个
病人套用同一 UID，通常不是你想要的，因此程序会打印警告。

每个病人目录中的输出包括：

- `registered_ct_hu.nii.gz`：CBCT 网格上的配准 CT；
- `registered_ct_mu.nii.gz`：可用于训练的非负线性衰减系数；
- `resample_cbct_to_ct.tfm`：SimpleITK 重采样所需的 fixed→moving 反向映射，可直接传给
  `sitk.Resample(CT, CBCT, transform, ...)`；
- `ct_to_cbct.tfm`：CT 物理点到 CBCT 物理点的正向变换；
- `registration_metrics.json`：粗搜索、优化器、Dice、NMI、相关系数、变换矩阵和自动质控结论；
- `registration_qa.png`：轴位、冠状位、矢状位叠加质控图。

默认只做刚性＋仿射，不会改变局部病灶形态。保存结果时，以原始 CBCT 的物理中心和完整 FOV
建立 2 mm 等方网格，并将每一维向上扩展到4的倍数。当前病例由原生
512×512×118、0.96165×0.96165×1.99621 mm 统一为
248×248×120、2×2×2 mm。批处理时应根据质控阈值筛出少量失败病例人工复核，
不能仅凭优化器收敛就认定配准正确。测试参数或只需要刚性变换时可加 `--rigid-only`。

配准质控通过后，直接把重采样后的 CT 选为训练 GT。注册结果每个病人一层子文件夹，
`prepare_thorax.py` 会按病人自动定位：

```powershell
python -m tools.thorax_preprocessing.prepare_thorax `
  --root dataset/thorax `
  --output dataset/thorax/syn_data `
  --gt-source registered-ct `
  --registered-ct-root dataset/thorax/registration/current `
  --projection-bin 1 `
  --projection-resolution 256
```

> ✅ `prepare_thorax.py` 已改为与 `registration.py` 一致的**按病人批处理**入口：每个病人都用
> 自己的 `image/<case>` DICOM 和 `registration/current/<case>/registered_ct_mu.nii.gz`，
> 单个病人失败或数据缺失只记录进 `batch_summary.json` 并继续。转换后在
> `dataset/thorax/syn_data/batch_summary.json` 查看成功/跳过/失败清单。

准备脚本会验证注册 CT 与 CBCT 的方向及物理中心兼容，并把 CBCT 重采样到注册 CT 的标准训练
网格，避免把未重采样的 CT 错当作投影监督标签。
病例目录中的 `ct_volume.nii.gz` 只是保留用于排查的原始 spacing 中心裁剪 pCT；训练实际读取的
是 `gt_volume.nii.gz`，选择 `registered-ct` 时两者不能混用。

### 为什么必须统一 spacing 和尺寸

原始数据并不一致：

| 数据 | size (X,Y,Z) | spacing mm (X,Y,Z) | 轴向覆盖 |
|---|---|---|---|
| Varian CBCT | 512×512×118 | 0.96165×0.96165×1.99621 | 约235.6 mm |
| 计划 pCT | 512×512×220 | 0.97656×0.97656×3.0 | 约660 mm |

SimpleITK 配准在物理坐标中计算，因此配准优化前不要求两者 spacing 相等。配准结束后用于监督的
pCT 必须重采样到统一训练网格。模型的 decoder 将低分辨率特征放大 `train_scale=4`；原生 z=118
会先得到30个采样点，decoder 输出120层，与118层 GT 无法计算损失。因此所有输出尺寸都必须是4
的倍数。

本流程采用：

```text
完整 pCT（512×512×220，0.9766×0.9766×3 mm）
    → 在原始物理空间进行刚性＋仿射配准
    → 以 CBCT 中心/FOV 建立 248×248×120、2 mm 等方网格
    → 线性插值 pCT HU
    → 空间上自动截取到 CBCT 覆盖范围，网格外填充 -1000 HU
    → HU 截断到 [-1000, 3000]
    → HU 转换为 μ，范围 [0, 0.088]
```

这里有两种不同的“截断”：

1. **空间截取**：完整 pCT 只保留 CBCT FOV 对应的胸部范围；通过配准变换和参考网格完成，不能
   简单按固定 DICOM 序号切片。
2. **强度截断**：HU 限制到 `[-1000,3000]`，抑制空气以下值、金属和异常重建值。训练器中
   thorax 的 μ 范围进一步配置为 `[0,0.09]`。

连续灰度体（CT、CBCT、投影）使用线性插值；二值身体掩膜使用最近邻插值。pCT 先在 HU 域
插值，再做 HU→μ，避免插值造成非物理的标签值。不要把 spacing 仅修改在 NIfTI header 中而不
重采样像素，那会破坏真实几何。

## 3. 校验

```powershell
python -m tools.thorax_preprocessing.validate_thorax `
  --data dataset/thorax/syn_data
```

校验器检查三个训练必需文件、NIfTI 维度、帧数、JSON resolution、NaN/Inf、负值、体尺寸是否
可被4整除、体素/探测器 spacing 是否各向同性、几何向量长度以及 CBCT/GT 网格是否一致。

## 4. 建立训练划分并训练

转换会在输出根目录生成一个含 `train/val/test/visual` 的 `thorax_split.json` 草稿，它只是把
本次成功转换的病例按名称排序后切分（test/val 各约占 10%），**不含任何患者级去泄漏逻辑，也
不保证覆盖全部病例**。若 `data/dataset_split/thorax_split.json` 已有经过确认的划分，请
**不要**用草稿覆盖它；正确做法是对照 `batch_summary.json` 的 `skipped_detail`，把本次被跳过的
病例从已有划分中移除，其余划分保持不变。只有在还没有划分时，才把草稿作为起点：

```powershell
Copy-Item dataset/thorax/syn_data/thorax_split.json data/dataset_split/thorax_split.json
```

然后运行：

```powershell
python train.py `
  -n=thorax_real `
  -D=./dataset/thorax/syn_data `
  --datatype=thorax `
  --require-gt-source registered-ct `
  --train_scale=4 `
  --fusion=ada `
  --start=0 --end=360 --nviews=20 `
  --angle_sampling=uniform --is_train
```

`train.py`（以及 `evaluate.py`）会先打印 3D 标签体数据的来源报告：`CBCTDataset` 固定读取
`<datadir>/<case>/gt_volume.nii.gz`，报告从各病例的 `transforms.json` 读出 `gt_source` 字段，
并列出来源分布与体积网格。加 `--require-gt-source registered-ct` 可把报告变成硬校验：只要有病例的
`gt_source` 不是 `registered-ct`（或缺少该字段），训练会在加载数据前直接报错，避免误把 CBCT 当标签。

> 病例目录里的三个体积文件含义不同，不要混用：
> - `gt_volume.nii.gz` —— **训练/评估实际读取的标签**，本数据集里等于配准后的 pCT
>   （与 `registration/current/<case>/registered_ct_mu.nii.gz` 逐字节相同）；
> - `ct_volume.nii.gz` —— 原生 spacing 的中心裁剪计划 CT，**未配准**，仅供排查；
> - `cbct_volume.nii.gz` —— CBCT（仅用于生成 `proj.nii.gz`）。

真实投影不支持 `angle_sampling=random`，因为该分支会从 GT 在线生成 DRR；训练和评估都应使用
`uniform`，实际位姿从 `transforms.json/frames[*].vec` 读取。

在 `uniform` 分支中，主模型的数据加载器直接用 SimpleITK 读取病例目录中的 `proj.nii.gz`，再按
`--start/--end/--nviews` 选择投影帧，并以相同索引读取 `transforms.json` 中的实测几何。它不会从
GT重新生成网络输入投影。训练器仍会对网络预测的三维体积执行可微前向投影，用于计算2D重投影
损失；这是监督损失的一部分，不是再次生成输入数据。

## 质量控制

转换完成后仍应至少人工检查三个方向的 CT/CBCT 切片、若干角度的 `proj.nii.gz`，并用几何做一次
投影/反投影可视化。尤其在把 CT 设为 GT 前，必须确认解剖位置、左右/头脚方向和等中心一致。

### `Zero-valued spacing` 排查

若SimpleITK报告 `Refusing to change spacing ... to [..., ..., 0]`，说明DICOM序列的相邻
`ImagePositionPatient` 中存在重复位置，或位置标签缺失，旧版读取逻辑因此算出了0 mm层间距。
当前读取器会按 `SOPInstanceUID` 和物理切片位置去重，只使用非零位置差估算spacing，并在位置
不可用时依次回退到 `SpacingBetweenSlices`、`SliceThickness`。可先运行检查命令确认CT和CBCT的
切片数及z-spacing均大于0：

```bash
python -m tools.thorax_preprocessing.inspect_thorax --root dataset/thorax
```
