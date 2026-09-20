# Geometry-Aware Attenuation Learning for Sparse-View CBCT Reconstruction
[Zhentao Liu](https://zhentao-liu.github.io/), [Yu Fang](https://yuffish.github.io/), [Changjian Li](https://enigma-li.github.io/), [Han Wu](http://hanwu.website/), [Yuan Liu](https://liuyuan-pal.github.io/), [Dinggang Shen](https://idea.bme.shanghaitech.edu.cn/), and [Zhiming Cui](https://shanghaitech-impact.github.io/)

## [Paper](https://ieeexplore.ieee.org/document/10705334) | [Arxiv](https://arxiv.org/abs/2303.14739) | [Dataset](https://huggingface.co/datasets/Zhentao-Liu/TMI2024_SVCT_dataset) | [Project Page](https://shanghaitech-impact.github.io/Geometry-Aware-Attenuation-Learning-for-Sparse-View-CBCT-Reconstruction/)

This is the official repo of our paper **Geometry-Aware Attenuation Learning for Sparse-View CBCT Reconstruction** in **IEEE TMI 2024**. In this work, we describe a novel encoder-decoder framework for sparse-view CBCT reconstruction which integrates the inherent geometry of CBCT scanning system. It produces high quality CBCT reconstructions with sparse input (20 views or less) in a time-efficient manner, which aims to reduce radiation exposure.

![](./image/CBCT_recon_TMI.png)

## Updated Feature
- **[2024-10-24]** Debugging. We have provided a `if_intersect` function in `models/render.py` that decides whether X-rays intersect with bbx. No more bugs for no intersection in `ray_AABB` function.
- **[2024-10-21]** We have provided a new `angle2vec` function in `models/render.py` that incorporates both `PrimaryAngle` and `SecondaryAngle`, which are commonly used in real-world CBCT scanning system. You may refer to [DICOM Geometry](https://dicom.innolitics.com/ciods/x-ray-angiographic-image/xa-positioner/00181510) for more details about these two angles. And in our paper (DRR simulation for simulated datasets), we only consider about `PrimaryAngle` (rotation angle in our paper), assuming `SecondaryAngle` is set to zero by default. `SecondaryAngle` could also be applied for [Computed Laminography](https://iopscience.iop.org/article/10.1088/1361-6501/aafcae) (CL) imaging as discussed in issue [#2](https://github.com/ShanghaiTech-IMPACT/Geometry-Aware-Attenuation-Learning-for-Sparse-View-CBCT-Reconstruction/issues/2), just setting `SecondaryAngle` as the oblique alpha angle.

## Setup
First clone this repo. And then set up an environment and install packages. We use single A100 80G GPU card for training. Make sure you have enough resources.

    git clone https://github.com/ShanghaiTech-IMPACT/Geometry-Aware-Attenuation-Learning-for-Sparse-View-CBCT-Reconstruction.git
    cd Geometry-Aware-Attenuation-Learning-for-Sparse-View-CBCT-Reconstruction
    conda create -n CBCTrecon python=3.8
    conda activate CBCTrecon
    pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt

## Dataset-Preparation

### Dental Dataset (Simulated)
We provide the preprocessed dental CBCT volumes in the dataset link. 130 cases in total, including 100 cases for training, 10 cases for validation, and 20 cases for testing. You may download them, and then put them in a self-built folder `./dataset/dental/raw_volume`. As for X-ray simulation, please refer to [DRR-Simulation](#DRR-Simulation).

### Spine Dataset (Simulated)
As for the spinal dataset, please refer to [CTSpine1K](https://github.com/MIRACLE-Center/CTSpine1K) for more details. We provide the preprocessed spine CT volumes in the dataset link. 130 cases in total, including 100 cases for training, 10 cases for validation, and 20 cases for testing. You may download them, and then put them in a self-built folder `./dataset/spine/raw_volume`. As for X-ray simulation, please refer to [DRR-Simulation](#DRR-Simulation).
### Walnut Dataset (Real-World)
As for the walnut dataset, please refer to [WalnutScan](https://github.com/cicwi/WalnutReconstructionCodes) for more details. It is a large-scale real-world walnut CBCT scans dataset collected for machine learning purpose. Many thanks to this great work. We provide the preprocessed walnut CBCT volumes, real-world projections, and geometry description files in the dataset link. 42 cases in total, including 32 cases for training, 5 cases for validation, and 5 cases for testing. You may download them, and then put them in a self-built folder `./dataset/walnut`.

The dataset split is set as default in `./data/dataset_split`. All datasets have been uploaded to [Hugging Face](https://huggingface.co/datasets/Zhentao-Liu/TMI2024_SVCT_dataset).

## DRR-Simulation

![](./image/DRR.png)

In our experiments, we apply Digitally Reconstructed Radiography (DRR) technique to simulate 2D X-ray projections of given 3D CBCT/CT volumes from dental/spine dataset. You need to first prepare your datasets as instructed in [Dataset-Preparation](#Dataset-Preparation). Then, run the following command.

    # for dental dataset
    python DRR_simulation.py --start=0 --end=360 --num=360 --sad=500 --sid=700 --datapath=./dataset/dental
    # for spine dataset
    python DRR_simulation.py --start=0 --end=360 --num=360 --sad=1000 --sid=1500 --datapath=./dataset/spine

In this way, you will get a data folder `./dataset/dental/syn_data` or `./dataset/spine/syn_data` that containing synthesized X-ray projections and geometry description files for each scanned object. It will generate 360 projections uniformly spaced within the angle range of [0, 360).

## Train
After preparing the dataset and X-ray simulation, you could run the following command to train your model.

    python train.py -n=<Expname> -D=./dataset/dental/syn_data --datatype=dental --train_scale=4 --fusion=ada --start=0 --end=360 --nviews=20 --angle_sampling=uniform --is_train 

In this way, you would train a model with 20 input views uniformly spaced within [0, 360) on dental dataset. The downsampling rate during training S=4, and it adopts adaptive feature fusing strategy proposed in our paper. Other hyperparameters are set as default. You may modify these hyperparamters to train your own model. The training process may take about 20 hours until convergence.

### Transfer the pCT-pretrained decoder into the full model

The decoder pretrainer under `submodel/decoder` saves both `decoder` and
`feature_stem` state dictionaries. During full-model training, `decoder` initializes
the reconstruction decoder, while the frozen `feature_stem` acts as a training-only
teacher:

```text
pCT/GT (ZYX) -> transpose to XYZ -> average pooling x4 -> frozen feature_stem -> z_prior
projections -> 2D encoder -> geometric backprojection -> view fusion             -> z_projection
                                                                               latent alignment
z_projection -> pretrained 3D decoder -> reconstructed volume
```

The teacher is not part of inference. `--pretrained_decoder` alone only initializes
the decoder from the prior and then runs ordinary joint training; the latent
alignment loss stays off in that mode.

Staged prior transfer — freeze the decoder, align the latents, then progressively
unfreeze — additionally requires `--use_adapter` (or `--use_prior_completion`) plus a
main-model `--pretrained_backbone` checkpoint, because the first stage freezes
Encoder/Aggregator. See `submodel/adapter/README.md` for the four-phase schedule and
`submodel/continuous_prior_completion/README.md` for the completion variant. When the
backbone is trained from scratch there is no backbone checkpoint to freeze, so Phase A
must be opened with `--phase_a_encoder_lr_factor` / `--phase_a_aggregator_lr_factor`.

Without `--use_adapter` (and without `--use_prior_completion`) `use_four_phase` is false,
so `_training_stage()` returns 0 for every epoch: **the whole schedule is inert**.  The
phase arguments are then ignored (no error), and `--latent_lambda` is silently disabled
because `_latent_weight()` returns 0 outside the phased schedule.  `--pretrained_decoder`
still initialises the decoder, and `--prior_encoder_type` must still match the checkpoint
(`deep` for a `submodel/deep_encoder` mean/detail prior) because the frozen teacher's
weights are loaded with `strict=True`.  In that mode every optimizer group shares the base
LR, so use `--stage0_decoder_lr_factor` to keep the pretrained decoder from being
fine-tuned at full LR: `0` freezes it, `0.1` gives it 0.1×.

Example for 100 total epochs:

```bash
python train.py \
  --name dental_prior_transfer \
  --datadir ./dataset/dental/syn_data \
  --datatype dental \
  --train_scale 4 \
  --fusion ada \
  --start 0 \
  --end 360 \
  --nviews 20 \
  --angle_sampling uniform \
  --is_train \
  --epochs 100 \
  --pretrained_decoder submodel/decoder/checkpoints/dental_batch3/ckpt_best_val.pt \
  --latent_lambda 0.1 \
  --latent_cosine_lambda 0.1 \
  --decoder_lr_factor 0.1 \
  --query_chunk_size 25000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01
```

The original full model currently supports object batch size 1; `--batch_size 3`
from decoder-only pretraining must not be reused here. It is separate from the number
of projection views selected by `--nviews`.

New arguments:

| Argument | Default | Meaning |
|---|---:|---|
| `--pretrained_decoder` | none | Decoder-pretraining checkpoint with `decoder` and `feature_stem` keys. It initializes the decoder and builds the frozen latent teacher. |
| `--require-gt-source` | none | Fail before training unless every case's `transforms.json` records this `gt_source` for the `gt_volume.nii.gz` label volume (thorax: pass `registered-ct` to guarantee the 3-D labels are the registered pCT). |
| `--prior_encoder_type` | `shallow` | Use `deep` for the mean/detail prior encoder written by `submodel/deep_encoder`; it must match the checkpoint. |
| `--latent_lambda` | `0.0` | Latent alignment loss weight in Phase A; it decays linearly across Phase B/C and is disabled in Phase D. |
| `--latent_cosine_lambda` | `0.1` | Weight of the cosine-distance term inside latent alignment. |
| `--adapter_lr_factor` | `1.0` | Adapter LR relative to the base LR; `0` freezes the adapter. |
| `--stage0_decoder_lr_factor` | none | Decoder LR multiplier for ordinary joint training (no `--use_adapter`/`--use_prior_completion`). Unset keeps the base LR for every group; `0` freezes the decoder. |
| `--phase_a_epochs` | `20` | Phase-A length. Phase A keeps Encoder/Aggregator frozen unless the two `--phase_a_*_lr_factor` options below are positive. |
| `--phase_a_encoder_lr_factor` | `0.0` | Phase-A encoder LR factor; set it > 0 when the encoder is trained from scratch. |
| `--phase_a_aggregator_lr_factor` | `0.0` | Phase-A aggregator LR factor; set it > 0 when the aggregator is trained from scratch. |
| `--phase_b_epochs` | `40` | Phase-B length; unfreezes `encoder.layer3/layer4` and the aggregator. |
| `--phase_c_epochs` | `80` | Phase-C length; unfreezes the backbone and progressively unfreezes the decoder. |
| `--decoder_lr_factor` | `0.1` | Decoder LR relative to the base encoder/aggregator LR. |
| `--phase_d_backbone_lr_factor` | `0.01` | Encoder/aggregator LR factor in Phase D. Set it to `0` to freeze them completely. |
| `--bone_lambda` | `0.0` | GT 骨区掩码 L1 权重；设为 `0` 关闭。骨区定义为 GT HU ≥ `--bone_lower_hu`。 |
| `--bone_lower_hu` | `300` | 骨区 GT 掩码的 HU 下限。 |
| `--soft_mask_lambda` | `0.0` | GT 软组织窗口掩码 L1 权重；设为 `0` 关闭。GT 位于给定窗口内才计入，但预测不会预先截断。 |
| `--soft_window_low` | `-160` | Lower HU boundary of the soft-tissue window. |
| `--soft_window_high` | `240` | Upper HU boundary of the soft-tissue window. |
| `--ssim_lambda` | `0.0` | 可微局部 3D `1-SSIM` 损失权重；设为 `0` 关闭。采用固定衰减系数范围归一化和 (3\times3\times3) 局部窗口。 |
| `--query_chunk_size` | `25000` | Number of 3D points processed by backprojection and multi-view fusion at once. Reduce to `12500` if a 32 GB GPU still runs out of memory. |
| `--disable_query_checkpoint` | off | Disable activation recomputation for backprojection/view fusion. This is faster but requires substantially more memory. |
| `--no_amp` | off | Disable CUDA mixed precision. Do not use this on a 32 GB GPU unless debugging numerical behavior. |

这三项专项损失都默认关闭，以保证旧 checkpoint 可直接加载。启用时，骨区与软组织掩码都只由
GT 生成，预测值不截断；因此即使预测跑到窗外，仍保留将其拉回 GT 的梯度。`--ssim_lambda`
使用可反传的 PyTorch 局部 SSIM，不是评估阶段的 `skimage` SSIM。

### TensorBoard for full-model training

Events are written to:

```text
train/logs/<experiment-name>/tensorboard/
```

Launch TensorBoard with:

```bash
tensorboard --logdir train/logs/dental_prior_transfer/tensorboard --port 6006
```

Step-level training scalars include total loss, every component, latent weight,
training stage, and the learning rate of each parameter group. Epoch-level `train`,
`val`, and `test` scalars include:

```text
G_loss                    weighted total
mse_loss_3d               attenuation-space voxel L1
gd1_loss                  first-order spatial-gradient L1
mse_loss_2d               projection-domain L1
latent_loss               weighted latent alignment
latent_smooth_l1_raw      unweighted normalized latent Smooth L1
latent_cosine_raw         unweighted latent cosine distance
bone_gt_mask_raw          unweighted GT bone-mask normalized L1
bone_gt_mask_loss         weighted GT bone-mask contribution
soft_mask_raw             unweighted GT soft-tissue-mask normalized L1
soft_mask_loss            weighted GT soft-tissue-mask contribution
ssim_loss_raw             unweighted differentiable local 1-SSIM
ssim_loss                 weighted 1-SSIM contribution
psnr_3d_clamp
ssim_3d_clamp             validation/test only
```

Validation and test projection losses use fixed uniformly spaced detector samples,
so their logged values are repeatable rather than changing with random ray indices.

### CUDA out-of-memory in multi-view fusion

The original implementation processed 100,000 queried 3D points at once. With 20
views and 256 feature channels, `adafusor` constructs a tensor whose effective input
is approximately `[20, 100000, 768]`, requiring about 5.72 GiB for that allocation
alone in FP32. The full training path now defaults to:

```text
query chunk size: 25,000
mixed precision: enabled on CUDA
query/fusion activation recomputation: enabled during training
```

For an RTX 5090 32 GB, start with the default `25000`. If memory still overflows,
retry with:

```bash
--query_chunk_size 12500
```

Do not add `--no_amp` or `--disable_query_checkpoint` in the memory-constrained run.
The allocator suggestion `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is only
useful when reserved-but-unused memory is large; it does not solve a genuine 5-6 GiB
working allocation when only about 4 GiB is free.

### Resume transfer training

Resume the latest full checkpoint with the same structural arguments and pretrained path:

```bash
python train.py \
  --name dental_prior_transfer \
  --datadir ./dataset/dental/syn_data \
  --datatype dental \
  --is_train \
  --resume \
  --epochs 100 \
  --pretrained_decoder submodel/decoder/checkpoints/dental_batch3/ckpt_best_val.pt \
  --latent_lambda 0.1 \
  --bone_lambda 0.05 \
  --soft_mask_lambda 0.01 \
  --ssim_lambda 0.01
```

`--resume` restores the full model, optimizer, scheduler, epoch, and global TensorBoard
step. It does not overwrite the resumed decoder with the standalone pretrained weights.

## Evaluate
Once the above training converged, you could run the following command to evaluate your model on test dataset.

    python evaluate.py -n=<Expname> -D=./dataset/dental/syn_data --datatype=dental --train_scale=4 --fusion=ada --start=0 --end=360 --nviews=20 --angle_sampling=uniform --eval_scale=4 --resume_name=200

In this way, it would test the model with 20 input views uniformly spaced within [0, 360) on dental dataset. The downsampling rate during evaluation S=4. Resumed from 200 epoch. You may modify these hyperparameters to evaluate your own model.

You can also take a quick verification with the pretrained weights. Just find them in the [Hugging Face](https://huggingface.co/datasets/Zhentao-Liu/TMI2024_SVCT_dataset).

For a transferred model, inference uses the normal full-model checkpoint and does not
load `feature_stem` or require pCT:

```bash
python evaluate.py \
  --name dental_prior_transfer \
  --datadir ./dataset/dental/syn_data \
  --datatype dental \
  --train_scale 4 \
  --eval_scale 4 \
  --fusion ada \
  --start 0 \
  --end 360 \
  --nviews 20 \
  --angle_sampling uniform \
  --resume_name 99
```

This loads `train/checkpoints/dental_prior_transfer/ckpt_history/ckpt_99`.

To report the same optional loss components used in training, append their
weights to `evaluate.py`.  A zero weight disables that component; evaluation
still always reports PSNR and the reporting SSIM:

```bash
  --bone_lambda 0.05 --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 --soft_window_low -160 --soft_window_high 240 \
  --ssim_lambda 0.01
```

`metric_batch.txt` contains each case; `logs_avg.txt` contains mean and standard
deviation for `bone_gt_mask_*`, `soft_mask_*`, and `ssim_loss_*` in addition to
PSNR/SSIM.

## Related Links
- Vector-based CBCT scanning geometry description (source, detector, uvector, vvector) is inspired by [WalnutScan](https://github.com/cicwi/WalnutReconstructionCodes) and [Astra-toolbox](https://github.com/astra-toolbox/astra-toolbox).
- Parts of our code are adapted from [PixelNeRF](https://github.com/sxyu/pixel-nerf) implementation.
- Pioneer NeRF-based framework for CBCT reconstruction: [NAF](https://github.com/Ruyi-Zha/naf_cbct), [SNAF](https://arxiv.org/abs/2211.17048).
- Check the concurrent work [DIF-Net](https://github.com/xmed-lab/DIF-Net) and its improvement [C2RV](https://github.com/xmed-lab/C2RV-CBCT) which also combine feature backprojection and generalization ability to solve sparse-view CBCT reconstruction as we do.
- It is recommended to observe medical data in nii format with [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php/) or [3D Slicer](https://www.slicer.org/).

Thanks to all these great works.

## Contact
There may be some errors during code cleaning. If you have any questions on our code or our paper, please feel free to contact with the author: liuzht2022@shanghaitech.edu.cn, or raise an issue in this repo. We shall continue to update this repo. TBC.

## Citation
If you find this work is useful for you, please cite our paper.

    @ARTICLE{SVCT,
          author={Liu, Zhentao and Fang, Yu and Li, Changjian and Wu, Han and Liu, Yuan and Shen, Dinggang and Cui, Zhiming},
          journal={IEEE Transactions on Medical Imaging}, 
          title={Geometry-Aware Attenuation Learning for Sparse-View CBCT Reconstruction}, 
          year={2024},
          doi={10.1109/TMI.2024.3473970}
    }
