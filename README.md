# DuSRFlow

## Code terminology

The public code follows the terminology used in the paper:

- `DuSRFlow`: the complete dual-lens super-resolution model;
- `DuFlowNet`: the dedicated dual-lens flow estimator;
- `KernelFreeMatching`: KF matching between LR and LR-center;
- `PatchWarping`: index-based reference patch transfer;
- `KFDCNWarping`: kernel-free-guided deformable warping.

Legacy checkpoint prefixes are translated automatically when loading.
Components without an explicit method name retain their implementation names.

Official PyTorch implementation of **DuSRFlow: Dual-Lens Super-Resolution via
Parallax-Aware Flow Estimation and Complementary Reference Warping**.

This repository contains the released training and evaluation pipelines for:

- **DuFlowNet**, the parallax-aware optical-flow module;
- **DuSRFlow**, the dual-lens super-resolution model.

Comparison methods and ablation-only variants are not part of this release.

## Acknowledgements

The CUDA deformable-convolution operator is derived from open-source DCNv2
implementations. Its original license is preserved in
`dusrflow/models/archs/dcn/LICENSE`. Parts of the restoration utilities follow
the organization of [BasicSR](https://github.com/XPixelGroup/BasicSR).

## Project structure

```text
.
├── configs/
│   ├── flow/                 # Dataset-specific DuFlowNet configurations
│   └── sr/                   # Dataset/loss-specific DuSRFlow experiments
├── coordinates/             # Evaluation regions for external datasets
├── dusrflow/
│   ├── data/                 # SR and Flow datasets
│   ├── models/
│   │   ├── archs/            # DuSRFlow, DuFlowNet, and CUDA DCN operators
│   │   ├── flow.py           # DuFlowNet training/evaluation wrapper
│   │   └── sr_trainer.py     # DuSRFlow optimization and validation
│   ├── metrics.py
│   └── utils.py
├── scripts/
│   ├── launchers/            # Dataset and multi-GPU launch helpers
│   └── run.sh                # Run one SR experiment YAML
├── tools/
│   ├── train_flow.py / test_flow.py
│   ├── train_sr.py / test_sr.py
│   ├── profile_model.py
│   └── run_experiment.py
├── weights/                  # Released checkpoints (distributed separately)
├── run_all.sh                # Optional multi-GPU checkpoint evaluation
└── train.sh                  # Train one SR dataset/loss configuration
```

The Flow and SR pipelines deliberately use the same organization: dataset code
under `dusrflow/data`, model code under `dusrflow/models`, experiment settings
under `configs`, and executable entry points under `tools`.

## Environment

- Python 3.10
- PyTorch with CUDA support
- A CUDA toolkit compatible with the installed PyTorch build
- NVIDIA GPU

Install dependencies:

```bash
pip install -r requirements.txt
```

Run commands from the repository root. Set a custom Python executable when
needed:

```bash
export PYTHON_BIN=/path/to/python
```

Create the local path configuration once after cloning:

```bash
cp configs/paths.example.yaml configs/paths.yaml
```

Then update `configs/paths.yaml` for the local Python environment, datasets,
and Flow foreground patches. This machine-specific file is not tracked by Git.

## Dataset layout

Set `DATASET_ROOT` to the directory containing the four datasets:

```bash
export DATASET_ROOT=/path/to/dual_lens_datasets
```

Source files and YAML configurations do not need to be edited. Dataset and
checkpoint paths can also be supplied directly to `train.sh` and `test.sh`.

Expected layout:

```text
dual_lens_datasets/
├── DuSR-RealV2/
│   └── Paired/
│       ├── train/{LR,HR_CC_curve,HR_SIFT_CC_curve}
│       └── test/{LR,LR_center,HR_CC_curve,ref}
├── CameraFusion-Real/
│   ├── train/{LR,LR_center,Ref_full,HR,Ref_SIFT}
│   └── test/{LR,LR_center,HR,Ref_SIFT}
├── DuSR-Real/
│   └── ...
└── RealMCVSR-Real/
    └── ...
```

## Checkpoints

Only the pretrained checkpoints used to report the paper results are released.
Checkpoints produced by the independent retraining verification are not part
of this repository or its model release.

Place the released checkpoints as follows:

```text
weights/pretrained/
├── DuFlowNet_DuSR-RealV2-Paired.pth
├── DuFlowNet_CameraFusion-Real.pth
├── DuFlowNet_DuSR-Real.pth
├── DuFlowNet_RealMCVSR-Real.pth
├── DuSRFlow_DuSR-RealV2-Paired_L1.pth
├── DuSRFlow_DuSR-RealV2-Paired_GAN.pth
├── DuSRFlow_CameraFusion-Real_L1.pth
├── DuSRFlow_CameraFusion-Real_GAN.pth
├── DuSRFlow_DuSR-Real_L1.pth
├── DuSRFlow_DuSR-Real_GAN.pth
├── DuSRFlow_RealMCVSR-Real_L1.pth
└── DuSRFlow_RealMCVSR-Real_GAN.pth
```

## Evaluate released models

Evaluate each Flow checkpoint independently:

```bash
bash scripts/launchers/flow.sh test DuSR-RealV2-Paired 0
bash scripts/launchers/flow.sh test CameraFusion-Real 0
bash scripts/launchers/flow.sh test DuSR-Real 0
bash scripts/launchers/flow.sh test RealMCVSR-Real 0
```

Evaluate each dataset-specific SR checkpoint independently:

```bash
# Reconstruction-only models
bash test.sh dusr_realv2_paired_l1 0
bash test.sh camera_fusion_real_l1 0
bash test.sh dusr_real_l1 0
bash test.sh realmcvsr_real_l1 0

# Hybrid reconstruction/perceptual/adversarial models
bash test.sh dusr_realv2_paired_gan 0
bash test.sh camera_fusion_real_gan 0
bash test.sh dusr_real_gan 0
bash test.sh realmcvsr_real_gan 0
```

To evaluate a checkpoint stored outside the default `weights/` directory:

```bash
bash test.sh dusr_realv2_paired_l1 0 \
    --dataset-root /path/to/dual_lens_datasets \
    --checkpoint /path/to/sr_checkpoint.pth
```

The DuSR-Real and RealMCVSR-Real hybrid-loss YAML files use
`model_variant: v2` with `patch_phase: native`. This preserves the original
patch-grid origins used to generate the paper PNGs. Changing the phase to
`even`, or using the V3 coordinate-aware overlap stitching, changes a subset of
images whose crop coordinates lie on a different modulo-4 phase.

To evaluate all four Flow checkpoints and all eight SR checkpoints on four
GPUs:

```bash
bash run_all.sh test-all 0,1,2,3
```

Verify one checkpoint group only:

```bash
bash run_all.sh test-flow 0,1,2,3
bash run_all.sh test-sr-l1 0,1,2,3
bash run_all.sh test-sr-gan 0,1,2,3
```

Flow results are written to `flow_results/`; SR results are written to
`results/`; multi-GPU launcher logs are written to `launcher_logs/`.

## Train DuFlowNet

Flow training uses real dataset backgrounds and synthetic RGBA foreground
objects. Each dataset is trained with an independent command and YAML:

```bash
bash scripts/launchers/flow.sh train DuSR-RealV2-Paired 0 /path/to/rgba_foregrounds
bash scripts/launchers/flow.sh train CameraFusion-Real 1 /path/to/rgba_foregrounds
bash scripts/launchers/flow.sh train DuSR-Real 2 /path/to/rgba_foregrounds
bash scripts/launchers/flow.sh train RealMCVSR-Real 3 /path/to/rgba_foregrounds
```

The released settings use 448 x 448 patches, batch size 4, 400k iterations,
Charbonnier photometric loss, and edge-aware second-order smoothness loss.
Settings are recorded independently in `configs/flow/<dataset>.yaml`.

## Train DuSRFlow

The release exposes one command per dataset and loss. These are the eight SR
training commands:

```bash
# Reconstruction-only training
bash train.sh dusr_realv2_paired_l1 0
bash train.sh camera_fusion_real_l1 0
bash train.sh dusr_real_l1 0
bash train.sh realmcvsr_real_l1 0

# Hybrid reconstruction/perceptual/adversarial training
bash train.sh dusr_realv2_paired_gan 0
bash train.sh camera_fusion_real_gan 0
bash train.sh dusr_real_gan 0
bash train.sh realmcvsr_real_gan 0
```

The paired `DuSR-RealV2-Paired` setting uses 128 x 128 patches and displacement
`16..48 x2`. The other three datasets use 256 x 256 patches and displacement
`32..96 x2`. Each exact training and inference configuration is stored in one
file under `configs/sr/`.

To initialize SR training with another set of Flow checkpoints:

```bash
bash train.sh dusr_realv2_paired_l1 0 \
    --dataset-root /path/to/dual_lens_datasets \
    --flow-checkpoint /path/to/flow_checkpoint.pth
```

When the released checkpoints remain under `weights/`, only `DATASET_ROOT`
needs to be set. Standalone DuFlowNet training additionally requires the RGBA
foreground-patch directory described above.

## Configuration checks

Inspect the resolved command without running it:

```bash
bash scripts/run.sh dusr_realv2_paired_l1 train 0 --dry-run
```

Validate all SR input paths without loading CUDA operators or starting
training:

```bash
bash scripts/run.sh dusr_realv2_paired_l1 train 0 --check-only
```

Validate one Flow setup:

```bash
bash scripts/launchers/flow.sh test DuSR-RealV2-Paired 0 --check-only
```

## Complexity profiling

```bash
python -m tools.profile_model --all
python -m tools.profile_model --params
python -m tools.profile_model --memory
python -m tools.profile_model --timing --warmup 10 --repeat 50
```

The default profiling inputs match the paper latency setting: LR and Ref are
`896 x 448`, while LRC is `448 x 224`.
