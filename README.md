# DuSRFlow

PyTorch implementation of DuSRFlow for dual-lens image super-resolution.
This repository provides training and evaluation code for the DuFlowNet
alignment module and the complete DuSRFlow model.

## Environment

The code has been tested with Python 3.10 and CUDA-enabled PyTorch.

```bash
pip install -r requirements.txt
```

Create the local path configuration:

```bash
cp configs/paths.example.yaml configs/paths.yaml
```

Then update the following entries in `configs/paths.yaml`:

```yaml
python_bin: /path/to/python
dataset_root: /path/to/dual_lens_datasets
flow_foreground_root: /path/to/dual_lens_datasets/DuSR-RealV2/rgba_foregrounds
```

## Data and Pretrained Models

DuSR-RealV2 and the pretrained models are hosted on Hugging Face.

- DuSR-RealV2: [Hugging Face dataset](https://huggingface.co/datasets/zhangfangpu/DuSR-RealV2)
- CameraFusion-Real, DuSR-Real, and RealMCVSR-Real:
  [KeDuSR repository](https://github.com/ZifanCui/KeDuSR)
- Pretrained models: [Hugging Face model repository](https://huggingface.co/zhangfangpu/DuSRFlow)

Download DuSR-RealV2 into your configured dataset root:

```bash
hf download zhangfangpu/DuSR-RealV2 \
  --repo-type dataset \
  --local-dir /path/to/dual_lens_datasets/DuSR-RealV2
```

Download all official DuFlowNet and DuSRFlow checkpoints:

```bash
hf download zhangfangpu/DuSRFlow --local-dir weights/pretrained
```

Place the datasets under the configured `dataset_root`:

```text
dual_lens_datasets/
├── DuSR-RealV2/
│   ├── Paired/{train,test}/
│   ├── Unpaired/test/
│   └── rgba_foregrounds/
├── CameraFusion-Real/{train,test}/
├── DuSR-Real/{train,test}/
└── RealMCVSR-Real/{train,test}/
```

Place the downloaded checkpoints as follows:

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

## Test

Run all commands from the repository root. The final argument is the GPU ID.

Test the four DuFlowNet checkpoints:

```bash
bash test.sh flow_dusr_realv2_paired 0
bash test.sh flow_camera_fusion_real 0
bash test.sh flow_dusr_real 0
bash test.sh flow_realmcvsr_real 0
```

Test the reconstruction-oriented DuSRFlow checkpoints:

```bash
bash test.sh dusr_realv2_paired_l1 0
bash test.sh camera_fusion_real_l1 0
bash test.sh dusr_real_l1 0
bash test.sh realmcvsr_real_l1 0
```

Test the perceptual DuSRFlow checkpoints:

```bash
bash test.sh dusr_realv2_paired_gan 0
bash test.sh camera_fusion_real_gan 0
bash test.sh dusr_real_gan 0
bash test.sh realmcvsr_real_gan 0
```

Run inference on the unpaired split:

```bash
bash test.sh dusr_realv2_unpaired 0
```

Outputs are saved under `flow_results/` and `results/`.

## Training

Train DuFlowNet with real backgrounds and synthetic RGBA foreground objects:

```bash
bash scripts/launchers/flow.sh train DuSR-RealV2-Paired 0
bash scripts/launchers/flow.sh train CameraFusion-Real 0
bash scripts/launchers/flow.sh train DuSR-Real 0
bash scripts/launchers/flow.sh train RealMCVSR-Real 0
```

Train DuSRFlow with the reconstruction loss:

```bash
bash train.sh dusr_realv2_paired_l1 0
bash train.sh camera_fusion_real_l1 0
bash train.sh dusr_real_l1 0
bash train.sh realmcvsr_real_l1 0
```

Train DuSRFlow with the perceptual and adversarial losses:

```bash
bash train.sh dusr_realv2_paired_gan 0
bash train.sh camera_fusion_real_gan 0
bash train.sh dusr_real_gan 0
bash train.sh realmcvsr_real_gan 0
```

The dataset, loss, crop size, displacement range, Flow checkpoint, and
evaluation settings for each experiment are defined under `configs/`.

## Project Structure

```text
DuSRFlow/
├── configs/             # Dataset and experiment configurations
├── coordinates/         # Evaluation regions
├── dusrflow/
│   ├── data/            # Dataset implementations
│   └── models/          # DuFlowNet and DuSRFlow models
├── scripts/             # Experiment launchers
├── tools/               # Training, testing, and profiling entry points
├── weights/             # Pretrained checkpoints
├── train.sh
└── test.sh
```

## Acknowledgement

The deformable-convolution operator is based on open-source DCNv2
implementations. Its original license is included with the source code.
