# Launch scripts

The public launchers are:

- `run.sh`: resolve one SR experiment YAML and run its train/test stage;
- `launchers/flow.sh`: train or evaluate Flow on one dataset;
- `launchers/train_sr.sh`: convenience wrapper for one SR training run;
- `launchers/run_four_datasets.sh`: distribute four dataset jobs over the
  supplied GPU IDs.

SR training is intentionally launched one dataset and one loss at a time.
The eight training commands are:

```bash
# L1 models
bash train.sh dusr_realv2_paired_l1 0
bash train.sh camera_fusion_real_l1 0
bash train.sh dusr_real_l1 0
bash train.sh realmcvsr_real_l1 0

# GAN models
bash train.sh dusr_realv2_paired_gan 0
bash train.sh camera_fusion_real_gan 0
bash train.sh dusr_real_gan 0
bash train.sh realmcvsr_real_gan 0
```

The multi-dataset launcher is intended for checkpoint evaluation; SR training
is launched one dataset and one loss at a time.

Examples:

```bash
bash scripts/run.sh dusr_realv2_paired_l1 train 0 --dry-run
bash test.sh dusr_real_l1 1
bash scripts/launchers/flow.sh test CameraFusion-Real 2
bash scripts/launchers/run_four_datasets.sh sr test 0,1,2,3 l1
```

Every SR experiment has one YAML file under `configs/sr/`; Flow configurations
are under `configs/flow/`. Supported path variables are `DATASET_ROOT`,
`FLOW_WEIGHT_ROOT`, `TRAIN_OUTPUT_ROOT`, and `RESULT_ROOT`.
