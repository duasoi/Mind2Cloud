<div align="center">

# Mind2Cloud: EEG-to-Point Cloud Generation with Two-Granularity Diffusion Decoding

**An EEG-based 3D point cloud generation and reconstruction project**

[![Repository](https://img.shields.io/badge/GitHub-Mind2Cloud-181717?logo=github&logoColor=white)](https://github.com/duasoi/Mind2Cloud)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![GPU](https://img.shields.io/badge/Verified-Tesla_V100-76B900?logo=nvidia&logoColor=white)

[**Paper**](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/14635.pdf) ·
[**Code**](https://github.com/duasoi/Mind2Cloud) ·
[**Dataset**](#dataset) ·
[**Training**](#quick-start) ·
[**Inference**](#inference)

</div>

> This README is intended for first-time users. Follow it from top to bottom to configure the environment, prepare the data, train the model, monitor logs, run inference, and save results.

## Overview

Mind2Cloud generates 3D point clouds from EEG signals. The main entry points in the current runnable version are:

```text
Training:   PointCNN/Train.py
Inference:  PointCNN/Test.py
```

The current model uses EEG inputs at two temporal granularities:

- `6s_100Hz`：a dynamic EEG response captured over a longer time window.
- `1s_250Hz`：a static/local EEG response captured over a shorter time window.

The EEG encoder produces a `1024`-dimensional conditioning feature. It is concatenated with noisy 3D point coordinates and passed to the diffusion decoder:

```text
3 point coordinates + 1024 EEG feature = 1027 input channels
```

Therefore, both training and inference in the current version require:

```bash
--in_channels 1027
```

## Project Links

| Resource | Link |
| :--- | :--- |
| Paper | [PDF](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/14635.pdf) |
| Code repository | [https://github.com/duasoi/Mind2Cloud](https://github.com/duasoi/Mind2Cloud) |
| Dataset | [Baidu Netdisk](https://pan.baidu.com/s/1_DfNgEw7cPMrW7ZKh3xxoQ?pwd=oa51) (access code: `oa51`) |
| Pretrained checkpoints | **TODO: checkpoint link** |
| Project page / Demo | **TODO: project page link** |

## Path Convention

This document does not use absolute server paths. Run all commands from the project root unless stated otherwise.

We recommend placing the code and dataset directories at the same level, for example:

```text
workspace/
|-- neuro_3D_clear/
`-- neuro-3D-main_V100/
    `-- EEG_3Datasets/
```

Enter the project root:

```bash
cd neuro_3D_clear
```

If your directory has a different name, replace `neuro_3D_clear` with your project directory name.

The remaining commands use the following three relative-path variables:

```bash
DATA_PATH="../neuro-3D-main_V100/EEG_3Datasets"
RESULT_DIR="./neuro3d_result"
LOG_DIR="./logs"
```

If the dataset is stored elsewhere, change only `DATA_PATH` to the appropriate relative path.

## Repository Layout

Recommended project layout:

```text
neuro_3D_clear/
|-- PointCNN/
|   |-- Train.py
|   |-- Test.py
|   `-- model_PCNN/
|-- PointGeneration/
|-- eeg_data_process/
|-- train_utils/
|-- cls/
|-- chamfer3D/
|-- PyTorchEMD/
|-- requirements.txt
|-- logs/
`-- neuro3d_result/
```

The dataset does not need to be inside the code directory. We recommend referencing it with a relative path:

```bash
../neuro-3D-main_V100/EEG_3Datasets/
```

## Dataset

The preprocessed dataset is available from [Baidu Netdisk](https://pan.baidu.com/s/1_DfNgEw7cPMrW7ZKh3xxoQ?pwd=oa51); access code: `oa51`.

Expected dataset layout:

```text
EEG_3Datasets/
|-- EEGdata/
|   |-- sub01/
|   |   |-- sub01_train_data_6s_100Hz.npy
|   |   |-- sub01_train_data_1s_250Hz.npy
|   |   |-- sub01_test_data_6s_100Hz.npy
|   |   `-- sub01_test_data_1s_250Hz.npy
|   |-- sub02/
|   |-- sub03/
|   `-- ...
|-- point_cloud_simple/
|-- video_new/
|-- color_label.xlsx
|-- clip_feature.pth
`-- clip_feature_gray.pth
```

Check that the data exists before training:

```bash
cd neuro_3D_clear

DATA_PATH="../neuro-3D-main_V100/EEG_3Datasets"
ls "$DATA_PATH"
ls "$DATA_PATH/EEGdata/sub02"
```

Training will fail during data loading if `clip_feature_gray.pth` or any required subject EEG file is missing.

## Environment

Python 3.9 is recommended:

```bash
conda create -n neuro3d python=3.9 -y
conda activate neuro3d
```

Install the PyTorch build that matches the server's CUDA environment. Skip this step if PyTorch and CUDA already work correctly.

Install the project dependencies:

```bash
cd neuro_3D_clear
pip install -r requirements.txt
```

For a V100 server, we recommend setting:

```bash
export TORCH_CUDA_ARCH_LIST="7.0"
```

This reduces unnecessary warnings while compiling CUDA extensions.

## Important Parameters

Common parameters:

```text
--device              Execution device; use cuda for GPU training
--sub                 Subject identifier, for example sub02
--data_path           Dataset root directory
--output_dir          Root directory for training outputs
--experiment_name     Name of the current experiment
--in_channels         Must be set to 1027 for the current model
--batch_size          Training batch size
--val_batch_size      Validation batch size
--max_steps           Maximum number of training steps
--checkpoint_freq     Checkpoint save interval
--log_step_freq       Logging and validation interval
--num_workers         Number of DataLoader workers
```

Training and inference have been verified on an NVIDIA Tesla V100. We recommend starting with the following settings:

```text
batch_size=4
val_batch_size=4
```

If sufficient GPU memory is available, you can also try `batch_size=8` or `batch_size=16`.

## Quick Start

The following command trains `sub02` and saves the results under `./neuro3d_result/`.

Training command:

```bash
cd neuro_3D_clear

DATA_PATH="../neuro-3D-main_V100/EEG_3Datasets"
RESULT_DIR="./neuro3d_result"
LOG_DIR="./logs"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

TORCH_CUDA_ARCH_LIST="7.0" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python -u PointCNN/Train.py \
  --device cuda \
  --task train \
  --sub sub02 \
  --data_path "$DATA_PATH"/ \
  --output_dir "$RESULT_DIR" \
  --experiment_name double_10w_2048_dp8_EnhanceGD_only_mse \
  --in_channels 1027 \
  --max_steps 100000 \
  --checkpoint_freq 5000 \
  --log_step_freq 10 \
  --batch_size 4 \
  --val_batch_size 4 \
  --num_workers 4 \
  > "$LOG_DIR/sub02_shape.log" 2>&1 &
```

Training results are saved to:

```bash
./neuro3d_result/sub02/double_10w_2048_dp8_EnhanceGD_only_mse/
```

## Monitor Training

Follow the training log in real time:

```bash
tail -f ./logs/sub02_shape.log
```

Check whether the training process is still running:

```bash
ps -ef | grep "PointCNN/Train.py" | grep -v grep
```

Check GPU utilization:

```bash
nvidia-smi
```

Check whether checkpoints have been saved:

```bash
ls -lh ./neuro3d_result/sub02/double_10w_2048_dp8_EnhanceGD_only_mse/
```

A successful run creates files such as:

```text
checkpoint-5000.pth
checkpoint-10000.pth
log.txt
tensorboard_logs/
```

## Train Another Subject

When training `sub03`, use a separate log file to keep it distinct from the `sub02` log:

```bash
cd neuro_3D_clear

DATA_PATH="../neuro-3D-main_V100/EEG_3Datasets"
RESULT_DIR="./neuro3d_result"
LOG_DIR="./logs"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

TORCH_CUDA_ARCH_LIST="7.0" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python -u PointCNN/Train.py \
  --device cuda \
  --task train \
  --sub sub03 \
  --data_path "$DATA_PATH"/ \
  --output_dir "$RESULT_DIR" \
  --experiment_name double_10w_2048_dp8_EnhanceGD_only_mse \
  --in_channels 1027 \
  --max_steps 100000 \
  --checkpoint_freq 5000 \
  --log_step_freq 10 \
  --batch_size 4 \
  --val_batch_size 4 \
  --num_workers 4 \
  > "$LOG_DIR/sub03_shape.log" 2>&1 &
```

Follow the `sub03` log:

```bash
tail -f ./logs/sub03_shape.log
```

## Inference

After training, use `PointCNN/Test.py` to generate point clouds from the test-set EEG signals.

Example for `sub02`:

```bash
cd neuro_3D_clear

DATA_PATH="../neuro-3D-main_V100/EEG_3Datasets"
RESULT_DIR="./neuro3d_result"
OUTPUT_DIR="./PointCNN/outputs"
EXPERIMENT_NAME="double_10w_2048_dp8_EnhanceGD_only_mse"
CHECKPOINT_PATH="$RESULT_DIR/sub02/$EXPERIMENT_NAME/checkpoint-100000.pth"

mkdir -p "$OUTPUT_DIR"

TORCH_CUDA_ARCH_LIST="7.0" python -u PointCNN/Test.py \
  --device cuda \
  --sub sub02 \
  --data_path "$DATA_PATH"/ \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --infer_output_dir "$OUTPUT_DIR" \
  --experiment_name "$EXPERIMENT_NAME" \
  --in_channels 1027 \
  --test_batch_size 4 \
  --num_points 2048 \
  --infer_repeat 5
```

Generated results are saved to:

```bash
./PointCNN/outputs/sub02/double_10w_2048_dp8_EnhanceGD_only_mse/
```

Outputs use the `.ply` format and can be viewed with Open3D, MeshLab, CloudCompare, or similar tools.

## TensorBoard

TensorBoard logs are saved automatically during training:

```bash
./neuro3d_result/sub02/double_10w_2048_dp8_EnhanceGD_only_mse/tensorboard_logs/
```

Start TensorBoard:

```bash
tensorboard --logdir ./neuro3d_result --port 6006
```

Open the following address in a browser:

```text
http://<server-ip>:6006
```

## Common Problems

### 1. `unrecognized arguments: --output_dir`

The server is not running the latest code. Check first:

```bash
python PointCNN/Train.py --help | grep -E "output_dir|experiment_name|val_batch_size"
```

If the command returns no matching options, upload the latest versions of:

```text
train_utils/parse.py
PointCNN/Train.py
PointCNN/Test.py
```

### 2. `No module named 'eeg_data_process.extract_eeg_feature_double'`

The compatibility module is missing. Run the following command on the server:

```bash
cat > eeg_data_process/extract_eeg_feature_double.py <<'PY'
from eeg_data_process.extract_eeg_feature import VideoImageEEGClassifyColor3 as _BaseVideoImageEEGClassifyColor3

class VideoImageEEGClassifyColor3(_BaseVideoImageEEGClassifyColor3):
    def forward(self, x, x2):
        clip_out, _, _, _ = super().forward(x, x2)
        return clip_out
PY
```

### 3. `get_optimizer() missing 1 required positional argument: 'accelerator'`

`Train.py` and `training_utils.py` are from incompatible versions. Change the function definition in `train_utils/training_utils.py` to:

```python
def get_optimizer(args, model: torch.nn.Module, accelerator: Optional[Accelerator] = None) -> torch.optim.Optimizer:
```

### 4. `expected input ... to have 131 channels, but got 1027`

`--in_channels` is configured incorrectly. The current model requires:

```bash
--in_channels 1027
```

The current diffusion decoder receives:

```text
3 point coordinates + 1024 EEG feature = 1027
```

### 5. No GPU process is visible

Inspect the log first:

```bash
tail -n 150 ./logs/sub02_shape.log
```

Then inspect the process:

```bash
ps -ef | grep "PointCNN/Train.py" | grep -v grep
```

If the log contains a `Traceback`, the process has already exited and the error must be resolved first.

### 6. CUDA out of memory

Reduce the batch size:

```bash
--batch_size 1 --val_batch_size 1
```

Once the run succeeds, increase it gradually:

```text
2 -> 4 -> 8 -> 16
```

### 7. Log files are mixed together

Use a separate log file for each subject and GPU:

```text
./logs/sub02_shape.log
./logs/sub03_shape.log
```

Do not write multiple experiments to the same `.log` file.

## Notes For Beginners

- `nohup ... &` runs the process in the background so it continues after the terminal disconnects.
- `python -u` flushes output to the `.log` file promptly.
- `tail -f xxx.log` follows the log in real time.
- `checkpoint_freq=5000` saves a checkpoint every 5,000 steps.
- `log_step_freq=10` prints training and validation logs every 10 steps.

## Citation

Replace the placeholder BibTeX below with the official citation when the paper is available:

```bibtex
@article{TODO,
  title   = {Mind2Cloud: EEG-to-Point Cloud Generation with Two-Granularity Diffusion Decoding},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

## Acknowledgements

This project builds on work in EEG-based 3D reconstruction, point cloud diffusion, and neural decoding. Add the relevant repositories, datasets, and papers here before the official release:

```text
Neuro-3D:
TODO

Related repositories:
TODO

Related papers:
TODO

Dataset source:
TODO
```

