# FinalVLA

FinalVLA is the V15 LIBERO vision-language-action model with matched CUDA Mamba. This repository contains the training and evaluation code for the run evaluated from 85k through 99k steps. The best checkpoint is step 95,000.

## Model and training

- Two DINOv3 camera views, BERT language features, R3M features, and robot state feed the policy.
- A 12-step history encoder uses matched CUDA Mamba. R3M belief and tacit features condition the flow-matching action head.
- Robot state is zero-padded into the action-head condition. The action chunk length is 12; the evaluated rollout executes 10 actions before replanning.
- The training objective is action flow matching. There is no KL constraint or auxiliary future-state prediction.
- The published run trained from scratch for 100,000 optimizer steps using four GPUs.

## Results

The official LIBERO evaluation used 500 episodes for each suite at each checkpoint. All results from 85k to 99k are in [85k_99k_all4.json](experiments/libero/results/85k_99k_all4.json).

| Suite | Step 95k successes | Success rate |
| --- | ---: | ---: |
| LIBERO-Spatial | 496 / 500 | 99.2% |
| LIBERO-Object | 500 / 500 | 100.0% |
| LIBERO-Goal | 491 / 500 | 98.2% |
| LIBERO-10 | 475 / 500 | 95.0% |
| **All four** | **1962 / 2000** | **98.10%** |

## LIBERO-PRO smoke evaluation

Using the official [LIBERO-PRO repository](https://github.com/Zxy-MLlab/LIBERO-PRO), the 95k checkpoint completed one episode on task 0 of `libero_object_with_mug` with success (1/1). This is a single-episode runtime check, not a benchmark success-rate estimate. The [result JSON](experiments/libero/results/libero_pro_object_mug_first_result.json) records the task and protocol. The run used a separate Python virtual environment, the official repository’s BDDL and initial-state files, and local OSMesa rendering. PyTorch 2.6 required `weights_only=False` for the official trusted `.pruned_init` files in that local evaluation checkout.

## Checkpoint

Download both assets from the [v15-95k release](https://github.com/xzhuzhu/finalvla/releases/tag/v15-95k), then reconstruct and verify:

```bash
cat finalvla_95k.pth.part-00 finalvla_95k.pth.part-01 > finalvla_95k.pth
printf '%s  %s\n' '63383e42566ff5f3eef88ef5025b343797ef79d7a5325c3cb6e098e6b0df8710' 'finalvla_95k.pth' | sha256sum -c -
```

The full training checkpoint includes the model and optimizer state. It was checked against this source tree with zero missing or unexpected model keys. The two release assets are split because one checkpoint exceeds GitHub's per-asset size limit.

## Setup

Use Python 3.10+ with a CUDA-enabled PyTorch installation. Install this package and its LIBERO dependencies:

```bash
pip install -e '.[libero]'
```

Provide local copies of DINOv3 ViT-B/16, `bert-base-uncased`, and the R3M ResNet-18 backbone. The default paths are `pretrained/dinov3-vitb16`, `pretrained/bert-base-uncased`, and `pretrained/r3m-resnet18/backbone.pth`. These pretrained files and the LIBERO RLDS datasets are not bundled in Git.

## Train

Set `DATASET_DIRS` to the four comma-separated LIBERO RLDS directories. The launcher defaults to GPUs 0,1,2,3; set `TRAIN_GPUS` to change them. `PYTHON_BIN` can point to the desired Python executable.

```bash
DATASET_DIRS='/path/to/libero_spatial,/path/to/libero_object,/path/to/libero_goal,/path/to/libero_10' \
  bash scripts/libero/train_100k.sh
```

Check `finalvla-train --help` for available runtime arguments. The training launcher fixes the architecture and optimizer recipe used for the published run.

## Evaluate

The parallel evaluator supports a single checkpoint and accepts GPU IDs as a comma-separated list. For four suites, run one suite at a time with the same protocol used above:

```bash
python scripts/libero/evaluate_official_parallel.py \
  --ckpt finalvla_95k.pth --gpus 0 \
  --task-suite-name libero_10 --num-trials-per-task 50 \
  --action-head flow_matching --flow-state-dim 8 \
  --num-open-loop-steps 10
```

For the reported 500 episodes per suite, set `--num-trials-per-task 50` (10 tasks per suite) and repeat for `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`. The evaluator's other flow settings default to this checkpoint's configuration. Ensure the LIBERO simulator and EGL/CUDA environment are configured for the host.

## License

Apache-2.0. See [LICENSE](LICENSE).
