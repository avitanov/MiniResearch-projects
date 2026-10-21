# Experiment Log

## Scope

This log contains only the GSM8K experiments from Block B (Tasks 4-7). It intentionally excludes the personal-project work.

All runs used `unsloth/qwen3-0.6B` with 4-bit loading, seed 3407, three epochs, learning rate `2e-4`, cosine scheduling, 5% warmup, `adamw_8bit`, per-device batch size 4, gradient accumulation 2, and maximum sequence length 512. Unless stated otherwise, LoRA targeted attention and MLP modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.

`NR` means the value was not retained in the visible saved notebook output. It is left explicit rather than reconstructed or estimated. Accuracy and validation loss below come from the 200-example validation evaluation recorded by the training helper, except for the pilot, which used its 10% validation slice.

## Training Runs

| Run | Model | r | alpha | LR | Epochs | Train loss | Val loss | GSM8K accuracy | Peak VRAM (GB) | Time (min) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pilot_r8_10pct` | unsloth/qwen3-0.6B | 8 | 16 | 2e-4 | 3 | 0.387138 | 0.467859 | 0.500 | 5.442 | 4.352 |
| `rank_8_2` (standard run) | unsloth/qwen3-0.6B | 8 | 16 | 2e-4 | 3 | 0.340337 | 0.421812 | 0.115 | 5.668 | 42.965 |
| `rank_4` | unsloth/qwen3-0.6B | 4 | 16 | 2e-4 | 3 | 0.343957 | 0.422618 | 0.355 | NR | NR |
| `rank_8` | unsloth/qwen3-0.6B | 8 | 16 | 2e-4 | 3 | 0.341693 | 0.421279 | 0.165 | NR | NR |
| `rank_16` | unsloth/qwen3-0.6B | 16 | 16 | 2e-4 | 3 | 0.341123 | 0.419591 | **0.505** | 3.068* | 0.155* |
| `alpha_4` | unsloth/qwen3-0.6B | 16 | 4 | 2e-4 | 3 | 0.396255 | 0.426294 | 0.365 | NR | NR |
| `alpha_8` | unsloth/qwen3-0.6B | 16 | 8 | 2e-4 | 3 | 0.373207 | 0.421156 | 0.395 | NR | NR |
| `alpha_16` | unsloth/qwen3-0.6B | 16 | 16 | 2e-4 | 3 | 0.340004 | 0.419945 | 0.320 | NR | NR |

\* The saved `rank_16` summary was produced after resuming an already completed checkpoint, so its 0.155-minute time and 3.068 GB peak VRAM describe only that resume/evaluation invocation, not end-to-end training. They should not be compared with the full pilot and standard-run measurements.

The later artifact-loading cell reports five rank-prefixed summary directories, while the saved notebook exposes names and metrics for only `rank_4`, `rank_8`, `rank_8_2`, and `rank_16`. The fifth cached summary is not identifiable from the repository artifact, so no row or metrics are fabricated for it.

## Rank Ablation Finding

Rank 16 won the recorded rank comparison with 0.505 validation accuracy and the lowest validation loss, 0.419591. Rank 4 reached 0.355, while rank 8 reached only 0.165 despite very similar losses, showing that loss alone was not a reliable selector for downstream exact match. The duplicate standard r=8 run scored 0.115, reinforcing that this configuration was weak in the recorded evaluations.

## Alpha Ablation Finding

At fixed r=16, alpha 8 produced the best downstream result, with 0.395 accuracy and a 0.010 parse-failure rate. Alpha 16 had the lowest validation loss (0.419945) but only 0.320 accuracy, while alpha 4 reached 0.365. Adapter scale therefore affected exact-match behavior more clearly than it affected validation loss.

## Evaluation Snapshot

The later 100-example test evaluation recorded zero-shot base accuracy of 0.50, 5-shot base accuracy of 0.02, and fine-tuned accuracy of 0.42, with parse-failure rates of 0.00, 0.03, and 0.03 respectively. The LLM judge ranked the models in the same order for reasoning and arithmetic: zero-shot scored 3.62/3.82, fine-tuned 2.70/3.02, and 5-shot 1.00/1.00. Fine-tuning's clearest gain was format compliance (3.96 versus 1.04 for zero-shot), and the pairwise judge showed a 30% order-flip rate, so judge results are treated as supporting evidence rather than ground truth.

## Winning Configuration and Next Experiment

The recorded winner is `rank_16` (r=16, alpha=16) because it achieved the highest validation exact-match accuracy, 0.505, with the best rank-ablation validation loss. That conclusion is provisional: the separately named `alpha_16` run used the same nominal r/alpha/LR settings but reached only 0.320 accuracy, indicating that resume state, run isolation, or uncontrolled variation needs investigation. With one more day of compute, the first priority would be clean from-scratch replications of r=16 with alpha 8 and alpha 16 across multiple seeds, isolated output directories, complete VRAM/runtime capture, and evaluation on the full held-out set; only after reproducibility is established would I tune further around the winning scale.
