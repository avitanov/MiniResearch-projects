# GSM8K LoRA Fine-Tuning and Evaluation Study

**Internship project:** This work was completed as part of the Loka Machine Learning Internship. References to "Week X", task numbers, and labs follow the original internship curriculum.

A study by Atanas Vitanov during the Loka ML internship, investigating how LoRA
rank, adapter scaling, prompt formatting, and evaluation choices affect arithmetic
reasoning with `unsloth/qwen3-0.6B`. The main experiment uses 4-bit model loading
and LoRA adapters on attention and MLP modules.

## Recorded findings

| Evaluation | Accuracy |
|---|---:|
| Zero-shot base model, 100-example test subset | 50% |
| Fine-tuned model, same reported test subset size | 42% |
| Five-shot base model, 100-example test subset | 2% |
| Best recorded rank run, 200-example validation subset | 50.5% |

Fine-tuning did not improve the reported test accuracy. The useful findings are
that validation loss alone did not predict answer accuracy, adapter scale affected
exact match, and formatting gains did not imply better reasoning. The pairwise
LLM judge had a 30% order-flip rate, limiting how strongly its scores can be interpreted.

Two runs with the same nominal rank/alpha settings produced inconsistent validation
accuracy. Clean replications with isolated run directories and multiple seeds are
needed before treating an adapter configuration as reliably best. Some recorded
resource measurements describe a resumed invocation rather than full training.
See [experiment_log.md](experiment_log.md) for the complete measurements and caveats.

## Notebooks

1. [LoRA rank visualization](notebooks/lora_rank_visualization.ipynb): low-rank approximation and parameter counts.
2. [Base-model exploration](notebooks/base_model_exploration.ipynb): probing and the GSM8K zero-shot baseline.
3. [LoRA from scratch](notebooks/lora_from_scratch.ipynb): a PyTorch `LoRALinear` implementation.
4. [GSM8K fine-tuning](notebooks/gsm8k_finetuning.ipynb): formatting, training, ablations, evaluation, and LLM judging.

The original notebook task numbers are retained inside the notebooks to preserve
experimental context. Shared helper code is in `src/`. Saved outputs are included;
trained adapters and base-model weights are not.

## Run

For local notebook inspection and the foundation exercises:

```bash
uv sync
uv run jupyter lab notebooks/
```

The recorded environment requires Python 3.12.3. The base-model and main fine-tuning
notebooks target Google Colab with a T4 CUDA GPU; the local environment alone does
not supply the full Unsloth training stack. In Colab:

1. Open the chosen notebook and enable the T4 GPU runtime.
2. Upload the helper files from `src/` to the working directory, as described in the notebook.
3. Run the notebook's dependency-install cells and mount Drive when prompted.
4. Review the output-directory settings before training. The main notebook retains
   the original Drive path, so choose a fresh run location to avoid mixing checkpoints.
5. Run the base-model notebook and verify its baseline before using the comparison
   in the main notebook; that baseline is carried over manually.
6. Supply the OpenRouter API key only when running the LLM judging section.

The notebooks download their base model/data and generate adapters during a run.
GPU runs and hosted judging require their respective resources. No new training or
validation run was performed while extracting this study.
