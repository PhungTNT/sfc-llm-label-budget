# UIT-VSFC: Zero-Shot LLMs, PhoBERT, and Human-Label Budgets

**Research artifacts for Vietnamese student-feedback classification**  
**Repository:** [PhungTNT/sfc-llm-label-budget](https://github.com/PhungTNT/sfc-llm-label-budget)  
**Status:** Experimental results and research-in-progress; not a camera-ready paper or a fully reproducible release.

## Overview

This repository documents experiments on the **UIT-VSFC** (Vietnamese Student Feedback Corpus) dataset. We investigate three questions:

1. How does Qwen2.5-7B zero-shot classification compare with supervised PhoBERT and TF-IDF baselines?
2. How many manually labeled examples are needed for PhoBERT to match or exceed the zero-shot LLM on sentiment classification?
3. Can PhoBERT trained with **LLM-generated pseudo-labels**, without human training labels, improve on its teacher model?

The experimental record includes notebook code, preserved cell outputs where available, per-run evaluation metrics and predictions, figures, and bootstrap/McNemar analyses. **Some notebook cells and saved runtime outputs remain in Vietnamese; an English filename and README do not translate those cells.**

## Dataset and evaluation

| Split | Examples |
| --- | ---: |
| Training | 11,426 |
| Validation | 1,583 |
| Test | 3,166 |

UIT-VSFC contains **three sentiment classes** (negative, neutral, positive) and **four topic classes**. The learning-curve and pseudo-label student experiments reported here concern **sentiment classification**. The primary metric is **test Macro-F1**; accuracy and classwise scores are supplementary. Training and validation labels are not used to supervise the E5 pseudo-label student runs.

## Repository files

| File | Contents |
| --- | --- |
| [`UIT_VSFC_Experiments.ipynb`](UIT_VSFC_Experiments.ipynb) | The Kaggle/Colab experimental notebook: code, visible cell outputs, and available execution logs. This is a **research log**, not a clean one-click reproduction notebook. |
| [`archive.zip`](archive.zip) | Original experiment backup: dataset artifacts, TF-IDF baselines, supervised PhoBERT results, and Qwen2.5-7B zero-shot predictions/metrics. |
| [`UIT_VSFC_FINAL_RESEARCH_RESULTS.zip`](UIT_VSFC_FINAL_RESEARCH_RESULTS.zip) | Consolidated E3 and E5 evaluations (**67 archived files**), including 21 learning-curve runs, six pseudo-label student runs, figures, summary tables, and statistical analyses. |
| [`E5_train_11426_dev_1583.zip`](E5_train_11426_dev_1583.zip) | Qwen-generated pseudo-labels for all 11,426 training and 1,583 validation examples (13,009 total); includes the saved confidence-based labeling outputs. |

**Note:** The English notebook filename above is the intended name after renaming the existing GitHub file. If the file still appears as `th-c-nghi-m.ipynb`, rename it on GitHub so this link works.

### Contents of the consolidated results archive

```text
UIT_VSFC_FINAL_RESEARCH_RESULTS.zip
├── E3_learning_curve/
│   └── lc_sentiment_<fraction>pct_seed_<seed>/
│       ├── metrics.csv
│       └── pred.csv
├── E5_pseudo_student/
│   ├── all_seed_<seed>/
│   ├── conf80_seed_<seed>/
│   ├── E5_all_results.csv
│   ├── E5_summary_final.csv
│   ├── E5_bootstrap_comparison.csv
│   └── E5_mcnemar_comparison.csv
└── final_analysis/
    ├── table1_main_sentiment.csv
    ├── learning_curve_final_summary.csv
    ├── learning_curve_final.png
    └── learning_curve_final.pdf
```

The `metrics.csv` files provide numerical evaluation results, while `pred.csv` files contain individual test predictions. **Complete per-step training logs and trained model weights are not necessarily included** in the archived outputs; check the notebook for preserved logs.

## Experimental coverage

| Experiment | Description | Status |
| --- | --- | --- |
| E1: Baselines | TF-IDF (LR/SVM) and supervised PhoBERT | Completed for archived runs |
| E2: Zero-shot | Qwen2.5-7B, prompts P1–P3, raw/calibrated variants; sentiment and topic | Completed for Qwen2.5-7B |
| E3: Learning curve | PhoBERT sentiment, 1%, 2%, 5%, 10%, 25%, 50%, 100% training fractions × 3 seeds | **21/21 completed** |
| E5: Pseudo-labeling | Qwen P2 calibrated pseudo-labels; PhoBERT students trained on all labels or confidence-filtered labels | **6/6 student runs completed** |
| Statistical analysis | Paired bootstrap and McNemar comparisons of E5 and Qwen P2 | Completed for reported comparisons |

Proposed experiments with **Qwen3, Llama, and SeaLLM** are **not part of the completed results** documented here.

## Main sentiment-classification results

| Method | Test Macro-F1 |
| --- | ---: |
| TF-IDF + SVM | 0.74762 |
| Supervised PhoBERT (five seeds) | **0.83484 ± 0.00500** |
| Qwen2.5-7B, P2 raw zero-shot | 0.68315 |
| Qwen2.5-7B, P2 calibrated zero-shot | 0.67234 |
| PhoBERT student, all pseudo-labels (three seeds) | 0.67862 ± 0.01158 |
| PhoBERT student, confidence-filtered pseudo-labels (`conf80`, three seeds) | **0.70925 ± 0.00455** |

The **five-seed supervised PhoBERT baseline** and the **100%-data E3 learning-curve condition** follow different experimental protocols; their results should not be combined.

### Human-label budget

- **329** labeled training-plus-validation examples: mean PhoBERT Macro-F1 **0.67152**.
- **671** labeled examples: mean PhoBERT Macro-F1 **0.75220**. This is the **first observed experimental point** exceeding Qwen P2 raw (**0.68315**) among the tested fractions.
- **Approximately 365** labels is a *log-scale interpolation* between neighboring tested points, **not an observed training run or validated minimum**.

### Pseudo-label learning

- The `conf80` PhoBERT student scored **0.70925** mean Macro-F1, a **+0.02610** difference over Qwen P2 raw.
- A paired test-set bootstrap with **2,000 resamples** estimated a **95% confidence interval [0.00819, 0.04385]** for this Macro-F1 difference (mean of three student seeds versus the Qwen raw predictions).
- McNemar tests compare paired **correct-versus-incorrect predictions (accuracy)**, rather than Macro-F1 directly. These tests and the confidence intervals have not been adjusted for multiple comparisons.

## How to review the experiments

1. Open `UIT_VSFC_Experiments.ipynb` on GitHub to inspect the original execution sequence and any saved notebook cell outputs.
2. Download `archive.zip` for original baselines and zero-shot predictions.
3. Download and extract `UIT_VSFC_FINAL_RESEARCH_RESULTS.zip` for the E3/E5 run-level CSV files and final plots.
4. Inspect `E5_train_11426_dev_1583.zip` when evaluating the pseudo-label data generation procedure.
5. Recalculate metrics from archived per-example predictions if conducting an independent audit.

**Reproducibility caveat:** The notebook reflects an exploratory Kaggle workflow and may contain troubleshooting/restarted-session cells and environment-dependent dependencies. It is **not guaranteed** to execute from top to bottom without modification. The CSV/ZIP archives are experimental evidence, not a complete executable environment specification. Before redistributing any source dataset records, review the original dataset's license and redistribution conditions.

## Limitations and research status

The current evidence is strongest for **sentiment classification with Qwen2.5-7B and PhoBERT on UIT-VSFC**. Findings may not generalize to other datasets, language models, or label distributions. Results involving human-label thresholds should be interpreted descriptively because only seven training fractions were tested and model configurations were evaluated against the same test set. Further model comparisons, reproducibility cleanup, bibliographic verification, and venue-specific manuscript formatting remain future work.
