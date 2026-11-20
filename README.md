# thesis-ppm-xai

Comparison of explanation sensitivity (LSTM and Transformer models) for predictive process monitoring (PPM) on next-activity and remaining-time prediction, using five explainers: Integrated Gradients, Expected Gradients, Gradient Saliency, KernelSHAP and LIME.

## Setup

**Prerequisites:** Python 3.12, a virtual/conda environment.

**Installation:**
```bash
conda create -n tfm-ppm python=3.12
conda activate tfm-ppm
pip install -r requirements.txt
```

## Usage

### Data

Raw event logs are not included in this repository. Download the BPI Challenge logs from [4TU.ResearchData](https://data.4tu.nl/) and place the CSVs under `data/`:
- BPIC12 (complete, A, W complete)
- BPIC17 (complete)
- BPIC20 (Request for Payment, Permit log)

### Running the full pipeline

Preprocessing, training and test evaluation (overall and by prefix length) for all dataset x task x architecture combinations are run with a single resumable script:

```bash
python scripts/train_full_sweep.py
```

This writes processed datasets to `processed_datasets/full_sweep/`, trained model checkpoints and metrics to `experiments/full_sweep/`. Re-running the script skips any dataset, model or result already produced.

### Data structure

```
data/                                   
processed_datasets/full_sweep/<name>/processed_multi/
    metadata.json
    <task>_train.csv
    <task>_test.csv
experiments/full_sweep/
    results.csv                         
    results_by_length.csv               
    <name>/<task>_<architecture>/model/best_model_<task>.keras
```

### Explainability

Each explainer in `explainability/` takes a trained model plus a batch of prefixes and returns a `(length, n_features)` attribution matrix.

**Integrated Gradients** (`ig_explainer.py`):
```python
from explainability.ig_explainer import explain_ig
explain_ig(model, model_module, task, prefix, length, baseline_builders, column_blocks, feature_names)
```

**Expected Gradients** (`eg_explainer.py`, via `alibi`):
```python
from explainability.eg_explainer import explain_eg
explain_eg(model, model_module, task, prefix, length, background_builder, column_blocks, feature_names)
```

**Gradient Saliency** (`gradient_saliency_explainer.py`):
```python
from explainability.gradient_saliency_explainer import explain_gradient_saliency
explain_gradient_saliency(model, model_module, task, prefix, length, column_blocks, feature_names)
```

**KernelSHAP** (`shap_kernel_explainer.py`):
```python
from explainability.shap_kernel_explainer import explain_shap
explain_shap(model, model_module, task, prefix, length, feature_names)
```

**LIME** (`lime_explanations.py`):
```python
from explainability.lime_explanations import explain_lime
explain_lime(model, model_module, task, prefix, length, background_builder, col_index,
             event_vars, case_vars, get_layout, feature_names)
```

### Sensitivity evaluation

`sensitivity_evaluation/` aims to compute explanation-stability metrics (Relative Input/Output Stability, Feature Agreement, Rank Agreement) over perturbed prefixes.
