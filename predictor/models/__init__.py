"""Modeling ladder: baselines -> gradient boosting -> small deep model.

Per CLAUDE.md, report the full ladder honestly: ElasticNet / RandomForest
baselines, then XGBoost / LightGBM (usually the winner on this tabular
feature set), then a small PyTorch 1D-CNN / k-mer MLP on raw sequence. If
boosting beats the deep model at the available N, say so plainly.
"""
