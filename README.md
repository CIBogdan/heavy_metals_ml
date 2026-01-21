# Heavy Metals Risk Analysis with Machine Learning

End-to-end machine learning pipelines for heavy-metal contamination analysis and risk classification, designed for reproducible scientific research and applied environmental and food safety risk assessment.

This project implements supervised classification and artificial neural network (ANN) regression models for the analysis of heavy-metal contamination patterns (e.g., Pb, Cd, Hg, As) in regulatory alert-style datasets.

---

## Key features

• Supervised machine learning for heavy-metal risk classification (binary and multi-class)  
• ANN regression models for concentration prediction across multiple metals  
• Feature importance and model interpretability tools  
• Automated generation of paper-ready figures (300 dpi)  
• CI-stable, reproducible research workflow (ruff, black, pytest, GitHub Actions)  

---

## Example output

<p align="center">
  <img src="outputs/paper_figures/classification/confusion_binary.png" width="600">
</p>

*Example: binary risk classification performance for heavy-metal notifications.*

---

## Reproducibility

The dataset used in this study is **not distributed** with the repository due to data-sharing constraints.

To reproduce the analyses and generate all paper-ready figures (300 dpi), provide the local dataset path and run:

```bash
python scripts/make_paper_figures.py --data PATH/TO/df_clean.csv
