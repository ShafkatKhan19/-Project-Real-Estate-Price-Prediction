# Week 2 Project: Real Estate Price Prediction
**Course:** AI250 — LIU Brooklyn  
**Author:** Shafkat Khalek Khan  
**Project:** Week 2 — Supervised Learning with California Housing Data

---

## Overview
Builds and compares supervised learning models to predict house prices using the California Housing dataset.

## What's Included
- `week2_project.py` — Full implementation (Parts 1–5)
- `Week2_Project_Report.docx` — Complete report with embedded visualizations
- 6 PNG visualizations generated automatically

## Parts Completed
- **Part 1**: Data exploration, statistics, correlation heatmap, feature histograms
- **Part 2**: Linear Regression (R²=0.736, RMSE=$63,469)
- **Part 3**: Decision Tree depth experiments (best depth=5, R²=0.865)
- **Part 4**: Classification — Logistic Regression (80.8%) vs Decision Tree (77.4%)
- **Part 5**: Full model comparison and analysis

## Results Summary
| Model | Task | Performance |
|-------|------|-------------|
| Linear Regression | Regression | R² = 0.736 |
| Decision Tree (depth=5) | Regression | R² = 0.865 ★ |
| Logistic Regression | Classification | Accuracy = 80.8% ★ |
| Decision Tree (depth=5) | Classification | Accuracy = 77.4% |

## How to Run
```bash
pip install numpy pandas matplotlib scikit-learn seaborn
python week2_project.py
```
