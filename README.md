# Precision, Recall, and Threshold

A two-notebook series that teaches precision, recall, and decision threshold selection for binary classification. Both notebooks use the same conceptual framework applied to different domains and models: breast cancer detection with logistic regression, and spam email detection with a support vector machine (SVM). By working through both, you will see how the same precision-recall trade-off shows up differently depending on the real-world cost of false positives versus false negatives.

## Learning Objectives

- Define precision and recall from the confusion matrix and explain what each metric measures
- Explain the precision-recall trade-off and why it is inherent to any probabilistic binary classifier
- Apply decision threshold adjustment to shift a model's precision-recall balance without retraining
- Build and tune a scikit-learn `Pipeline` using `GridSearchCV` with stratified cross-validation
- Interpret the impact of the cross-validation scoring metric on which model is selected
- Compare precision-recall outcomes across two different classifiers and two different domains

## Data / File Dictionary

| File | Description |
|---|---|
| `01_precision_recall_logistic_regression.ipynb` | Main lesson notebook - teaches precision, recall, and threshold using breast cancer data and logistic regression |
| `02_precision_recall_svm.ipynb` | Applied notebook - applies the same framework to spam detection using an SVM |
| `BreastCancer.csv` | Breast Cancer Wisconsin (Diagnostic) dataset - 569 samples, 30 features, binary diagnosis label (B/M) |
| `SpamEmail.csv` | Spambase dataset - 4,601 email samples, 57 word/character frequency features, binary spam label (0/1) |

## Workflow Diagram

```
BreastCancer.csv  --+
                    +--> 01_precision_recall_logistic_regression.ipynb
SpamEmail.csv    ---+--> 02_precision_recall_svm.ipynb
```

Each notebook is self-contained: the two datasets feed their respective notebooks independently. Notebook 02 builds on the conceptual vocabulary introduced in Notebook 01.

## Step-by-Step Walkthrough

**Start with `01_precision_recall_logistic_regression.ipynb`**

- **Part 1** introduces the confusion matrix, defines precision and recall mathematically, and explains when to prioritize one over the other. This conceptual grounding applies to both notebooks - read this carefully before running any model code.
- **Part 2** loads and explores the Breast Cancer Wisconsin dataset. The exploration motivates the choice of recall as the primary metric: missing a malignant diagnosis is much more costly than a false alarm that leads to a follow-up biopsy.
- **Part 3** prepares the data. The key design choice is splitting train/test before scaling, not after. Scaling after the split prevents the test set's statistics from leaking into the scaler, which would produce optimistic estimates of generalization performance.
- **Part 4** builds a logistic regression pipeline, runs two hyperparameter searches (one optimizing recall, one optimizing precision), plots precision and recall as a function of the decision threshold, and evaluates the chosen model on the held-out test set.

**Then run `02_precision_recall_svm.ipynb`**

- **Part 1** loads and explores the Spambase dataset. Note the different cost asymmetry from cancer detection: blocking a legitimate email is more disruptive than missing a spam message, which shifts the recommended threshold toward higher precision.
- **Part 2** prepares the data using the same three-way split strategy as Notebook 01.
- **Part 3** introduces the SVM and the RBF kernel, explains why feature scaling is essential for distance-based models, and runs the same dual hyperparameter search.
- **Part 4** plots the threshold curve for the SVM model and evaluates the final model with a precision-favoring threshold.

## How to Run

Install dependencies from a clean environment:

```bash
pip install -r requirements.txt
```

Launch Jupyter and open either notebook:

```bash
jupyter notebook
```

Run all cells top-to-bottom after a clean kernel restart. Notebook 02 (`svm`) may take several minutes to complete the `GridSearchCV` step due to SVM training cost with `probability=True`.

## Key Concepts Glossary

| Term | Definition |
|---|---|
| **Confusion matrix** | A 2x2 table counting true positives, true negatives, false positives, and false negatives for a binary classifier |
| **Precision** | Of all samples predicted positive, the fraction that are actually positive: TP / (TP + FP) |
| **Recall** | Of all samples that are actually positive, the fraction the model correctly identified: TP / (TP + FN) |
| **False positive** | A negative sample incorrectly predicted as positive (a false alarm) |
| **False negative** | A positive sample incorrectly predicted as negative (a miss) |
| **Decision threshold** | The probability cutoff above which a model predicts positive; defaults to 0.5 but can be adjusted |
| **Precision-recall trade-off** | Lowering the threshold increases recall but decreases precision; raising it does the opposite |
| **Pipeline** | A scikit-learn object that chains preprocessing and modeling steps to prevent data leakage during cross-validation |
| **Stratified K-Fold** | Cross-validation that preserves the class ratio in each fold, used when classes are imbalanced |
| **Regularization (C)** | The inverse of regularization strength in logistic regression and SVM - larger C allows more complex boundaries |
| **RBF kernel** | Radial basis function kernel; maps inputs into a higher-dimensional space to find nonlinear decision boundaries |

## Further Reading

- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* - chapters on classification and model evaluation
- *An Introduction to Statistical Learning* - Chapter 4 on classification
- [scikit-learn: Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [Google ML Crash Course: Precision and Recall](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)

## Credits and Acknowledgements

- Breast Cancer Wisconsin (Diagnostic) dataset: W.H. Wolberg, W.N. Street, O.L. Mangasarian - University of Wisconsin, via UCI Machine Learning Repository
- Spambase dataset: M. Hopkins, E. Reeber, G. Forman, J. Suermondt - Hewlett-Packard Labs, via UCI Machine Learning Repository
- Dataset collection and original notebook structure
---

## Contact

<div align="center">
  <img src="images/thumbnails/ehcastroh_teach_banner_flower.png" alt="ehcastroh" width="90" style="border-radius: 50%;" />

  <sub>ehcastroh</sub>

  <a href="https://github.com/ehcastroh">GitHub</a> · <a href="https://www.linkedin.com/in/ehcastroh/">LinkedIn</a>
</div>
