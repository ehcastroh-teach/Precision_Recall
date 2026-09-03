# Precision, Recall, and Threshold

A two-notebook series that teaches precision, recall, and decision threshold selection for binary classification. Both main notebooks use the same conceptual framework applied to different domains and models: breast cancer detection with logistic regression, and spam email detection with a support vector machine (SVM). Companion homework notebooks let you practice the same skills on new datasets (wine quality and handwritten digit recognition) without repeating the instructional scaffolding. By working through all four, you will see how the same precision-recall trade-off shows up differently depending on the real-world cost of false positives versus false negatives - and how to make a principled decision about where to set the threshold.

## Learning Objectives

- Define precision and recall from the confusion matrix and explain what each metric measures
- Explain the precision-recall trade-off and why it is inherent to any probabilistic binary classifier
- Apply decision threshold adjustment to shift a model's precision-recall balance without retraining
- Build and tune a scikit-learn `Pipeline` using `GridSearchCV` with stratified cross-validation
- Interpret the impact of the cross-validation scoring metric on which model is selected
- Compare precision-recall outcomes across two different classifiers and two different domains
- Implement precision and recall from scratch before relying on library functions
- Transfer the workflow to an unseen dataset with minimal guidance

## Data / File Dictionary

| File | Description |
|---|---|
| `01_precision_recall_logistic_regression.ipynb` | Main lesson notebook - teaches precision, recall, and threshold using breast cancer data and logistic regression |
| `02_precision_recall_svm.ipynb` | Applied lesson notebook - applies the same framework to spam detection using an SVM with RBF kernel |
| `precision_recall_logistic_regression_homework.ipynb` | Homework worksheet - logistic regression precision-recall analysis on the Wine Recognition dataset |
| `precision_recall_svm_homework.ipynb` | Homework worksheet - SVM precision-recall analysis on the Digits dataset with transfer to Wine Recognition |
| `BreastCancer.csv` | Breast Cancer Wisconsin (Diagnostic) dataset - 569 samples, 30 features, binary diagnosis label (B/M) |
| `SpamEmail.csv` | Spambase dataset - 4,601 email samples, 57 word/character frequency features, binary spam label (0/1) |
| `requirements.txt` | Python package dependencies for reproducing the notebooks |

## Workflow Diagram

```
BreastCancer.csv -----> 01_precision_recall_logistic_regression.ipynb
                                |
                                | introduces concepts: confusion matrix,
                                | precision, recall, threshold, Pipeline,
                                | GridSearchCV, feature importance
                                v
SpamEmail.csv --------> 02_precision_recall_svm.ipynb
                                |
                                | applies same framework with SVM/RBF;
                                | contrasts spam cost structure vs. cancer
                                v
                   (built-in sklearn datasets)
                                |
              +-----------------+-----------------+
              |                                   |
              v                                   v
precision_recall_logistic_           precision_recall_svm_
regression_homework.ipynb            homework.ipynb
(Wine Recognition dataset)           (Digits dataset -> Wine dataset)
```

Each lesson notebook is self-contained: the two CSV datasets feed their respective notebooks independently. Notebook 02 builds on the conceptual vocabulary introduced in Notebook 01. The homework notebooks depend only on sklearn's built-in datasets and require no additional data files.

## Step-by-Step Walkthrough

**Start with `01_precision_recall_logistic_regression.ipynb`**

- **Part 1 - The Concepts:** introduces the confusion matrix, defines precision and recall mathematically, and explains when to prioritize one over the other. This conceptual grounding applies to both lesson notebooks - read this section carefully before running any model code. The decision threshold is introduced here as a continuous control knob, not a fixed value.

- **Part 2 - Dataset and Exploration:** loads the Breast Cancer Wisconsin dataset and runs three sanity checks before touching the data: shape, missing values, and duplicates. Each check catches a different class of problem. The EDA motivates the choice of recall as the primary metric: missing a malignant diagnosis is much more costly than a false alarm that leads to a follow-up biopsy. This cost-asymmetry reasoning is the key design decision in the notebook.

- **Part 3 - Data Preparation:** separates features from labels and performs a three-way split into training, validation, and test sets. The critical design choice is splitting before fitting the scaler. Scaling after the split prevents the test set's statistics from leaking into the scaler, which would produce optimistic estimates of generalization performance. The test set is held out completely and not touched until the very end.

- **Part 4 - Model Training and Analysis:** builds a logistic regression `Pipeline` (scaler + model in one object, preventing leakage during cross-validation), runs two separate `GridSearchCV` passes - one optimizing recall and one optimizing precision - then plots precision and recall as a function of the decision threshold. The dual search is intentional: it makes the impact of the scoring metric on hyperparameter selection visible. The notebook concludes with feature importance via logistic regression coefficients and final evaluation on the held-out test set using a recall-favoring threshold of 0.4.

**Then run `02_precision_recall_svm.ipynb`**

- **Part 1 - Dataset and Exploration:** loads the Spambase dataset. Note the different cost asymmetry from cancer detection. Blocking a legitimate email is typically more disruptive to the user than letting some spam through, which shifts the recommended threshold toward higher precision - the opposite choice from Notebook 01.

- **Part 2 - Data Preparation:** uses the same three-way split strategy as Notebook 01 with a fixed random state for reproducibility.

- **Part 3 - SVM Pipeline and Hyperparameter Tuning:** introduces the SVM with RBF kernel and explains why feature scaling is essential for distance-based models. Features on different scales would dominate the distance calculation and bias the decision boundary. The same dual `GridSearchCV` strategy is applied, but training is slower because `probability=True` triggers an internal Platt scaling step.

- **Part 4 - Threshold Analysis:** plots the threshold-precision-recall curve for the SVM and evaluates the final model with a precision-favoring threshold. Comparing this curve with the logistic regression curve from Notebook 01 illustrates how the shape reflects the model's probability calibration for its specific dataset.

**Practice with the homework notebooks**

- **`precision_recall_logistic_regression_homework.ipynb`:** applies the logistic regression workflow to the Wine Recognition dataset. Section 2 asks you to implement precision and recall from scratch using only a confusion matrix, before reaching for sklearn's metric functions. This confirms you understand the definitions rather than just calling library code. Section 5 transfers the workflow to the breast cancer sklearn toy dataset as a sanity check.

- **`precision_recall_svm_homework.ipynb`:** applies the SVM workflow to the Digits dataset, binarizing the target to detect the digit "3" versus all other digits. Section 4 asks you to find the threshold that maximizes recall subject to a precision floor, a more realistic decision rule than simply minimizing the absolute precision-recall gap. Section 5 transfers the workflow to the Wine Recognition dataset, contrasting how the SVM performs on a dataset with different feature count and sample size.

## How to Run

Clone the repo and install dependencies into a clean Python environment:

```bash
git clone https://github.com/ehcastroh-teach/Precision_Recall.git
cd Precision_Recall
pip install -r requirements.txt
```

Launch Jupyter and open either lesson notebook:

```bash
jupyter notebook
```

Run all cells top-to-bottom after a clean kernel restart (`Kernel > Restart & Run All`). Start with `01_precision_recall_logistic_regression.ipynb` before `02_precision_recall_svm.ipynb`. Notebook 02 may take several minutes on the `GridSearchCV` step because SVM training with `probability=True` is slower than logistic regression. The homework notebooks have no additional dependencies beyond `requirements.txt` and load their datasets directly from sklearn.

## Key Concepts Glossary

| Term | Definition |
|---|---|
| **Confusion matrix** | A 2x2 table counting true positives, true negatives, false positives, and false negatives for a binary classifier |
| **Precision** | Of all samples predicted positive, the fraction that are actually positive: TP / (TP + FP) |
| **Recall** | Of all samples that are actually positive, the fraction the model correctly identified: TP / (TP + FN) |
| **False positive** | A negative sample incorrectly predicted as positive (a false alarm) |
| **False negative** | A positive sample incorrectly predicted as negative (a miss) |
| **Decision threshold** | The probability cutoff above which a model predicts positive; defaults to 0.5 but can be adjusted without retraining |
| **Precision-recall trade-off** | Lowering the threshold increases recall but decreases precision; raising it does the opposite |
| **Pipeline** | A scikit-learn object that chains preprocessing and modeling steps to prevent data leakage during cross-validation |
| **Stratified K-Fold** | Cross-validation that preserves the class ratio in each fold, used when classes are imbalanced |
| **Regularization (C)** | The inverse of regularization strength in logistic regression and SVM - larger C allows more complex boundaries by reducing the penalty on large coefficients |
| **RBF kernel** | Radial basis function kernel; maps inputs into a higher-dimensional space so a linear boundary in that space becomes a nonlinear boundary in the original feature space |
| **Platt scaling** | A technique that converts SVM decision function scores into calibrated probability estimates; enabled by `probability=True` and the reason SVM training is slower than logistic regression here |
| **Feature leakage** | Contamination of the model's evaluation when information from the test set influences preprocessing steps such as scaling or imputation during training |

## Further Reading

- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* - chapters on classification and model evaluation
- *An Introduction to Statistical Learning* - Chapter 4 on classification
- *scikit-learn User Guide: Model Evaluation and Scoring* - section on classification metrics
- *Google ML Crash Course: Classification* - sections on precision, recall, and ROC curves

## Credits and Acknowledgements

- Breast Cancer Wisconsin (Diagnostic) dataset: W.H. Wolberg, W.N. Street, O.L. Mangasarian - University of Wisconsin, via UCI Machine Learning Repository
- Spambase dataset: M. Hopkins, E. Reeber, G. Forman, J. Suermondt - Hewlett-Packard Labs, via UCI Machine Learning Repository

## Contact

<div align="center">
  <img src="images/thumbnails/ehcastroh_teach_banner_flower.png" alt="ehcastroh" width="90" style="border-radius: 50%;" />

  <sub>ehcastroh</sub>

  <a href="https://github.com/ehcastroh">GitHub</a> · <a href="https://www.linkedin.com/in/ehcastroh/">LinkedIn</a>
</div>
