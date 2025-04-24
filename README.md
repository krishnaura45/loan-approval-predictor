# loan-approval-predictor
Predicting loan approvals with probabilistic modeling to support financial risk mitigation and smarter lending decisions<br>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![ROC AUC Optimized](https://img.shields.io/badge/Optimized--for-ROC%20AUC-yellowgreen?style=for-the-badge)

### **Project Duration**: Mar 1, 2025 - Apr 1, 2025

## 🎯 Problem Statement
Financial institutions need accurate tools to determine the creditworthiness of applicants. The goal of this project was to build a binary classification model to predict whether a loan applicant would be approved or not. This challenge was hosted on Kaggle as part of **Playground Series - Season 4, Episode 10**. Participants were required to submit predicted probabilities for loan approvals, and the evaluation metric used was **Area Under the ROC Curve (AUC-ROC)**. Higher AUC scores indicated better model performance.

<img src="https://github.com/user-attachments/assets/50c08d5a-c699-417a-8545-8f46423685d7" height=600 alt="LoanSure Cover">

---

## 🧩 Top Approach

You can explore the complete methodology in my notebook:  
🔗 [PS4E10 - Solution 2](https://github.com/krishnaura45/loan-approval-prediction/blob/main/ps4e10-lap-solution-2.ipynb)

Key steps followed:

- 📊 **Data Augmentation & Preprocessing**:
  - Merged the official competition training data with the original open-source credit risk dataset to increase training volume.
  - Performed proper alignment and null handling to ensure feature consistency.

- 🧠 **Model Stacking with OOFs & Ensembling**:
  - Leveraged pre-computed out-of-fold (OOF) predictions and test set predictions from diverse base models including `lgbm`, `catboost`, `tabnet`, and others.
  - Combined OOFs for robust out-of-sample validation.

- 🧮 **Hill Climbing Ensembling**:
  - Applied a **hill climbing algorithm** to greedily find the optimal blend of base models by maximizing the AUC on validation data.
  - Iteratively selected models that improved validation AUC until no further gain was observed.

- 🧪 **Final Prediction & Submission**:
  - Used the optimal ensemble weights derived via hill climbing to combine model predictions on the test set.
  - Generated submission probabilities accordingly.

---

## 📈 Results

- ✅ **Public Leaderboard Scores**:
  - Achieved scores of 0.97324, 0.97347, 0.97369, and 0.97285.

- 🏁 **Private Leaderboard Scores**:
  - Final scores: 0.96855, **0.96886** (best), 0.96877, and 0.96712 for subsequent submissions.

- 🥇 **Rank Achieved**:
  - Ranked **218 / 4080 participants** and **3858 teams**, as a **solo participant**.
 
![Leaderboard Standings](https://github.com/user-attachments/assets/e1eab2c0-017d-435c-a621-21167eaa942d)


---

## 🧰 References

- 📂 Kaggle Competition: [Loan Approval Prediction](https://www.kaggle.com/competitions/playground-series-s4e10)
- 📁 Dataset: [Competition Data](https://www.kaggle.com/datasets/chilledwanker/loan-approval-prediction)
- 📁 Source Dataset: [Open Credit Risk Dataset](https://www.kaggle.com/datasets/ajay1735/historical-loan-application-dataset)

---

## 🛠️ Tech Stack

- **Language**: Python 🐍
- **Libraries**:
  - `pandas`, `numpy` for data handling
  - `matplotlib`, `seaborn` for basic EDA
  - `lightgbm`, `catboost`, `tabnet` for modeling
  - `sklearn` for evaluation and metrics
- **Tools**:
  - Jupyter Notebook 📓 for modeling
  - Kaggle Kernels / Colab for experimentation

---

📌 *This project demonstrates the power of ensembling and model optimization in achieving high-performance predictive modeling under AUC evaluation criteria.*
<!--## 🚀 Future Improvements
- Hyperparameter tuning with Optuna
- Deep learning model experimentation
- Deployment of a loan approval web app-->


