# 🩺 Diabetes Risk Predictor

A machine learning web app that estimates diabetes risk from basic health measurements, built end to end: data cleaning, model comparison, and deployment.

🔗 **Live demo:** - https://diabetes--risk--predictor.streamlit.app/

---

## 📌 Problem

Early detection of diabetes matters, but many people are not screened until symptoms appear. This project shows how machine learning can work as a quick **screening aid**, flagging people who may need a medical check-up.

## 📊 Dataset

**Pima Indians Diabetes Dataset**: 768 patients with 8 health features and a diabetes outcome.

| Feature | Meaning |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Blood sugar level |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Skin fold thickness (mm) |
| Insulin | Insulin level in blood |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Score of diabetes in family history |
| Age | Age in years |
| **Outcome** | **1 = diabetes, 0 = no diabetes (target)** |

## 🛠️ Approach

1. **Data cleaning:** The dataset had no empty cells, but zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are biologically impossible. I treated them as hidden missing values (for example, 374 in Insulin) and filled them with the **median**, since the median is not pulled up by extreme values.
2. **Exploratory analysis:** Class distribution (about 65% no diabetes, 35% diabetes), histograms, and a correlation heatmap. Glucose (0.49) and BMI (0.31) had the strongest correlation with the outcome.
3. **Train/test split:** 80/20, stratified so both sets keep the same class ratio.
4. **Models compared:** Logistic Regression and Random Forest (200 trees, balanced class weights).
5. **Evaluation:** Accuracy, precision, recall, F1 and a confusion matrix, not accuracy alone, because of the class imbalance.
6. **Deployment:** Streamlit app hosted on Streamlit Community Cloud.

## 📈 Results (test set, 154 patients)

| Model | Accuracy | Precision (diabetes) | Recall (diabetes) | F1 (diabetes) |
|---|---|---|---|---|
| Logistic Regression | 0.70 | 0.59 | 0.50 | 0.54 |
| **Random Forest** | **0.74** | **0.62** | **0.69** | **0.65** |

**Why Random Forest:** In screening, missing a diabetic patient is worse than a false alarm, so recall on the diabetes class matters most. Random Forest raised recall from 0.50 to 0.69.

**Top features:** Glucose and BMI were the most important features, matching what the correlation heatmap suggested.

## 🖥️ App Features

- Sidebar sliders for all 8 health inputs
- Risk level (Low / Moderate / High) with a probability and progress bar
- Feature importance chart
- Model performance and limitations shown in an "About the Model" tab

## 📸 Screenshots

<img width="1917" height="963" alt="image" src="https://github.com/user-attachments/assets/f045b7dc-2d10-4cdf-9c29-dca8b3c1b4fc" />


## 🧰 Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, Streamlit

## 📁 Project Structure

```
diabetes_prediction/
├── app.py                 # Streamlit web app
├── diabetes.ipynb         # EDA, cleaning, training, evaluation
├── diabetes_model.pkl     # Trained Random Forest model
├── feature_columns.pkl    # Feature order used by the model
├── medians.pkl            # Median values used for imputation
├── requirements.txt       # Libraries needed to run the app
└── README.md
```

## ▶️ Run Locally

```
git clone https://github.com/AanchalChauhan-356/diabetes_prediction.git
cd diabetes_prediction
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Limitations

- Small dataset (768 patients), all women of Pima heritage aged 21 and above, so results may not generalise to other groups.
- Median imputation reduces the spread of Insulin values. A method like KNN imputation could work better.
- Results come from a single train/test split. Cross-validation would give a more reliable estimate.
- The probability shows how strongly the trees agree, not an exact medical chance.
- This is a learning project and **not a medical tool**. Please consult a doctor for any health concern.

## 🚀 Future Improvements

- Cross-validation and hyperparameter tuning
- Try XGBoost and compare with Random Forest
- Add SHAP explanations for each prediction
- Try KNN imputation for missing values

## 👤 Author

**Aanchal Chauhan**
GitHub: [AanchalChauhan-356](https://github.com/AanchalChauhan-356)
