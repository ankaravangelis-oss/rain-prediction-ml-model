# End-to-End Supervised Machine Learning Pipeline for Rain Prediction

This project presents a supervised machine learning pipeline designed to predict next-day precipitation (`RainTomorrow`) using historical meteorological observations from the Australian Commonwealth Bureau of Meteorology.

---

## 📌 Project Architecture

- **Data Cleaning & Missing Value Imputation**: Handling missing values using `SimpleImputer` (median strategy for numerical features, most frequent for categorical variables) and dropping sparse attributes with excessive missingness (`Sunshine`, `Evaporation`).
- **Feature Engineering**: Compressing highly collinear morning (9am) and afternoon (3pm) measurements into aggregate mean indicators (`Ps_mean`, `Temp_mean`, `Ws_mean`, `Humid_mean`, `Cloud_mean`) to reduce redundancy and multicollinearity.
- **Seasonality Decomposition**: Extracting cyclical temporal indicators (`Month_number`) from timestamp records.
- **Leak-Free Pipelines**: Encapsulating preprocessing steps within Scikit-Learn `Pipeline` and `ColumnTransformer` modules to ensure strict isolation between training and validation folds.
- **Baseline Model Benchmarking**: Evaluating 7 classical classification algorithms with default hyperparameter configurations:
  - Gaussian Naive Bayes
  - K-Nearest Neighbors (`KNeighborsClassifier`)
  - Logistic Regression
  - Multi-Layer Perceptron (`MLPClassifier`)
  - Support Vector Classifier (`SVC`)
  - Decision Tree (`DecisionTreeClassifier`)
  - Random Forest (`RandomForestClassifier`)
- **Hyperparameter Optimization**: Conducting exhaustive Grid Search with 5-Fold Stratified Cross-Validation (`StratifiedKFold`), optimizing specifically for the **F1-score** to address target class imbalance and minimize False Negatives.

---

## Key Findings

- **Top Performers**: **MLP** and **Random Forest** achieved the best overall balance between Precision and Recall. MLP demonstrated slightly superior classification accuracy by reducing critical False Negative errors.
- **Class Imbalance Impact**: The natural imbalance between rain and no-rain days introduces model bias toward the majority class (`No`), highlighting the necessity of threshold tuning and metric prioritization (F1-score over simple accuracy).
- **Optimization Gains**: Grid Search yielded substantial performance improvements for algorithms sensitive to hyperparameters, notably **SVC** and **Decision Tree**.

---

## Tech Stack

- **Language**: Python 3
- **Data Manipulation & Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
