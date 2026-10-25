"""
Salary Prediction Regressor Training Module
Trains and benchmarks regression models to predict compensation (Average Salary in LPA)
for job postings where salary data is available.
Saves the champion regression bundle to disk.
"""

import os
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


def train_salary_model(data_path="dataset/cleaned_naukri_jobs.csv", model_dir="models"):
    """
    Filters disclosed salary records, trains regression models,
    evaluates R2/MAE/RMSE, and serializes the best model bundle.
    """
    os.makedirs(model_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    # Disclosed salary subset (exclude extreme unverified outliers > 80 LPA)
    sal_df = df[df["Average Salary"].notna() & (df["Average Salary"] <= 80)].copy()
    print(f"Disclosed salary training records: {len(sal_df)}")

    features = [
        "Role Searched",
        "Primary Location",
        "Average Experience",
        "Rating",
        "Reviews",
        "Skill Count",
        "Skills"
    ]
    target = "Average Salary"

    X = sal_df[features].copy()
    y = sal_df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    num_features = ["Average Experience", "Rating", "Reviews", "Skill Count"]
    cat_features = ["Role Searched", "Primary Location"]

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    column_preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features)
        ],
        remainder="drop"
    )

    tfidf_vectorizer = TfidfVectorizer(
        max_features=150,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_proc = column_preprocessor.fit_transform(X_train)
    X_test_proc = column_preprocessor.transform(X_test)

    X_train_skills = tfidf_vectorizer.fit_transform(X_train["Skills"].fillna(""))
    X_test_skills = tfidf_vectorizer.transform(X_test["Skills"].fillna(""))

    X_train_final = hstack([X_train_proc, X_train_skills])
    X_test_final = hstack([X_test_proc, X_test_skills])

    regressors = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=250, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    }

    results = []
    trained = {}

    for name, reg in regressors.items():
        reg.fit(X_train_final, y_train)
        preds = reg.predict(X_test_final)
        trained[name] = reg

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)

        results.append({
            "Regressor": name,
            "R2 Score": round(r2, 4),
            "MAE (LPA)": round(mae, 2),
            "RMSE (LPA)": round(rmse, 2)
        })

    benchmark_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
    print("\n" + "="*60)
    print("SALARY REGRESSION BENCHMARK RESULTS")
    print("="*60)
    print(benchmark_df.to_string(index=False))

    best_name = benchmark_df.iloc[0]["Regressor"]
    best_model = trained[best_name]
    print(f"\nChampion Regressor Selected: {best_name}")

    salary_bundle = {
        "model": best_model,
        "model_name": best_name,
        "column_preprocessor": column_preprocessor,
        "tfidf": tfidf_vectorizer,
        "benchmark": benchmark_df.to_dict(orient="records")
    }

    bundle_path = os.path.join(model_dir, "salary_predictor_bundle.joblib")
    joblib.dump(salary_bundle, bundle_path)
    print(f"Salary model bundle saved to: {bundle_path}")

    return salary_bundle


if __name__ == "__main__":
    train_salary_model()
