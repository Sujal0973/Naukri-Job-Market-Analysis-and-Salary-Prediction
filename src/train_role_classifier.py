"""
Job Role Multi-Class Classifier Training Module
Trains and benchmarks multiple classification models to predict the target job role
from job requirements, skills, experience, and metadata.
Saves the champion model (XGBoost) and preprocessing pipelines to disk.
"""

import os
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def build_feature_pipeline():
    """Builds numerical and categorical feature pipelines."""
    num_features = ["Average Experience", "Rating", "Reviews", "Skill Count", "Posted Days"]
    cat_features = ["Primary Location"]

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
        max_features=500,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english"
    )

    return column_preprocessor, tfidf_vectorizer


def train_models(data_path="dataset/cleaned_naukri_jobs.csv", model_dir="models"):
    """
    Trains all classification models, outputs benchmark metrics,
    and serializes the champion model bundle to `model_dir`.
    """
    os.makedirs(model_dir, exist_ok=True)
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    features = [
        "Primary Location",
        "Average Experience",
        "Rating",
        "Reviews",
        "Skill Count",
        "Posted Days",
        "Skills"
    ]
    target = "Role Searched"

    X = df[features].copy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    column_preprocessor, tfidf = build_feature_pipeline()

    # Fit transformers
    X_train_proc = column_preprocessor.fit_transform(X_train)
    X_test_proc = column_preprocessor.transform(X_test)

    X_train_skills = tfidf.fit_transform(X_train["Skills"].fillna(""))
    X_test_skills = tfidf.transform(X_test["Skills"].fillna(""))

    X_train_final = hstack([X_train_proc, X_train_skills])
    X_test_final = hstack([X_test_proc, X_test_skills])

    print(f"Training shape: {X_train_final.shape}, Testing shape: {X_test_final.shape}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=20, min_samples_split=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42
        )
    }

    results = []
    trained = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        trained[name] = model

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4)
        })

    benchmark_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    print("\n" + "="*60)
    print("MODEL BENCHMARK RESULTS")
    print("="*60)
    print(benchmark_df.to_string(index=False))

    best_name = benchmark_df.iloc[0]["Model"]
    best_model = trained[best_name]
    print(f"\nChampion Model Selected: {best_name}")

    # Evaluate champion model
    y_pred_best = best_model.predict(X_test_final)
    print("\nClassification Report (Champion Model):")
    print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_.astype(str)))

    # Save artifacts bundle
    model_bundle = {
        "model": best_model,
        "model_name": best_name,
        "column_preprocessor": column_preprocessor,
        "tfidf": tfidf,
        "label_encoder": label_encoder,
        "target_classes": label_encoder.classes_.tolist(),
        "benchmark": benchmark_df.to_dict(orient="records")
    }

    bundle_path = os.path.join(model_dir, "role_classifier_bundle.joblib")
    joblib.dump(model_bundle, bundle_path)
    print(f"\nArtifact saved successfully to: {bundle_path}")

    return model_bundle


if __name__ == "__main__":
    train_models()
