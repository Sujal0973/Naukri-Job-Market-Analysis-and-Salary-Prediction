"""
End-to-End Prediction Inference Engine
Allows users to predict:
1. Best matching Job Role given candidate skills, experience, location, and rating.
2. Estimated Market Salary Range (in Lacs Per Annum - LPA).
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack


def load_artifacts(models_dir="models"):
    """Loads saved classification and regression bundles."""
    role_bundle_path = os.path.join(models_dir, "role_classifier_bundle.joblib")
    salary_bundle_path = os.path.join(models_dir, "salary_predictor_bundle.joblib")

    role_bundle = joblib.load(role_bundle_path) if os.path.exists(role_bundle_path) else None
    salary_bundle = joblib.load(salary_bundle_path) if os.path.exists(salary_bundle_path) else None

    return role_bundle, salary_bundle


def predict_job(
    skills: str,
    experience: float = 3.0,
    location: str = "Bengaluru",
    rating: float = 3.8,
    reviews: float = 500.0,
    role: str = None,
    models_dir: str = "models"
):
    """
    Executes inference for job role classification and salary estimation.
    """
    role_bundle, salary_bundle = load_artifacts(models_dir)

    results = {}

    skill_list = [s.strip().lower() for s in skills.split(",") if s.strip()]
    skill_count = len(skill_list)
    skills_cleaned = ", ".join(skill_list)

    # 1. Job Role Classification
    if role_bundle:
        input_df = pd.DataFrame([{
            "Primary Location": location,
            "Average Experience": float(experience),
            "Rating": float(rating),
            "Reviews": float(reviews),
            "Skill Count": skill_count,
            "Posted Days": 10.0,
            "Skills": skills_cleaned
        }])

        X_proc = role_bundle["column_preprocessor"].transform(input_df)
        X_skills = role_bundle["tfidf"].transform(input_df["Skills"])
        X_final = hstack([X_proc, X_skills])

        pred_idx = role_bundle["model"].predict(X_final)[0]
        predicted_role = role_bundle["label_encoder"].inverse_transform([pred_idx])[0]
        
        # Probabilities if available
        role_probs = {}
        if hasattr(role_bundle["model"], "predict_proba"):
            probs = role_bundle["model"].predict_proba(X_final)[0]
            classes = role_bundle["label_encoder"].classes_
            for c, p in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True):
                role_probs[c] = round(float(p) * 100, 1)

        results["predicted_role"] = predicted_role
        results["role_probabilities"] = role_probs
    else:
        results["predicted_role"] = role or "Unknown"

    target_role = role if role else results.get("predicted_role", "Data Scientist")

    # 2. Salary Prediction
    if salary_bundle:
        sal_input_df = pd.DataFrame([{
            "Role Searched": target_role,
            "Primary Location": location,
            "Average Experience": float(experience),
            "Rating": float(rating),
            "Reviews": float(reviews),
            "Skill Count": skill_count,
            "Skills": skills_cleaned
        }])

        X_sal_proc = salary_bundle["column_preprocessor"].transform(sal_input_df)
        X_sal_skills = salary_bundle["tfidf"].transform(sal_input_df["Skills"])
        X_sal_final = hstack([X_sal_proc, X_sal_skills])

        predicted_salary_lpa = float(salary_bundle["model"].predict(X_sal_final)[0])
        predicted_salary_lpa = max(1.5, round(predicted_salary_lpa, 2))

        # Reasonable market range (+/- 15%)
        lower_bound = round(max(1.0, predicted_salary_lpa * 0.85), 1)
        upper_bound = round(predicted_salary_lpa * 1.15, 1)

        results["predicted_salary_lpa"] = predicted_salary_lpa
        results["salary_range_lpa"] = f"{lower_bound} - {upper_bound} LPA"
        results["salary_inr_annual"] = f"INR {int(predicted_salary_lpa * 100000):,}"
    else:
        results["predicted_salary_lpa"] = None

    return results


def print_report(res, experience, location, skills, specified_role=None):
    """Prints a styled CLI summary."""
    print("\n" + "="*65)
    print("      NAUKRI JOB MARKET INTELLIGENCE PREDICTION REPORT      ")
    print("="*65)
    print(f"Candidate Profile:")
    print(f"  * Experience : {experience} Years")
    print(f"  * Location   : {location}")
    print(f"  * Skills     : {skills}")
    if specified_role:
        print(f"  * Given Role : {specified_role}")
    print("-" * 65)

    if "predicted_role" in res:
        print(f"Predicted Best-Fit Role : {res['predicted_role'].upper()}")
        if res.get("role_probabilities"):
            print("Role Match Confidences:")
            for role_name, prob in list(res["role_probabilities"].items())[:4]:
                bar = "#" * int(prob / 5)
                print(f"  - {role_name:<26} : {prob:>5.1f}% | {bar}")

    if res.get("predicted_salary_lpa") is not None:
        print("-" * 65)
        print(f"Estimated Market Salary : {res['predicted_salary_lpa']} LPA (~{res['salary_inr_annual']} / yr)")
        print(f"Estimated Salary Range  : {res['salary_range_lpa']}")

    print("="*65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict job role fit and estimated salary range.")
    parser.add_argument("--skills", type=str, default="python, sql, machine learning, pandas, power bi", help="Comma-separated skills")
    parser.add_argument("--experience", type=float, default=4.0, help="Years of relevant experience")
    parser.add_argument("--location", type=str, default="Bengaluru", help="City / Primary location")
    parser.add_argument("--role", type=str, default=None, help="Target job role (optional)")
    parser.add_argument("--rating", type=float, default=3.8, help="Target company rating (1-5)")
    parser.add_argument("--reviews", type=float, default=500.0, help="Target company reviews count")

    args = parser.parse_args()

    res = predict_job(
        skills=args.skills,
        experience=args.experience,
        location=args.location,
        rating=args.rating,
        reviews=args.reviews,
        role=args.role
    )

    print_report(res, args.experience, args.location, args.skills, args.role)
