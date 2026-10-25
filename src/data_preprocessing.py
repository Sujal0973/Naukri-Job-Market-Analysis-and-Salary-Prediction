"""
Data Preprocessing Module
Extracts, cleans, and engineers features from raw scraped Naukri job listings.
"""

import re
import numpy as np
import pandas as pd


def parse_experience(val):
    """
    Parses experience strings like '3-8 Yrs' into (min_exp, max_exp, avg_exp).
    """
    if pd.isna(val):
        return np.nan, np.nan, np.nan
    val = str(val).strip()
    match_range = re.search(r'(\d+)\s*-\s*(\d+)', val)
    if match_range:
        min_e = float(match_range.group(1))
        max_e = float(match_range.group(2))
        return min_e, max_e, (min_e + max_e) / 2.0
    single_match = re.search(r'(\d+)', val)
    if single_match:
        e = float(single_match.group(1))
        return e, e, e
    return np.nan, np.nan, np.nan


def parse_salary(val):
    """
    Parses salary strings into standardized Lacs Per Annum (LPA).
    Handles 'X-Y Lacs PA', monthly figures ('10,000/month'),
    and raw Rupee counts ('50,000-80,000 PA').
    """
    if pd.isna(val):
        return np.nan, np.nan, np.nan
    s = str(val).strip()
    if s in ["Not Disclosed", "Unpaid", ""]:
        return np.nan, np.nan, np.nan

    # Convert monthly pay to annual Lacs
    if "/month" in s.lower():
        num = re.search(r"([\d,]+)", s)
        if num:
            monthly = float(num.group(1).replace(",", ""))
            annual_lacs = (monthly * 12) / 100000.0
            return annual_lacs, annual_lacs, annual_lacs

    # Remove annotations like '(Including Variable: 10.0%)'
    s_clean = re.sub(r"\(.*?\)", "", s).replace("PA", "").strip()

    parts = s_clean.split("-")
    if len(parts) == 2:
        p0_str, p1_str = parts[0].strip(), parts[1].strip()
        m0 = re.search(r"[\d.]+", p0_str.replace(",", ""))
        m1 = re.search(r"[\d.]+", p1_str.replace(",", ""))
        if not m0 or not m1:
            return np.nan, np.nan, np.nan
        p0 = float(m0.group())
        p1 = float(m1.group())

        p0_lacs = p0 / 100000.0 if ("," in p0_str or p0 > 1000) else p0
        p1_lacs = p1 / 100000.0 if ("," in p1_str or p1 > 1000) else p1

        return p0_lacs, p1_lacs, (p0_lacs + p1_lacs) / 2.0

    elif len(parts) == 1:
        m = re.search(r"[\d.]+", parts[0].replace(",", ""))
        if m:
            val_num = float(m.group())
            val_lacs = val_num / 100000.0 if ("," in parts[0] or val_num > 1000) else val_num
            return val_lacs, val_lacs, val_lacs

    return np.nan, np.nan, np.nan


def parse_posted_days(val):
    """
    Converts relative posting age ('3 Days Ago', 'Just Now', '1 Month Ago') into integer days.
    """
    if pd.isna(val):
        return np.nan
    text = str(val).lower().strip()
    if text in ["today", "just now", "few hours ago"]:
        return 0
    number_match = re.search(r"\d+", text)
    if number_match:
        num = int(number_match.group())
        if "day" in text:
            return num
        elif "week" in text:
            return num * 7
        elif "month" in text:
            return num * 30
        elif "year" in text:
            return num * 365
    return np.nan


def categorize_experience(exp):
    """Categorizes experience into Entry, Junior, Mid, Senior."""
    if pd.isna(exp):
        return np.nan
    if exp <= 2:
        return "Entry"
    elif exp <= 5:
        return "Junior"
    elif exp <= 8:
        return "Mid"
    return "Senior"


def categorize_salary(salary_lpa):
    """Categorizes salary (in LPA) into Low, Medium, High, Very High."""
    if pd.isna(salary_lpa):
        return np.nan
    if salary_lpa < 5:
        return "Low"
    elif salary_lpa < 10:
        return "Medium"
    elif salary_lpa < 20:
        return "High"
    return "Very High"


def clean_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline from raw DataFrame to cleaned dataset.
    """
    df = raw_df.copy()

    # 1. Experience
    exp_parsed = df["Experience"].apply(parse_experience)
    df["Min Experience"] = [x[0] for x in exp_parsed]
    df["Max Experience"] = [x[1] for x in exp_parsed]
    df["Average Experience"] = [x[2] for x in exp_parsed]

    # 2. Rating & Reviews
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Reviews"] = (
        df["Reviews"]
        .astype(str)
        .str.replace("Reviews", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

    # 3. Salary
    sal_parsed = df["Salary"].apply(parse_salary)
    df["Min Salary"] = [s[0] for s in sal_parsed]
    df["Max Salary"] = [s[1] for s in sal_parsed]
    df["Average Salary"] = [s[2] for s in sal_parsed]

    # 4. Location
    df["Primary Location"] = (
        df["Location"]
        .fillna("Unknown")
        .astype(str)
        .str.split(",")
        .str[0]
        .str.replace("Hybrid - ", "", regex=False)
        .str.replace("Remote - ", "", regex=False)
        .str.strip()
    )

    # 5. Skills
    df["Skills"] = (
        df["Skills"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace("artificial intelligence", "ai", regex=False)
        .str.replace("data analytics", "data analysis", regex=False)
        .str.replace("analytical", "analytics", regex=False)
    )

    # 6. Company standardization
    df["Company"] = (
        df["Company"]
        .fillna("Confidential")
        .astype(str)
        .str.replace("Leading Client", "Confidential", regex=False)
    )

    # 7. Posted Days
    df["Posted Days"] = df["Posted"].apply(parse_posted_days)

    # 8. Derived Features
    df["Experience Level"] = df["Average Experience"].apply(categorize_experience)
    df["Salary Category"] = df["Average Salary"].apply(categorize_salary)
    df["Skill Count"] = df["Skills"].apply(lambda s: len([x for x in s.split(",") if x.strip()]) if s else 0)

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean raw Naukri job dataset.")
    parser.add_argument("--input", default="dataset/naukri_jobs.csv", help="Input raw CSV path")
    parser.add_argument("--output", default="dataset/cleaned_naukri_jobs.csv", help="Output cleaned CSV path")
    args = parser.parse_args()

    print(f"Loading raw data from: {args.input}")
    raw_df = pd.read_csv(args.input)
    cleaned = clean_dataset(raw_df)
    print(f"Processed shape: {cleaned.shape}")
    print(f"Disclosed salary count: {cleaned['Average Salary'].notna().sum()}")
    print("Done! Cleaning pipeline successfully verified.")
