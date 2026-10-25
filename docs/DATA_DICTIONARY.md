# 📚 Data Dictionary & Schema Documentation

This document provides schema specifications, data types, missing value analyses, and transformation logic for both the raw scraped dataset and the feature-engineered dataset.

---

## 1. Raw Dataset (`dataset/naukri_jobs.csv`)

Collected via automated web scraping from **Naukri.com** targeting data and analytics roles across major Indian tech hubs.

| Column Name | Raw Data Type | Missing Count | Missing % | Description & Example Values |
| :--- | :--- | :--- | :--- | :--- |
| **`Role Searched`** | `object` | 0 | 0.0% | Job keyword query searched (`Data Scientist`, `Machine Learning Engineer`, `Data Analyst`, `Data Engineer`, `Business Analyst`, `AI Engineer`) |
| **`Job Title`** | `object` | 0 | 0.0% | Complete job title posted by employer (e.g., *"Lead Data Scientist - NLP"*, *"Senior Data Analyst"*) |
| **`Company`** | `object` | 0 | 0.0% | Name of the hiring organization or staffing agency (e.g., *"Accenture"*, *"Tata Consultancy Services"*, *"Leading Client"*) |
| **`Rating`** | `float64` / `object` | 959 | 23.2% | Employer glassdoor/internal rating out of 5.0 (e.g., `4.1`, `3.8`, `NaN`) |
| **`Reviews`** | `object` | 959 | 23.2% | Number of employee reviews submitted (e.g., *"1,452 Reviews"*, *"28 Reviews"*) |
| **`Experience`** | `object` | 52 | 1.3% | Experience span required by employer (e.g., *"3-8 Yrs"*, *"0-2 Yrs"*) |
| **`Salary`** | `object` | 0 | 0.0% | Publicly disclosed compensation. Note: `3,639` listings explicitly state *"Not Disclosed"* (e.g., *"15-30 Lacs PA"*, *"Not Disclosed"*, *"18,000/month"*) |
| **`Location`** | `object` | 0 | 0.0% | City, multiple locations, or workplace mode (e.g., *"Bengaluru"*, *"Hybrid - Pune, Maharashtra"*, *"Remote - Mumbai"*) |
| **`Description`** | `object` | 0 | 0.0% | Brief snippet or highlight summary of the job listing |
| **`Skills`** | `object` | 0 | 0.0% | Comma-separated technical keywords and tags (e.g., *"python, sql, machine learning, power bi"*) |
| **`Posted`** | `object` | 0 | 0.0% | Relative posting timestamp (e.g., *"Just Now"*, *"3 Days Ago"*, *"30+ Days Ago"*) |
| **`Job Link`** | `object` | 0 | 0.0% | URL to the official job listing on Naukri.com |

---

## 2. Cleaned & Engineered Dataset (`dataset/cleaned_naukri_jobs.csv`)

Total rows: **4,136** | Total features: **23**

| Feature Name | Clean Type | Source / Derivation | Description |
| :--- | :--- | :--- | :--- |
| **`Role Searched`** | `string` | Query label | Multi-class target label (6 classes) |
| **`Job Title`** | `string` | Direct | Cleaned title string |
| **`Company`** | `string` | Normalized | Standardized company name (replaced *"Leading Client"* with *"Confidential"*) |
| **`Rating`** | `float64` | Numeric coerce | Company employee review rating (1.0 to 5.0) |
| **`Reviews`** | `float64` | Numeric coerce | Cleaned integer count of employer reviews |
| **`Experience`** | `string` | Direct | Original experience requirement string |
| **`Salary`** | `string` | Direct | Raw salary string |
| **`Location`** | `string` | Direct | Original full location string |
| **`Description`** | `string` | Direct | Summary description |
| **`Skills`** | `string` | Normalized text | Lowercased, alias-normalized technical skills string |
| **`Posted`** | `string` | Direct | Original relative posting age |
| **`Job Link`** | `string` | Direct | Direct link URL |
| **`Min Experience`** | `float64` | Regex `(\d+)` | Minimum years of experience required |
| **`Max Experience`** | `float64` | Regex `-(\d+)` | Maximum years of experience required |
| **`Average Experience`**| `float64` | `(Min + Max) / 2` | Midpoint years of experience required |
| **`Min Salary`** | `float64` | Parsed regex | Minimum annual compensation in Lacs Per Annum (LPA) |
| **`Max Salary`** | `float64` | Parsed regex | Maximum annual compensation in Lacs Per Annum (LPA) |
| **`Average Salary`** | `float64` | `(Min + Max) / 2` | Midpoint annual compensation in LPA |
| **`Posted Days`** | `float64` | Regex time conversion | Integer days elapsed since job was published |
| **`Experience Level`** | `string` | Binned `Average Experience` | `Entry` (<=2 yrs), `Junior` (3-5 yrs), `Mid` (6-8 yrs), `Senior` (>8 yrs) |
| **`Salary Category`** | `string` | Binned `Average Salary` | `Low` (<5 LPA), `Medium` (5-10 LPA), `High` (10-20 LPA), `Very High` (>=20 LPA) |
| **`Skill Count`** | `int64` | Count comma separators | Number of distinct skills tagged on the job card |
| **`Primary Location`** | `string` | Extracted city | First listed primary city, stripped of "Hybrid -" or "Remote -" prefixes |

---

## 3. Missing Value Strategy & Domain Reality

```text
Feature Completeness Breakdown:
Total Listings: 4,136
├── Core Attributes (Role, Company, Location, Skills): 100% Complete
├── Ratings & Reviews: 76.8% Complete (959 Nulls — startups & small firms lack reviews)
├── Experience: 98.7% Complete (52 Nulls)
└── Disclosed Salary: 11.2% Complete (3,673 Undisclosed / Unpaid)
```

### Why Undisclosed Salaries Were Not Imputed
In Indian recruitment platforms (specifically Naukri.com), **88.8% of tech recruiters keep compensation undisclosed** to negotiate flexibly based on candidate seniority and prior CTC.
- **Data Integrity Decision**: Imputing 88.8% of values using mean or median would introduce severe synthetic bias and distortion into statistical analyses.
- **Architectural Solution**:
  1. Market analysis and salary correlations were conducted strictly on the **verified 463 disclosed listings**.
  2. The primary machine learning objective was formulated as **Multi-Class Job Role Classification** (100% complete ground truth labels).
  3. A specialized regression model was developed for salary estimation on disclosed profiles.
