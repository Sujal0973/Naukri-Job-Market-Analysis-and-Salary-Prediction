# 📊 Power BI Dashboard Architecture & User Guide

This guide provides complete documentation for the Power BI report file (`dashboard/Dashboard.pbix`), its visual layout, DAX calculations, and the machine-learning-powered **Key Influencers** AI visual.

---

## 1. Dashboard Overview

The **Data Science Job Market Dashboard** was engineered to synthesize 4,136 live job listings into strategic intelligence for job seekers, hiring directors, and talent acquisition teams.

![Power BI Dashboard](../screenshots/15_powerbi_dashboard.png)

---

## 2. Core Metrics & KPI Cards

| KPI Card | Metric Value | Analytical Context |
| :--- | :---: | :--- |
| **Avg Experience** | **5.90 Yrs** | Employers predominantly target mid-senior candidates capable of driving autonomous delivery. |
| **Average Rating** | **3.61 / 5.0** | Overall employee satisfaction among active hiring organizations across India. |
| **Total Locations**| **212 Cities** | Geographic spread across tier-1 hubs, tier-2 emerging centers, and remote postings. |
| **Total Companies**| **~2,000 Firms**| High market fragmentation with both tech titans and emerging AI startups actively hiring. |
| **Total Jobs** | **4,136 Posts** | Total curated listings across all 6 data roles. |

---

## 3. Visual Breakdown & Insights

### A. Job Distribution by Role
- **Visual Type**: Column Bar Chart
- **Findings**: **Data Scientist** recorded the highest demand (~1,000 listings), followed by **Machine Learning Engineer** (~850) and **Data Analyst** (~800). **AI Engineer** represents an emerging niche (~350 postings) with rapid month-over-month growth.

### B. Hiring Activity by Posting Age
- **Visual Type**: Trend Line Chart
- **Findings**: Hiring activity shows a pronounced peak around **20 to 21 days**. Postings remain active on the platform for 3 to 4 weeks before candidate shortlisting concludes.

### C. Top Hiring Companies
- **Visual Type**: Horizontal Bar Chart
- **Key Organizations**:
  1. **Accenture**: Dominates volume hiring with over 330 listings.
  2. **Confidential / Staffing Agencies**: High demand for contract-to-hire tech talent.
  3. **Tata Consultancy Services (TCS)**: Major enterprise analytics initiatives.
  4. **Infosys**: Strong cloud & AI practice expansions.

### D. Top Hiring Locations
- **Visual Type**: Horizontal Bar Chart
- **Hub Ranking**:
  1. **Bengaluru**: Undisputed tech capital with >1,100 listings (>25% market share).
  2. **Hyderabad**: Strong secondary hub with pharmaceutical and enterprise tech hubs.
  3. **Pune & Gurugram**: Major automotive, fintech, and enterprise analytics hubs.

### E. Most Demanded Skills (Word Cloud)
- **Visual Type**: Skill Cloud / Frequency Tag Cloud
- **Top Frequencies**: `Python`, `SQL`, `Machine Learning`, `Data Analysis`, `Deep Learning`, `NLP`, `Power BI`, `AWS`, `Spark`.

---

## 4. Power BI AI Visual: Key Influencers Deep Dive

The dashboard incorporates Power BI's built-in Machine Learning visual (**Key Influencers**) to uncover what factors cause a job posting to be categorized as a specific role.

![Key Influencers Demo](../screenshots/16_prediction_demo.png)

### Key Influencer Analysis for "Business Analyst":
1. **Primary Location is Mumbai Suburban**: Increases the probability of the role being Business Analyst by **4.05x** (due to heavy concentration of banking, financial services, and insurance - BFSI corporate offices).
2. **Company Rating is 0 - 1.3**: Increases probability by **3.56x** (early-stage consulting or boutique recruiting firms).
3. **Primary Location is Jaipur**: Increases likelihood by **2.92x**.
4. **Average Experience is 0 - 1.5 Yrs**: Increases probability by **2.79x** (strong demand for entry-level business and operations analysts).

---

## 5. Interactive Slicers & Filters

The dashboard provides four top-level dynamic slicers to support multi-dimensional drill-downs:
1. **Locations**: Multi-select dropdown filtering jobs by specific cities.
2. **Job Role**: Filter between AI Engineer, Business Analyst, Data Analyst, Data Engineer, Data Scientist, ML Engineer.
3. **Hiring Status**: Filter by freshness and posting recency.
4. **Experience**: Filter by required candidate experience bands.
