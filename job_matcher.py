import os
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
from pydantic import BaseModel, Field
from jobspy import scrape_jobs
from google import genai
from google.genai import types

# CUstomise Profile
SEARCH_TERMS = ["Operations Analyst", "Global Trade Compliance", "Supply Chain Analyst", "Marketing Analyst", "GTM", "Business Analyst", "Growth", "Business Management Associate", "Strategy", "SAP"]
COUNTRY_INDEED = "India"
RESULTS_PER_TERM = 15
MIN_SCORE = 8  

SEARCH_TARGETS = [
    {"location": "Bengaluru, Karnataka, India", "is_remote": False, "label": "Bengaluru"},
    {"location": "India", "is_remote": True, "label": "Remote"},
]

MY_CV = """
Operations Analyst with hands-on experience in trade compliance, inventory optimization, and data-driven process improvement.
Skilled in applying quantitative modeling (queuing theory, simulation, RFID/IoT systems) and BI tools to reduce operational inefficiencies
and support procurement decision-making.
CORE COMPETENCIES
Purchase Order Management | Vendor Onboarding & Supplier Relations | SAP GTS & SAP E4H | Procurement Compliance | KPI
Monitoring & Benchmarking | Process Standardization | Commodity Strategy Execution | Cross-Functional Collaboration | Data Analysis
(Python, Power BI, Tableau) | IoT & RFID Supply Chain Solutions | Global sanctioned party list (SPL) screening | SEO Optimization
PROFESSIONAL EXPERIENCE
Operations Analyst – Trade Compliance, Global Trade | HP Inc. Dec 2025 – Present
● Sustained a 100% compliance rate across high-volume global import/export transactions by identifying targeted regulatory risk.
● Eliminated processing backlogs and reduced delay risk across HP's international trade portfolio through proactive case triage.
● Leveraged SAP GTS and E4H platforms to conduct real-time compliance screening across the complete global trade life cycles.
● Streamlined trade workflow executions in SAP GTS and E4H, standardizing fulfillment procedures across cross-border activities.
● Reduced processing errors and significantly accelerated turnaround time for cross-border purchases and shipment transactions.
● Collaborated with cross-functional logistics and procurement teams to proactively resolve trade documentation discrepancies.
● Managed SPL screenings for 100% of global trade, securing strict regulatory compliance across all sales orders and deliveries.
Operations & Marketing Intern | Aditya Birla Fashion and Retail Ltd. (Pantaloons) Jun 2024 – Aug 2024
● Analyzed the warehouse layout, RFID tagging economics, and workflow bottlenecks to identify critical operational inefficiency.
● Designed IoT-enabled inventory framework projecting overall systems accuracy gains from 85% to 98% in retail environments.
● Projected order fulfillment time reduction from 48 to 38 hours through implementation of real-time inventory tracking design.
● Conducted a highly structured market research and detailed customer feedback surveys across 3 primary product categories.
● Generated actionable supplier and consumer insights that informed vendor selection criteria and key procurement decisions.
● Assisted in executing targeted digital marketing campaigns tied to promotional procurement cycles to maximize brand reach.
● Contributed to measurable gains in overall product visibility and in-store customer engagement via optimized retail displays.
EDUCATION
MBA — Operations Management & Marketing | Assam University B.Sc. Physics (Minor: Mathematics & Computer Science) | Assam University KEY PROJECTS
Jun 2023 – Jun 2025
Jun 2019 – Oct 2022
IoT-Based Warehouse Inventory Management | Pantaloons July 2024 – Aug 2024
● Designed and developed an RFID-based warehouse inventory tracking framework during the 60-day internship at Pantaloons.
● Conducted cost-benefit analysis of passive (₹4–₹85/tag) vs. active (₹1,650–₹2,500/tag) RFID tagging to guide optimal selection.
● Successfully applied the Clark & Wright savings algorithm to optimize picking sequences across a new 3-tier, 5.5m rack system.
● Accurately modeled projected gains: overall inventory accuracy from 85% to 98% and order fulfillment time from 48to 38 hours.
● Correctly modeled a 30% reduction in overall equipment downtime and a 40% improvement in maintenance record accuracy.
Vehicle Flow Optimization for Urban Fuel Stations | Research Project Oct 2024 – Apr 2025
● Accurately recorded peak-hour arrival rates of 28–38 vehicles/hour and observed queue lengths of up to 15 vehicles per station.
● Applied queuing theory (M/M/1, M/M/c) to accurately compute utilization (ρ = 0.64 – 0.875) and wait probabilities (62 - 85%).
● Built a SimPy-based event simulation model to automate the testing of lane segregation, staggered entry, and staff-guided flow.
● The optimized simulation cut the average customer wait time by 30 –45% and raised pump utilization from ~70% to over 90%.
● Achieved a remarkable 44.4% wait-time reduction at the highest congested station through strategic combined interventions.
Digital Content Creator | YouTube Mar 2023 – Present
● Scaled channel to 10M+ lifetime views across 314 videos, capturing 645K+ views and 3.4M+ impressions in a single peak month.
● Achieved a highly competitive 5.18% Click-Through Rate across millions of organic impressions via strategic video A/B testing.
● Directed the internal production pipeline, including market research, scripting, and SEO optimization for a 314-video library.
● Developed audience engagement techniques that secured 3,631 hours of watch time, boosting platform distribution metrics.
● Managed end-to-end production pipelines: data-driven topic research, scriptwriting, video editing, plus post-launch analytics.
SKILLS & TOOLS
Programming & Analytics: Python, MATLAB, Scilab, HTML, JavaScript, Power BI, Tableau, Google Analytics | Enterprise Systems: SAP
GTS, SAP E4H
"""

# --- 2. STRUCTURED OUTPUT SCHEMA ---
class JobMatch(BaseModel):
    score: int = Field(description="Fit score from 1 to 10 based strictly on CV match.")
    reason: str = Field(description="One concise sentence explaining why it fits.")
    missing_skills: list[str] = Field(description="Key requirements missing from the CV.")

# LAST 24 HOURS Jobs
def fetch_recent_jobs():
    all_jobs = []
    sites = ["indeed", "linkedin", "google", "naukri"]

    for term in SEARCH_TERMS:
        for target in SEARCH_TARGETS:
            print(f"Scraping '{term}' ({target['label']})...")
            for site in sites:
                try:
                    df = scrape_jobs(
                        site_name=[site],
                        search_term=term,
                        location=target["location"],
                        is_remote=target["is_remote"],
                        country_indeed=COUNTRY_INDEED,
                        results_wanted=RESULTS_PER_TERM,
                        hours_old=24,
                        fetch_description=True,
                    )
                    if df is not None and not df.empty:
                        all_jobs.append(df)
                except Exception as e:
                    print(f"Skipping {site} for '{term}' ({target['label']}): {e}")

    if not all_jobs:
        return pd.DataFrame()
    combined = pd.concat(all_jobs, ignore_index=True)
    return combined.drop_duplicates(subset=["job_url"])

# --- SCORE JOBS---
def evaluate_jobs(jobs_df):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    matches = []

    for _, row in jobs_df.iterrows():
        description = str(row.get("description", ""))
        if len(description) < 100:
            continue

        prompt = f"""
        Evaluate this job against my CV. Score 1-10 strictly on required experience,
        domain alignment, and skills. Penalize roles requiring mandatory skills I lack.
        Note: I am only looking for roles based in Bengaluru/Bangalore or Remote.

        MY CV:
        {MY_CV}

        JOB TITLE: {row.get('title')}
        COMPANY: {row.get('company')}
        LOCATION: {row.get('location')}
        DESCRIPTION:
        {description[:4000]}
        """
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=JobMatch,
                    temperature=0.1,
                ),
            )
            result = JobMatch.model_validate_json(response.text)
            if result.score >= MIN_SCORE:
                matches.append({
                    "score": result.score,
                    "title": row.get("title"),
                    "company": row.get("company"),
                    "location": row.get("location"),
                    "url": row.get("job_url"),
                    "reason": result.reason,
                    "gaps": ", ".join(result.missing_skills) or "None",
                })
            time.sleep(4)  # Keeps requests safely within free-tier RPM limits
        except Exception as e:
            print(f"Scoring error on {row.get('title')}: {e}")

    return sorted(matches, key=lambda x: x["score"], reverse=True)

# Send Email
def send_email(matches):
    if not matches:
        print("No high-scoring matches today.")
        return

    sender = os.environ["EMAIL_USER"]
    password = os.environ["EMAIL_PASS"]

    html_items = ""
    for m in matches:
        html_items += f"""
        <div style="margin-bottom:20px; padding:12px; border:1px solid #ddd; border-radius:6px;">
            <h3 style="margin:0;">[{m['score']}/10] {m['title']} — {m['company']}</h3>
            <p style="margin:4px 0; color:#555;">Location: {m['location']}</p>
            <p style="margin:6px 0;"><strong>Why it fits:</strong> {m['reason']}</p>
            <p style="margin:6px 0;"><strong>Gaps to note:</strong> {m['gaps']}</p>
            <a href="{m['url']}" style="display:inline-block; margin-top:6px; padding:8px 12px; background:#1a73e8; color:#fff; text-decoration:none; border-radius:4px;">Apply Now</a>
        </div>
        """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Daily Job Matches: {len(matches)} Strong Fits (Bengaluru & Remote)"
    msg["From"] = sender
    msg["To"] = sender
    msg.attach(MIMEText(f"<h2>Your Top CV Matches (Last 24h)</h2>{html_items}", "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, sender, msg.as_string())
    print(f"Emailed {len(matches)} matches.")

if __name__ == "__main__":
    jobs = fetch_recent_jobs()
    print(f"Total unique jobs fetched: {len(jobs)}")
    if not jobs.empty:
        top_matches = evaluate_jobs(jobs)
        send_email(top_matches)
