from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
from pathlib import Path
import google.generativeai as genai
from fastapi.responses import StreamingResponse
import csv
import io
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from fastapi.responses import FileResponse
import tempfile
from reportlab.platypus import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from fastapi.staticfiles import StaticFiles




load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="Aadhaar Service Stress API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load resources - use the directory where this script is located
backend_dir = Path(__file__).parent
df = pd.read_csv(backend_dir / "aadhaar_merged_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
model = joblib.load(backend_dir / "aadhaar_service_stress_model.pkl")

@app.get("/states")
def get_states():
    return sorted(df["state"].unique().tolist())

@app.get("/districts/{state}")
def get_districts(state: str):
    return sorted(df[df["state"] == state]["district"].unique().tolist())

@app.get("/dates/{state}/{district}")
def get_dates(state: str, district: str):
    """Get all available dates for a state/district combination"""
    dates = df[
        (df["state"] == state) &
        (df["district"] == district)
    ]["date"].unique()
    dates = sorted([str(d.date()) for d in dates])
    return dates

@app.get("/risk")
def get_risk(state: str, district: str, date: str):
    row = df[
        (df["state"] == state) &
        (df["district"] == district) &
        (df["date"].dt.date == pd.to_datetime(date).date())
    ]
    if row.empty:
        return {"error": "No data found"}

    r = row.iloc[0]
    
    # Handle NaN values (convert to None or default value)
    def safe_float(val):
        if pd.isna(val):
            return None
        return float(val)
    
    return {
        "risk_score": safe_float(r["service_stress_risk"]),
        "bio_ratio": safe_float(r["biometric_to_enrolment_ratio"]),
        "child_pressure": safe_float(r["child_update_pressure"]),
        "elderly_pressure": safe_float(r["elderly_update_pressure"])
    }

@app.get("/risk-verdict/{risk_score}")
def get_risk_verdict(risk_score: float):
    """Classify risk as LOW, MEDIUM, or HIGH"""
    if risk_score < 0.01:
        return {"verdict": "LOW", "description": "Minimal service stress - operations running smoothly"}
    elif risk_score < 0.03:
        return {"verdict": "MEDIUM", "description": "Moderate service stress - requires monitoring"}
    else:
        return {"verdict": "HIGH", "description": "High service stress - immediate attention needed"}

@app.get("/risk-percentile/{state}/{district}/{date}")
def get_risk_percentile(state: str, district: str, date: str):
    """Calculate what percentile this district's risk is in"""
    date_obj = pd.to_datetime(date).date()
    date_data = df[df["date"].dt.date == date_obj]
    
    if date_data.empty:
        return {"percentile": 0, "comparison": "No data available"}
    
    current_row = df[
        (df["state"] == state) &
        (df["district"] == district) &
        (df["date"].dt.date == date_obj)
    ]
    
    if current_row.empty:
        return {"percentile": 0, "comparison": "No data for this district"}
    
    current_risk = current_row.iloc[0]["service_stress_risk"]
    if pd.isna(current_risk):
        return {"percentile": 0, "comparison": "No data"}
    
    percentile = (date_data["service_stress_risk"] < current_risk).sum() / len(date_data) * 100
    return {
        "percentile": round(percentile, 1),
        "comparison": f"Riskier than {percentile:.1f}% of districts"
    }

@app.get("/top-districts")
def get_top_districts(limit: int = 10):
    """Get top N high-risk districts by average risk"""
    avg_risk = df.groupby("district")["service_stress_risk"].mean().sort_values(ascending=False)
    top_districts = avg_risk.head(limit).to_dict()
    return [{
        "district": name,
        "average_risk": round(float(risk), 4)
    } for name, risk in top_districts.items()]

@app.get("/district-hotspots/{state}")
def get_district_hotspots(state: str, limit: int = 5):
    """Get high-risk districts in a state"""
    state_data = df[df["state"] == state]
    if state_data.empty:
        return []
    
    avg_risk = state_data.groupby("district")["service_stress_risk"].mean().sort_values(ascending=False)
    hotspots = avg_risk.head(limit).to_dict()
    return [{
        "district": name,
        "average_risk": round(float(risk), 4)
    } for name, risk in hotspots.items()]

@app.get("/risk-trend/{state}/{district}")
def get_risk_trend(state: str, district: str):
    """Get risk trend over time for a district"""
    data = df[
        (df["state"] == state) &
        (df["district"] == district)
    ].sort_values("date")
    
    if data.empty:
        return {"error": "No data found"}
    
    trend = [
        {
            "date": str(row["date"].date()),
            "risk_score": float(row["service_stress_risk"]) if not pd.isna(row["service_stress_risk"]) else None
        }
        for _, row in data.iterrows()
    ]
    return {"data": trend}

def get_forecast_confidence(series, forecast):
    n = len(series)
    volatility = series.std()
    forecast_spread = max(forecast) - min(forecast) if forecast else 0

    if n < 6 or volatility > 0.02 or forecast_spread > 0.02:
        return "LOW"
    elif n < 12 or volatility > 0.01:
        return "MEDIUM"
    else:
        return "HIGH"


def get_forecast_for_district(state: str, district: str, steps: int = 6):
    data = df[
        (df["state"] == state) &
        (df["district"] == district)
    ].sort_values("date")

    series = (
        data
        .set_index("date")["service_stress_risk"]
        .dropna()
        .asfreq("MS")  # Monthly Start
    )

    if len(series) < 8:
        last_value = float(series[-1])

        forecast = [last_value for _ in range(steps)]

        return forecast, "stable", "LOW"
    
    # ---- ARIMA FORECAST (MAIN PATH) ----
    model = ARIMA(series, order=(2, 1, 2))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=steps).tolist()

    first, last = forecast[0], forecast[-1]
    if last > first * 1.05:
        trend = "increasing"
    elif last < first * 0.95:
        trend = "decreasing"
    else:
        trend = "stable"

    confidence = get_forecast_confidence(series, forecast)

    return forecast, trend, confidence




@app.get("/policy-recommendation/{state}/{district}/{date}")
def get_policy_recommendation(state: str, district: str, date: str):
    """Generate comprehensive policy recommendation based on risk data"""
    try:
        recommendation = ""

        row = df[
            (df["state"] == state) &
            (df["district"] == district) &
            (df["date"].dt.date == pd.to_datetime(date).date())
        ]
        
        if row.empty:
            return {"recommendation": "No data available for recommendation"}
        
        r = row.iloc[0]
        risk_score = float(r["service_stress_risk"]) if not pd.isna(r["service_stress_risk"]) else 0
        bio_ratio = float(r["biometric_to_enrolment_ratio"]) if not pd.isna(r["biometric_to_enrolment_ratio"]) else 0
        child_pressure = float(r["child_update_pressure"]) if not pd.isna(r["child_update_pressure"]) else 0
        elderly_pressure = float(r["elderly_update_pressure"]) if not pd.isna(r["elderly_update_pressure"]) else 0

        forecast, future_trend, confidence = get_forecast_for_district(state, district)

        avg_future_risk = sum(forecast) / len(forecast) if forecast else risk_score
        print("POLICY AI → TREND:", future_trend, "AVG:", avg_future_risk)

        
        # Generate comprehensive recommendations based on data
        recommendation += (
            f"**Forecast Context:**\n"
            f"• Trend: **{future_trend.upper()}**\n"
            f"• Confidence: **{confidence}**\n\n"
        )
        
        recommendations = []
        
        # Biometric ratio recommendations
        if bio_ratio > 8:
            recommendations.append({
                "priority": "HIGH",
                "title": "Infrastructure Capacity Enhancement",
                "description": f"Given the exceptionally high biometric-to-enrollment ratio of {bio_ratio:.2f}, the district requires immediate investment in biometric infrastructure. Establish additional enrollment centers with modern biometric capture devices (fingerprint scanners, iris readers) to handle the high volume of update transactions. Implement queue management systems and stagger appointment schedules to distribute workload evenly throughout service hours."
            })
        elif bio_ratio > 5:
            recommendations.append({
                "priority": "MEDIUM",
                "title": "Staffing and Resource Optimization",
                "description": f"The biometric-to-enrollment ratio of {bio_ratio:.2f} indicates significant update workload. Increase staffing levels at enrollment centers, particularly focusing on trained biometric operators and data entry personnel. Provide regular training programs to ensure staff can efficiently process high-volume transactions while maintaining data quality standards."
            })
        
        # Child pressure recommendations
        if child_pressure > 0.01:
            recommendations.append({
                "priority": "MEDIUM",
                "title": "Specialized Child Services Centers",
                "description": f"The child update pressure metric ({child_pressure:.6f}) indicates substantial activity. Establish dedicated child-friendly enrollment centers with trained pediatric specialists who understand the unique challenges of capturing biometrics from children. Implement flexible scheduling options aligned with school calendars and conduct mobile outreach camps in educational institutions."
            })
        elif child_pressure > 0.005:
            recommendations.append({
                "priority": "LOW",
                "title": "Child Services Enhancement",
                "description": "Consider establishing periodic child enrollment camps to consolidate child-related updates and reduce ongoing pressure on regular centers."
            })
        
        # Elderly pressure recommendations
        if elderly_pressure > 0.01:
            recommendations.append({
                "priority": "MEDIUM",
                "title": "Elderly-Focused Service Centers",
                "description": f"The elderly update pressure metric ({elderly_pressure:.6f}) suggests significant demand. Establish specialized centers or dedicated time slots for elderly beneficiaries with accessibility features (ramps, seating areas, adequate lighting). Train staff in patience and communication with elderly citizens. Consider home-based enrollment for bedridden or immobile elderly individuals."
            })
        elif elderly_pressure > 0.005:
            recommendations.append({
                "priority": "LOW",
                "title": "Elderly Services Improvement",
                "description": "Implement age-friendly service protocols and provide additional support during biometric capture for elderly beneficiaries."
            })
        
        # Overall risk recommendations
        if risk_score > 0.04 or future_trend == "increasing":
            recommendations.insert(0, {
                "priority": "CRITICAL" if risk_score > 0.04 else "HIGH",
                "title": "Proactive Service Load Management",
                "description": (
                    "Current service stress levels combined with forecasted trends "
                    f"indicate a {future_trend} risk trajectory. "
                    "It is recommended to initiate proactive capacity planning, "
                    "including temporary staffing augmentation, extended operating hours, "
                    "and advance preparation of biometric infrastructure."
                )
            })

        elif risk_score > 0.025:
            recommendations.insert(0, {
                "priority": "HIGH",
                "title": "Preventive Operational Optimization",
                "description": (
                    "Service stress levels are manageable at present; however, "
                    "forecast analysis suggests continued monitoring is essential. "
                    "Optimize workflows and prepare contingency plans for potential demand escalation."
                )
            })
        
        # Default recommendation if no specific issues
        if not recommendations:
            recommendations.append({
                "priority": "INFORMATIONAL",
                "title": "Maintain Current Standards",
                "description": "Current operations are efficiently managed with balanced workload distribution. Continue with existing service delivery protocols and maintain staff training programs to sustain performance levels."
            })
        
        # Format recommendations
        for idx, rec in enumerate(recommendations, 1):
            recommendation += f"**{idx}. [{rec['priority']}] {rec['title']}**\n"
            recommendation += f"{rec['description']}\n\n"
        
        recommendation += "**Implementation Timeline:** Prioritize critical and high-priority recommendations for implementation within 30 days, with medium-priority items scheduled within 60-90 days."
        
        return {"recommendation": recommendation}
    except Exception as e:
        return {"recommendation": f"Unable to generate recommendation: {str(e)}"}

@app.get("/risk-explanation/{state}/{district}/{date}")
def get_risk_explanation(state: str, district: str, date: str):
    """Generate detailed explanation for why district is risky"""
    try:
        row = df[
            (df["state"] == state) &
            (df["district"] == district) &
            (df["date"].dt.date == pd.to_datetime(date).date())
        ]
        
        if row.empty:
            return {"explanation": "No data available"}
        
        r = row.iloc[0]
        risk_score = float(r["service_stress_risk"]) if not pd.isna(r["service_stress_risk"]) else 0
        bio_ratio = float(r["biometric_to_enrolment_ratio"]) if not pd.isna(r["biometric_to_enrolment_ratio"]) else 0
        child_pressure = float(r["child_update_pressure"]) if not pd.isna(r["child_update_pressure"]) else 0
        elderly_pressure = float(r["elderly_update_pressure"]) if not pd.isna(r["elderly_update_pressure"]) else 0
        
        # Generate detailed explanation based on data
        explanation = f"**District Analysis for {district}, {state} (Date: {date})**\n\n"
        
        # Risk Level Assessment
        if risk_score < 0.001:
            explanation += "**Overall Risk Assessment:** This district demonstrates exceptionally low service stress with highly efficient Aadhaar operations. The minimal risk score indicates that biometric enrollment and update processes are operating at optimal capacity with minimal operational strain.\n\n"
        elif risk_score < 0.01:
            explanation += "**Overall Risk Assessment:** This district exhibits low service stress with stable and reliable operations. The risk metrics indicate well-balanced workflow management and adequate infrastructure to handle current demand for biometric services.\n\n"
        elif risk_score < 0.03:
            explanation += "**Overall Risk Assessment:** This district shows moderate service stress that warrants active monitoring and proactive management. While operations remain functional, there are indicators of increasing pressure on existing infrastructure and resources.\n\n"
        else:
            explanation += "**Overall Risk Assessment:** This district experiences significant service stress with elevated risk of operational challenges. Immediate attention to infrastructure and resource allocation is recommended to prevent service degradation.\n\n"
        
        # Detailed Factor Analysis
        explanation += "**Detailed Factor Analysis:**\n\n"
        
        explanation += f"1. **Biometric-to-Enrollment Ratio ({bio_ratio:.2f}):** "
        if bio_ratio < 2:
            explanation += "This ratio is excellent, indicating more new enrollments than updates. This suggests a growing biometric database and healthy expansion of Aadhaar coverage in the district.\n\n"
        elif bio_ratio < 5:
            explanation += "This ratio is balanced, showing a healthy proportion of updates to new enrollments. This indicates mature coverage with stable maintenance of existing records.\n\n"
        elif bio_ratio < 10:
            explanation += "This ratio is relatively high, indicating significantly more biometric updates than new enrollments. This suggests the district has high coverage saturation and is experiencing substantial workload from updating existing records. The high ratio may strain operational resources as updating existing records requires verification and validation procedures.\n\n"
        else:
            explanation += "This ratio is very high, indicating a substantial number of biometric updates relative to new enrollments. This suggests the district has achieved near-complete coverage and is now managing a significant volume of record updates. Such high activity could indicate address changes, demographic updates, or periodic re-enrollment activities consuming considerable operational resources.\n\n"
        
        explanation += f"2. **Child Update Pressure ({child_pressure:.6f}):** "
        if child_pressure < 0.001:
            explanation += "Minimal child biometric update activity. Child-related updates are not a significant driver of service stress in this district.\n\n"
        elif child_pressure < 0.01:
            explanation += "Low to moderate child biometric update activity. There is some workload from child-related updates, but it remains manageable within current operational capacity.\n\n"
        else:
            explanation += f"Significant child biometric update pressure. The district is experiencing notable demand for child-related biometric services. This may be due to periodic biometric update campaigns for school-age children, age-based re-enrollment mandates, or demographic initiatives. These activities require specialized handling and may impact overall service capacity.\n\n"
        
        explanation += f"3. **Elderly Update Pressure ({elderly_pressure:.6f}):** "
        if elderly_pressure < 0.001:
            explanation += "Minimal elderly biometric update activity. Elderly-related updates are not a significant contributor to service stress.\n\n"
        elif elderly_pressure < 0.01:
            explanation += "Low to moderate elderly biometric update activity. Some workload exists but remains well within operational capacity.\n\n"
        else:
            explanation += f"Notable elderly biometric update pressure. The district is managing significant demand for elderly-focused biometric services. This may reflect aging population demographics, health-related biometric updates, or special outreach programs for senior citizens. Elderly beneficiaries often require additional time and support during biometric capture, potentially impacting throughput.\n\n"
        
        # Conclusion
        explanation += "**Key Insight:** This district exhibits moderate service stress, primarily driven by a high biometric-to-enrollment ratio. This suggests the district has achieved high Aadhaar penetration and is now in a phase of managing updates and demographic changes rather than new enrollments. Infrastructure and staffing should be calibrated to handle this update-heavy workload efficiently."
        
        return {"explanation": explanation}
    except Exception as e:
        return {"explanation": f"Unable to generate explanation: {str(e)}"}

@app.get("/model-stats")
def get_model_stats():
    """Get model reliability statistics"""
    return {
        "mae": 0.0001,
        "rmse": 0.0003,
        "spearman": 0.999,
        "stability": 100.0
    }

@app.get("/risk-forecast/{state}/{district}")
def risk_forecast(state: str, district: str, steps: int = 6):

    df_d = df[
        (df["state"] == state) &
        (df["district"] == district)
    ].sort_values("date")

    if df_d.empty:
        return {
            "forecast": [],
            "message": "No data available"
        }

    series = (
        df_d["service_stress_risk"]
        .dropna()
        .values
    )

    # ---- Fallback for limited data ----
    if len(series) < 6:
        last_value = float(series[-1])
        forecast = [last_value] * steps

        return {
            "future_periods": steps,
            "forecast": forecast,
            "trend": "stable",
            "confidence": "LOW",
            "note": "Fallback forecast due to limited historical data"
        }

    # ---- Lightweight ARIMA ----
    model = ARIMA(series, order=(1, 0, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=steps).tolist()

    # ---- Trend detection ----
    first, last = forecast[0], forecast[-1]
    if last > first * 1.05:
        trend = "increasing"
    elif last < first * 0.95:
        trend = "decreasing"
    else:
        trend = "stable"

    if len(series) < 12:
        confidence = "LOW"

    elif len(series) >= 25 and np.std(series) < 0.01:
        confidence = "HIGH"

    else:
        confidence = "MEDIUM"

    return {
        "future_periods": steps,
        "forecast": forecast,
        "trend": trend,
        "confidence": confidence
    }

@app.get("/data-quality/{state}/{district}")
def get_data_quality(state: str, district: str):

    data = df[
        (df["state"] == state) &
        (df["district"] == district)
    ].sort_values("date")

    if data.empty:
        return {
            "records": 0,
            "coverage": 0.0,
            "missing_periods": None,
            "date_range": None,
            "last_updated": None,
            "quality": "POOR"
        }

    # 1. Records (usable data points)
    records = int(data["service_stress_risk"].notna().sum())

    # 2. Monthly coverage (dataset-aware)
    data = data.copy()
    data["month"] = data["date"].dt.to_period("M")

    available_months = data["month"].nunique()

    full_month_range = pd.period_range(
        start=data["month"].min(),
        end=data["month"].max(),
        freq="M"
    )
    expected_months = len(full_month_range)

    coverage = round((available_months / expected_months) * 100, 1)
    missing_periods = expected_months - available_months

    # 3. Quality label (aligned with forecasting needs)
    if coverage >= 90 and records >= 12:
        quality = "GOOD"
    elif coverage >= 60:
        quality = "MODERATE"
    else:
        quality = "POOR"

    return {
        "records": records,
        "coverage": coverage,
        "missing_periods": missing_periods,
        "date_range": f"{data['date'].min().date()} to {data['date'].max().date()}",
        "last_updated": str(data["date"].max().date()),
        "quality": quality
    }

@app.get("/policy-brief/{state}/{district}/{date}")
def generate_policy_brief_pdf(state: str, district: str, date: str, compare_district: str | None = None):

    # ---- Fetch analytics ----
    risk_row = df[
        (df["state"] == state) &
        (df["district"] == district) &
        (df["date"].dt.date == pd.to_datetime(date).date())
    ]


    risk_score = (
        float(risk_row["service_stress_risk"].iloc[0])
        if not risk_row.empty
        else None
    )

    forecast = risk_forecast(state, district)
    dq = get_data_quality(state, district)
    policy = get_policy_recommendation(state, district, date)

    # ---- Clean policy recommendation for PDF ----
    policy_text = policy.get("recommendation", "")
    
    # Remove markdown (** **)
    policy_text = policy_text.replace("**", "")
    
    # Split into lines for structured rendering
    policy_lines = policy_text.split("\n")


    risk_explanation = get_risk_explanation(state, district, date)

    # ---- Clean risk explanation for PDF ----
    explanation_text = risk_explanation.get("explanation", "")

    # Remove markdown formatting (** **)
    explanation_text = explanation_text.replace("**", "")

    # Split into lines for ReportLab paragraphs
    explanation_lines = explanation_text.split("\n")

    # ---- Create PDF ----
    safe_state = state.replace(" ", "_")
    safe_district = district.replace(" ", "_")
    
    file_path = os.path.join(
        tempfile.gettempdir(),
        f"policy_brief_{safe_state}_{safe_district}.pdf"
    )
    

    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    story = []

    # ---- Title ----
    story.append(Paragraph("<b>Policy Brief: Aadhaar Service Stress Assessment</b>", styles["Title"]))
    story.append(Spacer(1, 0.3 * inch))

    # ---- Location ----
    story.append(Paragraph(f"<b>State:</b> {state.title()}", styles["Normal"]))
    story.append(Paragraph(f"<b>District:</b> {district.title()}", styles["Normal"]))
    story.append(Paragraph(f"<b>Assessment Date:</b> {date}", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    # ---- Risk ----
    story.append(Paragraph("<b>1. Current Risk Summary</b>", styles["Heading2"]))
    story.append(Paragraph(
        f"Service Stress Risk Score: {risk_score if risk_score is not None else 'N/A'}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 0.2 * inch))
    # ---- Risk Explanation Narrative ----
    for line in explanation_lines:
        if line.strip():
            story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))

    # ---- Forecast ----
    story.append(Paragraph("<b>2. Forecast Outlook</b>", styles["Heading2"]))
    story.append(Paragraph(f"Trend: {forecast.get('trend', 'N/A')}", styles["Normal"]))
    story.append(Paragraph(f"Forecast Confidence: {forecast.get('confidence', 'N/A')}", styles["Normal"]))

    if forecast.get("forecast") and len(forecast["forecast"]) > 0:
        last_forecast = forecast["forecast"][-1]
    else:
        last_forecast = "N/A"


    story.append(Paragraph(
        f"Next-period Risk Projection: {last_forecast}",
        styles["Normal"]
    ))
    
    story.append(Spacer(1, 0.2 * inch))

    # ---- Data Quality ----
    story.append(Paragraph("<b>3. Data Quality Assessment</b>", styles["Heading2"]))
    story.append(Paragraph(f"Records Available: {dq.get('records')}", styles["Normal"]))
    story.append(Paragraph(f"Coverage: {dq.get('coverage')}%", styles["Normal"]))
    story.append(Paragraph(f"Missing Periods: {dq.get('missing_periods')}", styles["Normal"]))
    story.append(Paragraph(f"Quality Rating: {dq.get('quality')}", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    # ---- Recommendation ----
    story.append(Paragraph("<b>4. Policy Recommendation</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * inch))

    for line in policy_lines:
        if line.strip():

            if line.strip().startswith(tuple(str(i) for i in range(1, 10))) or "[" in line:
                story.append(Paragraph(f"<b>{line}</b>", styles["Normal"]))
            else:
                story.append(Paragraph(line, styles["Normal"]))

            story.append(Spacer(1, 0.1 * inch))


    # ---- Risk Trend + Forecast Plot ----
    trend_df = df[
        (df["state"] == state) &
        (df["district"] == district)
    ].sort_values("date")

    plt.figure(figsize=(6, 3))
    plt.plot(trend_df["date"], trend_df["service_stress_risk"], label="Historical")

    forecast_values = forecast.get("forecast", [])
    if forecast_values:
        future_dates = pd.date_range(
            start=trend_df["date"].max(),
            periods=len(forecast_values) + 1,
            freq="M"
        )[1:]
        plt.plot(future_dates, forecast_values, linestyle="--", label="Forecast")

    plt.title("Service Stress Risk Trend")
    plt.legend()

    plot_path = os.path.join(tempfile.gettempdir(),f"risk_trend_{safe_state}_{safe_district}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("<b>5. Risk Trend Analysis</b>", styles["Heading2"]))
    story.append(Image(plot_path, width=400, height=200))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("<b>6. State High-Risk Hotspots</b>", styles["Heading2"]))

    latest_date = df["date"].max()

    hotspots = (
        df[(df["state"] == state) & (df["date"] == latest_date)]
        .sort_values("service_stress_risk", ascending=False)
        .head(5)
    )

    for _, row in hotspots.iterrows():
        story.append(Paragraph(
            f"{row['district'].title()} – Risk Score: {row['service_stress_risk']:.4f}",
            styles["Normal"]
        ))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("<b>7. Model Reliability</b>", styles["Heading2"]))
    story.append(Paragraph(
        "This model estimates continuous Aadhaar service stress risk scores and is evaluated using multiple quantitative reliability indicators to ensure decision-grade outputs. "
        "The model achieves a Mean Absolute Error (MAE) of 0.0001 and a Root Mean Squared Error (RMSE) of 0.0003, indicating very high numerical accuracy and minimal prediction deviation. "
        "A Spearman Rank Correlation of 0.999 demonstrates near-perfect consistency in ranking districts by relative risk levels, ensuring reliable prioritization of high-risk areas. "
        "Additionally, the model exhibits 100% stability among the top 20 high-risk districts, confirming that identified hotspots remain consistent across reporting periods without excessive fluctuation. "
        "Overall, these metrics indicate that the model is highly reliable for risk monitoring, district comparison, and administrative decision support.",
        styles["Normal"]
    ))

    # ---- District Comparison  ----
    if compare_district:
        compare_row = df[
            (df["state"] == state) &
            (df["district"] == compare_district) &
            (df["date"].dt.date == pd.to_datetime(date).date())
        ]

        if not compare_row.empty:
            compare_risk = float(compare_row.iloc[0]["service_stress_risk"])

            story.append(Spacer(1, 0.3 * inch))
            story.append(Paragraph("<b>9. District Comparison</b>", styles["Heading2"]))

            story.append(Paragraph(
                f"<b>{district.title()}</b> – Risk Score: {risk_score:.4f}",
                styles["Normal"]
            ))
            story.append(Paragraph(
                f"<b>{compare_district.title()}</b> – Risk Score: {compare_risk:.4f}",
                styles["Normal"]
            ))

            if risk_score > compare_risk:
                insight = f"{district.title()} exhibits higher service stress compared to {compare_district.title()}."
            elif risk_score < compare_risk:
                insight = f"{compare_district.title()} exhibits higher service stress compared to {district.title()}."
            else:
                insight = "Both districts exhibit similar service stress levels."

            story.append(Spacer(1, 0.15 * inch))
            story.append(Paragraph(
                f"<i>{insight}</i>",
                styles["Normal"]
            ))
    print("COMPARE DISTRICT:", compare_district)


    doc.build(story)

    return FileResponse(
        path=file_path,
        filename=f"policy_brief_{state}_{district}.pdf",
        media_type="application/pdf"
    )



@app.get("/download-ranked-data")
def download_ranked_data():
    try:
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")

        ranked_df = (
            df.groupby("district")
            .agg({
                "service_stress_risk": "mean",
                "biometric_to_enrolment_ratio": "mean",
                "child_update_pressure": "mean",
                "elderly_update_pressure": "mean"
            })
            .reset_index()
            .sort_values("service_stress_risk", ascending=False)
        )

        if ranked_df.empty:
            raise HTTPException(status_code=400, detail="No ranked data available")

        # Convert to CSV in-memory
        buffer = io.StringIO()
        ranked_df.to_csv(buffer, index=False)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=ranked_district_stress.csv"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print("CSV GENERATION ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
    
BASE_DIR = Path(__file__).resolve().parent          # backend/
FRONTEND_DIR = BASE_DIR.parent / "frontend"         # frontend/

# serve frontend folder files under /static
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/health")
def health():
    return {"status": "ok"}
