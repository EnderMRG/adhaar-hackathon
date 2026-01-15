Perfect. Based on **your updated frontend, layout fixes, mobile responsiveness, and UI restructuring**, here is a **clean, updated README** that accurately reflects the **current state of the project**.

You can **replace your existing README.md entirely** with the following.

---

# Aadhaar Service Stress Dashboard

**Responsive Risk Analysis & Forecasting Platform**

## Overview

The **Aadhaar Service Stress Dashboard** is a full-stack data analytics and visualization platform designed to **monitor, forecast, and explain Aadhaar service stress across districts in India**.

It combines:

* A **machine learning–based risk model**
* A **FastAPI backend**
* A **modern, responsive HTML + Tailwind CSS dashboard**
* **AI-generated explanations and policy recommendations**

The dashboard is optimized for **desktop and mobile screens**, following a **government-grade, decision-support design**.

---

## Key Capabilities

### 🔍 Risk Assessment

* Computes a **continuous service stress risk score**
* Converts scores into **qualitative risk levels** (Low / Medium / High)
* Displays a clear **Risk Assessment card** for quick interpretation

### 📊 Comparative Analysis

* Shows **risk percentile ranking** among all districts nationwide
* Enables contextual comparison for policymakers

### 📈 Trend & Forecasting

* Visualizes **historical risk trends**
* Generates **future risk forecasts** for upcoming reporting periods
* Adds forecast overlays directly onto trend charts

### 📋 Data Quality Monitoring

* Evaluates data reliability using:

  * Records available
  * Coverage %
  * Missing periods
  * Date range
* Displays a **clear quality status (Good / Moderate / Poor)** using a visual indicator

### 🧠 AI-Powered Insights

* Natural-language **risk explanations**
* **Policy recommendations** generated using Google Gemini API
* Human-readable summaries for decision-makers

### 📱 Mobile-Responsive Design

* Fully responsive layout using **Tailwind CSS**
* Optimized for:

  * Phones
  * Tablets
  * Desktops
* No horizontal scrolling or layout breaks on small screens

---

## Technology Stack

### Frontend

* HTML5
* Tailwind CSS (CDN)
* Vanilla JavaScript
* Chart.js (data visualization)
* Marked.js (markdown rendering)

### Backend

* FastAPI
* Python
* Scikit-learn (ML model)
* Pandas / NumPy

### AI

* Google Gemini API (policy recommendations & explanations)

---

## Project Structure

```
.
├── app.py                      # Streamlit-based analytical dashboard (optional)
├── backend
│   ├── main.py                 # FastAPI backend
│   ├── requirements.txt        # Backend dependencies
│   ├── aadhaar_merged_dataset.csv
│   └── aadhaar_service_stress_model.pkl
├── frontend
│   └── index.html              # Responsive Tailwind-based dashboard
└── README.md
```

---

## Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone <repository-url>
cd aadhaar-service-dashboard
```

---

### 2️⃣ Backend Setup

Create and activate a virtual environment:

```bash
python -m venv venv
```

**Activate:**

* Windows:

  ```bash
  venv\Scripts\activate
  ```
* macOS / Linux:

  ```bash
  source venv/bin/activate
  ```

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

---

### 3️⃣ Environment Variables

Create a `.env` file inside the `backend` directory:

```
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

---

## Running the Application

### Option 1: FastAPI Backend + Responsive Frontend (Recommended)

#### Start Backend

```bash
cd backend
uvicorn main:app --reload
```

Backend runs at:

```
http://localhost:8000
```

#### Launch Frontend

Open:

```
frontend/index.html
```

in any modern browser.

✔ Works on desktop and mobile
✔ No build step required

---

## API Endpoints

| Endpoint                                               | Description                |
| ------------------------------------------------------ | -------------------------- |
| `GET /states`                                          | List all states            |
| `GET /districts/{state}`                               | Districts in a state       |
| `GET /dates/{state}/{district}`                        | Available reporting dates  |
| `GET /risk`                                            | Risk score & metrics       |
| `GET /risk-verdict/{risk_score}`                       | LOW / MEDIUM / HIGH        |
| `GET /risk-percentile/{state}/{district}/{date}`       | National percentile        |
| `GET /risk-trend/{state}/{district}`                   | Historical trend           |
| `GET /risk-forecast/{state}/{district}`                | Future forecast            |
| `GET /data-quality/{state}/{district}`                 | Data quality metrics       |
| `GET /top-districts`                                   | Top 10 high-risk districts |
| `GET /district-hotspots/{state}`                       | State-level hotspots       |
| `GET /risk-explanation/{state}/{district}/{date}`      | AI explanation             |
| `GET /policy-recommendation/{state}/{district}/{date}` | AI recommendation          |
| `GET /download-ranked-data`                            | CSV export                 |

---

## Design Principles

* **Decision-first UI** (not analytics clutter)
* **Clear visual hierarchy**
* **Explainable AI outputs**
* **Mobile-first responsiveness**
* **Government dashboard aesthetics**

---

## Intended Use

* Policy planning & intervention
* District-level operational monitoring
* Risk forecasting & preparedness
* Data quality auditing
* Evidence-based governance


