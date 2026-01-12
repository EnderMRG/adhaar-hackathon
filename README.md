# Aadhaar Service Stress Risk Analysis

This project provides a comprehensive analysis of Aadhaar service stress risk across different districts. It uses a machine learning model to predict risk scores based on various operational data points. The project includes a detailed Streamlit dashboard for in-depth analysis and a lightweight FastAPI backend with a simple frontend for quick risk checks.

## Features

- **Risk Prediction**: A machine learning model predicts the service stress risk score for a given district and date.
- **Interactive Dashboard**: A Streamlit application provides a rich user interface to visualize risk trends, compare districts, and get detailed explanations.
- **AI-Powered Recommendations**: The application uses Google's Gemini API to generate policy recommendations for mitigating service stress.
- **REST API**: A FastAPI backend exposes several endpoints to access the risk data and model predictions.
- **Minimalist Frontend**: A simple HTML/JavaScript frontend to interact with the FastAPI backend.

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── backend
│   ├── main.py             # FastAPI backend
│   ├── requirements.txt    # Python dependencies
│   ├── aadhaar_merged_dataset.csv  # Dataset
│   └── aadhaar_service_stress_model.pkl # Trained ML model
├── frontend
│   └── index.html          # Minimalist HTML frontend
└── README.md
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd adhaar-hackathon
    ```

2.  **Create a virtual environment and install dependencies:**
    - Create a virtual environment:
        ```bash
        python -m venv venv
        ```
    - Activate the virtual environment:
        - On Windows:
            ```bash
            venv\Scripts\activate
            ```
        - On macOS/Linux:
            ```bash
            source venv/bin/activate
            ```
    - Install the required packages:
        ```bash
        pip install -r backend/requirements.txt
        ```

3.  **Set up environment variables:**
    Create a `.env` file in the `backend` directory and add your Google API key:
    ```
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
    ```

## How to Run

You can run this project in two ways: as a Streamlit application or as a FastAPI backend with a separate frontend.

### 1. Run as a Streamlit Application

The Streamlit app provides a full-featured dashboard.

- **Navigate to the root directory and run the following command:**
    ```bash
    streamlit run app.py
    ```
- Open your browser and go to `http://localhost:8501`.

### 2. Run as a FastAPI Backend and Frontend

This option provides a lightweight interface for checking risk scores.

- **Start the FastAPI server:**
    - Navigate to the `backend` directory:
        ```bash
        cd backend
        ```
    - Run the following command:
        ```bash
        uvicorn main:app --reload
        ```
    - The API will be available at `http://localhost:8000`.

- **Open the frontend:**
    - In your file explorer, navigate to the `frontend` directory and open the `index.html` file in your web browser.

## API Endpoints (FastAPI)

The FastAPI server provides the following endpoints:

- `GET /states`: Get a list of all states.
- `GET /districts/{state}`: Get a list of districts for a given state.
- `GET /dates/{state}/{district}`: Get a list of available dates for a given state and district.
- `GET /risk?state=<state>&district=<district>&date=<date>`: Get the risk score and other metrics for a given selection.
- `GET /risk-verdict/{risk_score}`: Get a qualitative risk verdict (LOW, MEDIUM, HIGH).
- `GET /risk-percentile/{state}/{district}/{date}`: Get the risk percentile for a district.
- `GET /top-districts`: Get the top 10 high-risk districts.
- `GET /district-hotspots/{state}`: Get the top 5 high-risk districts in a state.
- `GET /risk-trend/{state}/{district}`: Get the risk trend over time for a district.
- `GET /policy-recommendation/{state}/{district}/{date}`: Generate a policy recommendation.
- `GET /risk-explanation/{state}/{district}/{date}`: Generate a detailed risk explanation.
- `GET /model-stats`: Get model reliability statistics.
- `GET /download-ranked-data`: Download a CSV of ranked district stress data.
