# 🏛️ Westminster City Council — Analytics Demo

## Strategy & Performance Team: Why Python, Streamlit & Regression Matter

This Streamlit app demonstrates how Python-based analytics can transform the
Risk & Performance Board reporting process — using real KPIs from the Q2 25/26 report.

---

## Quick Start

```bash
# 1. Install dependencies
pip install streamlit pandas numpy scikit-learn plotly openpyxl

# 2. Run the app
streamlit run app.py

# 3. Open http://localhost:8501 in your browser
```

---

## What's Inside

| Page | What It Shows |
|------|---------------|
| **Why Streamlit & Python?** | Side-by-side comparison with Power BI, pain-vs-gain analysis |
| **Live KPI Dashboard** | Interactive version of slides 3–21 of the board report |
| **Regression & Forecasting** | Linear & polynomial regression on any KPI, with projections |
| **Benchmarking Explorer** | London borough comparisons — replaces manual LG Inform lookups |
| **Early Warning System** | Automated alerts for KPIs trending towards failure |
| **Techniques Showcase** | Correlation, anomaly detection, what-if scenarios, R integration |

---

## Techniques Demonstrated

- Linear Regression
- Polynomial Regression
- Correlation Analysis
- Z-Score Anomaly Detection
- What-If Scenario Modelling
- R Integration via subprocess
- Automated RAG Rating
- Interactive Benchmarking

---

## Extending This App

To connect to live data sources:

```python
# Example: Read from a SharePoint/Excel file
df = pd.read_excel("//sharepoint/path/to/kpi_data.xlsx")

# Example: Read from a database
import sqlalchemy
engine = sqlalchemy.create_engine("postgresql://user:pass@host/db")
df = pd.read_sql("SELECT * FROM kpi_quarterly", engine)
```

---

## Requirements

- Python 3.9+
- See `requirements.txt` for packages

Built for the WCC Strategy & Performance Team.
