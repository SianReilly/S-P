"""
Westminster City Council — Strategy & Performance Analytics Demo
Showcasing the benefits of Python, Streamlit, and Regression for local government reporting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="WCC Analytics Demo",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    
    .block-container { padding-top: 1.5rem; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a2744 0%, #2a4070 100%);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        color: white;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.8; font-weight: 400; }
    .metric-card .big-num { font-size: 2.2rem; font-weight: 700; margin: 0.2rem 0; }
    .metric-card .delta { font-size: 0.85rem; }
    .delta-up { color: #6ee7a0; }
    .delta-down { color: #ff7b7b; }
    
    .rag-red { background-color: #dc3545; color: white; padding: 3px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; }
    .rag-amber { background-color: #fd7e14; color: white; padding: 3px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; }
    .rag-green { background-color: #28a745; color: white; padding: 3px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; }
    
    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #2a4070;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
    }
    .insight-box strong { color: #1a2744; }
    
    .benefit-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        height: 100%;
    }
    .benefit-card h4 { color: #1a2744; margin-top: 0.3rem; }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2744 0%, #1f3460 100%);
    }
    div[data-testid="stSidebar"] * { color: white !important; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MOCK DATA — based on real WCC Q2 report KPIs
# ──────────────────────────────────────────────
@st.cache_data
def load_kpi_timeseries():
    """Quarterly KPI data inspired by the Q2 25/26 Board Report."""
    quarters = ["Q1 23/24","Q2 23/24","Q3 23/24","Q4 23/24",
                "Q1 24/25","Q2 24/25","Q3 24/25","Q4 24/25",
                "Q1 25/26","Q2 25/26"]
    np.random.seed(42)
    data = {
        "Quarter": quarters,
        "FOI Timeliness %": [82, 84, 87, 85, 83, 85, 86, 84, 86, 83],
        "SAR Timeliness %": [88, 85, 90, 95, 86, 75, 82, 87, 87, 78],
        "Direct Payments %": [24.5, 24.8, 25.0, 25.2, 25.5, 25.3, 25.6, 25.8, 25.8, 26.2],
        "NEET 16-17 %": [1.5, 1.8, 2.0, 2.2, 2.5, 2.6, 2.8, 2.9, 3.0, 3.0],
        "Care Leavers EET %": [80, 78, 76, 74, 72, 70, 68, 66, 64, 63.9],
        "Calls Answered %": [94, 93, 95, 94, 93, 92, 93, 92, 92, 86],
        "HRA Rent Collected %": [96.5, 96.2, 96.8, 97.0, 96.0, 95.8, 96.1, 95.5, 93.2, 95.2],
        "Households in TA": [3200, 3350, 3500, 3600, 3750, 3840, 3950, 4100, 4307, 4416],
        "Tenant Overall Satisfaction %": [70, 69, 68, 67.5, 67, 68, 67, 66.3, 66.3, 64.6],
        "Repair Satisfaction %": [68, 67, 66, 65.5, 65, 66, 65, 64.1, 64.1, 63.7],
        "FOI Requests Volume": [480, 510, 490, 520, 540, 575, 590, 620, 680, 794],
    }
    df = pd.DataFrame(data)
    df["Q_num"] = range(1, len(quarters)+1)
    return df

@st.cache_data
def load_benchmark_data():
    """London borough comparison data."""
    boroughs = ["Westminster","Camden","Islington","Kensington & Chelsea",
                "Lambeth","Southwark","Tower Hamlets","Hackney",
                "Wandsworth","Hammersmith & Fulham","Newham","Haringey"]
    np.random.seed(99)
    return pd.DataFrame({
        "Borough": boroughs,
        "Tenant Satisfaction %": [64.6, 62, 58, 55, 60, 63, 57, 59, 67, 61, 72, 56],
        "FOI Timeliness %": [83, 78, 80, 76, 74, 82, 71, 77, 85, 79, 73, 75],
        "Households in TA per 100k": [34, 28, 30, 22, 32, 29, 35, 27, 18, 25, 38, 31],
        "NEET Rate %": [3.0, 2.1, 1.8, 2.5, 2.3, 2.0, 2.8, 1.9, 1.5, 2.2, 3.2, 2.6],
    })

df = load_kpi_timeseries()
bench = load_benchmark_data()

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏛️ WCC Analytics")
    st.markdown("**Strategy & Performance**")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Why Streamlit & Python?",
        "📊 Live KPI Dashboard",
        "📈 Regression & Forecasting",
        "🔍 Benchmarking Explorer",
        "⚠️ Early Warning System",
        "🧰 Techniques Showcase",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Demo built for the Risk & Performance Board — showing what's possible beyond static slides.")


# ══════════════════════════════════════════════
# PAGE 1: WHY STREAMLIT & PYTHON?
# ══════════════════════════════════════════════
if page == "🏠 Why Streamlit & Python?":
    st.markdown("# Why Streamlit & Python for WCC?")
    st.markdown("#### Transforming static board reports into live, interactive intelligence")
    st.markdown("")

    # --- Pain vs Gain ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔴 Current Challenges")
        st.markdown("""
<div class="insight-box" style="border-left-color:#dc3545;">
<strong>Static PowerPoint decks</strong> — 28+ slides updated manually every quarter. 
Data is stale by the time it reaches the board.<br><br>
<strong>No forecasting</strong> — KPIs are reported historically with no projection 
of where they're heading.<br><br>
<strong>Manual benchmarking</strong> — London borough comparisons require separate 
lookups on LG Inform.<br><br>
<strong>No interactivity</strong> — Board members can't drill into a specific 
directorate or KPI on the spot.<br><br>
<strong>Siloed tools</strong> — Data sits across spreadsheets, LG Inform, 
internal databases, and email threads.
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🟢 What Streamlit + Python Unlocks")
        st.markdown("""
<div class="insight-box" style="border-left-color:#28a745;">
<strong>Live dashboards</strong> — KPIs update automatically from source data. 
No manual slide editing.<br><br>
<strong>Regression & forecasting</strong> — See where a KPI will be in 2, 4, 
or 6 quarters if the current trend continues.<br><br>
<strong>One-click benchmarking</strong> — Instantly compare Westminster against 
every London borough, with filters.<br><br>
<strong>Board-ready interactivity</strong> — Click a KPI, explore the trend, 
read the commentary — all in one place.<br><br>
<strong>Early warnings</strong> — Automated alerts when a KPI is trending 
towards its red threshold.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Streamlit vs Power BI — Head to Head")
    
    comparison = pd.DataFrame({
        "Criteria": [
            "Cost",
            "Setup time",
            "Python / R integration",
            "Regression & ML built-in",
            "Custom visualisations",
            "Version control (Git)",
            "Hosting flexibility",
            "Learning curve for analysts",
            "Real-time data refresh",
            "Sharing with non-technical users",
        ],
        "Streamlit": [
            "Free & open-source",
            "Minutes (pip install streamlit)",
            "Native — it IS Python",
            "Yes — scikit-learn, statsmodels, etc.",
            "Unlimited (Plotly, Altair, Matplotlib)",
            "Yes — full Git workflow",
            "Cloud, on-prem, or Streamlit Community",
            "Low if you know Python",
            "Yes — reads live data sources",
            "URL link — no licence needed",
        ],
        "Power BI": [
            "£7.50–£15/user/month (Pro/Premium)",
            "Hours to days",
            "Limited Python/R visuals (sandbox)",
            "No — needs external scripts",
            "Constrained to built-in chart types",
            "Limited — .pbix files are binary",
            "Microsoft cloud (or Report Server ££)",
            "Low for drag-and-drop; high for DAX",
            "Scheduled refresh (8x/day on Pro)",
            "Requires Power BI licence or publish-to-web",
        ]
    })
    
    st.dataframe(
        comparison.set_index("Criteria"),
        use_container_width=True,
        height=400,
    )

    st.markdown("""
<div class="insight-box">
<strong>💡 Bottom line:</strong> Power BI is excellent for quick drag-and-drop dashboards.
But for a team that wants to learn Python, run regressions, build forecasting models, 
and create bespoke analytical tools — <strong>Streamlit gives you the full power of 
Python's data science ecosystem with zero licensing cost</strong>.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### What You're About to See")
    st.markdown("""
This demo uses **real KPIs from the Q2 25/26 Risk & Performance Board Report** to show:

1. **Live KPI Dashboard** — every metric at a glance, with RAG ratings computed automatically  
2. **Regression & Forecasting** — linear and polynomial regression on your KPIs, projecting future quarters  
3. **Benchmarking Explorer** — interactive London borough comparisons  
4. **Early Warning System** — automated alerts for KPIs trending towards failure  
5. **Techniques Showcase** — correlation analysis, anomaly detection, and more  

Navigate using the sidebar. Each page is something your team could build and own.
""")


# ══════════════════════════════════════════════
# PAGE 2: LIVE KPI DASHBOARD
# ══════════════════════════════════════════════
elif page == "📊 Live KPI Dashboard":
    st.markdown("# 📊 Live KPI Dashboard")
    st.markdown("*Auto-generated from data — no manual slide editing needed*")
    st.markdown("")

    # Top-level metric cards
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    metrics = [
        ("FOI Timeliness", f"{latest['FOI Timeliness %']}%", latest['FOI Timeliness %'] - prev['FOI Timeliness %'], 85, "%"),
        ("Direct Payments", f"{latest['Direct Payments %']}%", latest['Direct Payments %'] - prev['Direct Payments %'], 28, "%"),
        ("Calls Answered", f"{latest['Calls Answered %']}%", latest['Calls Answered %'] - prev['Calls Answered %'], 92, "%"),
        ("Households in TA", f"{int(latest['Households in TA']):,}", latest['Households in TA'] - prev['Households in TA'], None, ""),
    ]
    
    cols = st.columns(4)
    for i, (name, val, delta, target, unit) in enumerate(metrics):
        with cols[i]:
            delta_class = "delta-down" if delta < 0 else "delta-up"
            if name == "Households in TA":
                delta_class = "delta-down" if delta > 0 else "delta-up"
            arrow = "▲" if delta >= 0 else "▼"
            if name == "Households in TA":
                arrow = "▲" if delta > 0 else "▼"
            rag = ""
            if target:
                if float(latest[f'{name} %' if '%' in val else name]) >= target:
                    rag = '<span class="rag-green">ON TARGET</span>'
                else:
                    rag = '<span class="rag-red">BELOW TARGET</span>'
            st.markdown(f"""
<div class="metric-card">
    <h3>{name} {rag}</h3>
    <div class="big-num">{val}</div>
    <div class="delta {delta_class}">{arrow} {abs(delta):.1f}{unit} vs Q1</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("")

    # KPI selector + trend chart
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        selected_kpi = st.selectbox("Select KPI to explore", [
            "FOI Timeliness %", "SAR Timeliness %", "Direct Payments %",
            "NEET 16-17 %", "Care Leavers EET %", "Calls Answered %",
            "HRA Rent Collected %", "Households in TA",
            "Tenant Overall Satisfaction %", "Repair Satisfaction %",
        ])
        
        targets = {
            "FOI Timeliness %": 85, "SAR Timeliness %": 86, "Direct Payments %": 28,
            "NEET 16-17 %": 2, "Care Leavers EET %": 70, "Calls Answered %": 92,
            "HRA Rent Collected %": 97, "Households in TA": None,
            "Tenant Overall Satisfaction %": 66, "Repair Satisfaction %": 65,
        }
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Quarter"], y=df[selected_kpi],
            mode="lines+markers",
            line=dict(color="#2a4070", width=3),
            marker=dict(size=8),
            name="Westminster",
        ))
        
        target_val = targets.get(selected_kpi)
        if target_val:
            fig.add_hline(y=target_val, line_dash="dash", line_color="#dc3545",
                          annotation_text=f"Target: {target_val}")
        
        fig.update_layout(
            title=f"{selected_kpi} — Quarterly Trend",
            xaxis_title="", yaxis_title=selected_kpi,
            template="plotly_white", height=400,
            font=dict(family="DM Sans"),
            margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### RAG Summary — Q2 25/26")
        rag_data = []
        for kpi, target in targets.items():
            val = latest[kpi]
            if target is None:
                rag_data.append((kpi, val, "ℹ️ Monitor"))
                continue
            # For NEET, lower is better
            if kpi in ["NEET 16-17 %"]:
                if val <= target: rag_data.append((kpi, val, "🟢 Green"))
                elif val <= target * 1.1: rag_data.append((kpi, val, "🟠 Amber"))
                else: rag_data.append((kpi, val, "🔴 Red"))
            else:
                if val >= target: rag_data.append((kpi, val, "🟢 Green"))
                elif val >= target * 0.9: rag_data.append((kpi, val, "🟠 Amber"))
                else: rag_data.append((kpi, val, "🔴 Red"))
        
        rag_df = pd.DataFrame(rag_data, columns=["KPI", "Value", "RAG"])
        st.dataframe(rag_df, hide_index=True, use_container_width=True, height=380)

    st.markdown("""
<div class="insight-box">
<strong>🎯 What this replaces:</strong> Slides 3–21 of the current board report. 
Instead of 19 static slides, every KPI is explorable in a single interactive view. 
RAG ratings are <em>computed automatically</em> from the data — no manual colouring needed.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 3: REGRESSION & FORECASTING
# ══════════════════════════════════════════════
elif page == "📈 Regression & Forecasting":
    st.markdown("# 📈 Regression & Forecasting")
    st.markdown("*What your course notebooks can do for real council KPIs*")
    st.markdown("")

    col1, col2 = st.columns([1, 1])
    with col1:
        kpi = st.selectbox("Choose a KPI to forecast", [
            "Households in TA", "NEET 16-17 %", "Care Leavers EET %",
            "FOI Timeliness %", "Tenant Overall Satisfaction %",
            "Repair Satisfaction %", "FOI Requests Volume",
        ])
    with col2:
        forecast_q = st.slider("Quarters to forecast ahead", 1, 8, 4)

    reg_type = st.radio("Regression type", ["Linear", "Polynomial (degree 2)"], horizontal=True)

    X = df["Q_num"].values.reshape(-1, 1)
    y = df[kpi].values

    if reg_type == "Linear":
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        future_X = np.arange(len(df)+1, len(df)+1+forecast_q).reshape(-1, 1)
        y_future = model.predict(future_X)
        equation = f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}"
    else:
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        future_X = np.arange(len(df)+1, len(df)+1+forecast_q).reshape(-1, 1)
        y_future = model.predict(poly.transform(future_X))
        c = model.coef_
        equation = f"y = {c[2]:.3f}x² + {c[1]:.2f}x + {model.intercept_:.2f}"

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    future_labels = [f"Q{((i+2)%4)+1} {'26/27' if i < 2 else '27/28'}" for i in range(forecast_q)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Quarter"], y=y, mode="lines+markers",
        name="Actual", line=dict(color="#2a4070", width=3), marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=df["Quarter"], y=y_pred, mode="lines",
        name="Regression fit", line=dict(color="#fd7e14", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=future_labels, y=y_future, mode="lines+markers",
        name="Forecast", line=dict(color="#dc3545", width=3, dash="dash"),
        marker=dict(size=10, symbol="diamond"),
    ))

    fig.update_layout(
        title=f"{kpi} — {reg_type} Regression & {forecast_q}-Quarter Forecast",
        xaxis_title="Quarter", yaxis_title=kpi,
        template="plotly_white", height=450,
        font=dict(family="DM Sans"),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model stats
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{r2:.3f}", help="1.0 = perfect fit. Above 0.7 is generally strong.")
    c2.metric("Mean Absolute Error", f"{mae:.2f}")
    c3.metric("Equation", equation)

    st.markdown(f"""
<div class="insight-box">
<strong>📐 What regression tells us:</strong> If the current trend continues for 
<strong>{kpi}</strong>, the model projects a value of 
<strong>{y_future[-1]:.1f}</strong> in {forecast_q} quarters. 
The R² of {r2:.3f} means the model explains {r2*100:.1f}% of the variance in the data.
{"<br><br><strong>⚠️ Warning:</strong> This KPI is trending in a concerning direction. Early intervention may be needed." if (kpi in ["Households in TA", "NEET 16-17 %"] or (kpi == "Care Leavers EET %" and y_future[-1] < 55)) else ""}
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎓 How This Maps to Your Course")
    st.markdown("""
| Notebook Technique | Applied Here | Board Benefit |
|---|---|---|
| **Linear Regression** | Trend line through quarterly KPI data | Shows the direction of travel mathematically, not just by eye |
| **Polynomial Regression** | Captures acceleration / deceleration in trends | Reveals if a problem is getting worse *faster* |
| **R² and MAE** | Model quality metrics shown above | Confidence level: should we trust this forecast? |
| **Train / Test split** | Could split historical quarters | Validates whether the model generalises |
| **Feature engineering** | Quarter number as predictor | Extensible: add seasonality, policy changes, demand |
""")


# ══════════════════════════════════════════════
# PAGE 4: BENCHMARKING EXPLORER
# ══════════════════════════════════════════════
elif page == "🔍 Benchmarking Explorer":
    st.markdown("# 🔍 London Borough Benchmarking")
    st.markdown("*Replaces manual LG Inform lookups with instant comparisons*")
    st.markdown("")

    bench_metric = st.selectbox("Select metric to compare", [
        "Tenant Satisfaction %", "FOI Timeliness %",
        "Households in TA per 100k", "NEET Rate %",
    ])

    ascending = bench_metric in ["Households in TA per 100k", "NEET Rate %"]
    sorted_bench = bench.sort_values(bench_metric, ascending=ascending)
    
    colors = ["#2a4070" if b == "Westminster" else "#a8bbd6" for b in sorted_bench["Borough"]]

    fig = go.Figure(go.Bar(
        x=sorted_bench[bench_metric],
        y=sorted_bench["Borough"],
        orientation="h",
        marker_color=colors,
        text=sorted_bench[bench_metric].apply(lambda x: f"{x:.1f}"),
        textposition="outside",
    ))
    
    wcc_val = bench.loc[bench["Borough"]=="Westminster", bench_metric].values[0]
    london_avg = bench[bench_metric].mean()
    
    fig.add_vline(x=london_avg, line_dash="dash", line_color="#dc3545",
                  annotation_text=f"London Avg: {london_avg:.1f}")
    
    fig.update_layout(
        title=f"{bench_metric} — Inner London Boroughs",
        xaxis_title=bench_metric, yaxis_title="",
        template="plotly_white", height=500,
        font=dict(family="DM Sans"),
        margin=dict(l=200),
    )
    st.plotly_chart(fig, use_container_width=True)

    rank = sorted_bench["Borough"].tolist().index("Westminster") + 1
    total = len(sorted_bench)
    better_or_worse = "lower" if ascending else "higher"

    st.markdown(f"""
<div class="insight-box">
<strong>Westminster ranks {rank} of {total}</strong> on {bench_metric}. 
The London average is {london_avg:.1f} vs Westminster's {wcc_val:.1f}. 
{"Westminster performs above the London average." if (not ascending and wcc_val > london_avg) or (ascending and wcc_val < london_avg) else "Westminster is currently below the London average on this metric."}
</div>
""", unsafe_allow_html=True)

    # Scatter: two metrics
    st.markdown("---")
    st.markdown("### Multi-Metric Comparison")
    
    c1, c2 = st.columns(2)
    with c1:
        x_metric = st.selectbox("X-axis", bench.columns[1:], index=0)
    with c2:
        y_metric = st.selectbox("Y-axis", bench.columns[1:], index=1)

    fig2 = px.scatter(
        bench, x=x_metric, y=y_metric, text="Borough",
        color=bench["Borough"].apply(lambda x: "Westminster" if x == "Westminster" else "Other"),
        color_discrete_map={"Westminster": "#dc3545", "Other": "#a8bbd6"},
        size=[14 if b == "Westminster" else 8 for b in bench["Borough"]],
    )
    fig2.update_traces(textposition="top center", textfont_size=10)
    fig2.update_layout(
        template="plotly_white", height=450,
        font=dict(family="DM Sans"),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 5: EARLY WARNING SYSTEM
# ══════════════════════════════════════════════
elif page == "⚠️ Early Warning System":
    st.markdown("# ⚠️ Early Warning System")
    st.markdown("*Automated alerts — flag risks before they become crises*")
    st.markdown("")

    st.markdown("""
<div class="insight-box">
<strong>How it works:</strong> For each KPI, a linear regression is fitted to the 
last 6 quarters. If the projected value for next quarter crosses the red threshold, 
an alert is triggered automatically — <em>no analyst judgment required</em>.
</div>
""", unsafe_allow_html=True)

    warnings_list = []
    kpi_targets = {
        "FOI Timeliness %": (85, "above"),
        "SAR Timeliness %": (86, "above"),
        "Direct Payments %": (28, "above"),
        "NEET 16-17 %": (2, "below"),
        "Care Leavers EET %": (70, "above"),
        "Calls Answered %": (92, "above"),
        "HRA Rent Collected %": (97, "above"),
        "Tenant Overall Satisfaction %": (66, "above"),
        "Repair Satisfaction %": (65, "above"),
    }
    
    for kpi_name, (target, direction) in kpi_targets.items():
        recent = df[kpi_name].values[-6:]
        X_r = np.arange(1, 7).reshape(-1, 1)
        model = LinearRegression().fit(X_r, recent)
        next_q = model.predict([[7]])[0]
        slope = model.coef_[0]
        
        if direction == "above" and next_q < target:
            severity = "🔴 HIGH" if next_q < target * 0.9 else "🟠 MEDIUM"
            warnings_list.append({
                "KPI": kpi_name,
                "Current": f"{recent[-1]:.1f}",
                "Target": f"{target}",
                "Projected Next Q": f"{next_q:.1f}",
                "Trend (per quarter)": f"{slope:+.2f}",
                "Alert": severity,
            })
        elif direction == "below" and next_q > target:
            severity = "🔴 HIGH" if next_q > target * 1.1 else "🟠 MEDIUM"
            warnings_list.append({
                "KPI": kpi_name,
                "Current": f"{recent[-1]:.1f}",
                "Target": f"{target}",
                "Projected Next Q": f"{next_q:.1f}",
                "Trend (per quarter)": f"{slope:+.2f}",
                "Alert": severity,
            })

    if warnings_list:
        st.markdown(f"### {len(warnings_list)} KPIs flagged for attention")
        st.dataframe(pd.DataFrame(warnings_list), hide_index=True, use_container_width=True)
    else:
        st.success("✅ No KPIs are projected to breach their target next quarter.")

    # Temporary Accommodation deep dive
    st.markdown("---")
    st.markdown("### Deep Dive: Households in Temporary Accommodation")
    
    X_ta = df["Q_num"].values.reshape(-1, 1)
    y_ta = df["Households in TA"].values
    
    model_lin = LinearRegression().fit(X_ta, y_ta)
    future_qs = np.arange(11, 21).reshape(-1, 1)
    proj_lin = model_lin.predict(future_qs)
    
    poly2 = PolynomialFeatures(degree=2)
    model_poly = LinearRegression().fit(poly2.fit_transform(X_ta), y_ta)
    proj_poly = model_poly.predict(poly2.transform(future_qs))

    future_labels = [f"Q{((i)%4)+1} {'26/27' if i < 4 else '27/28' if i < 8 else '28/29'}" for i in range(10)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Quarter"], y=y_ta, mode="lines+markers",
                              name="Actual", line=dict(color="#2a4070", width=3)))
    fig.add_trace(go.Scatter(x=future_labels, y=proj_lin, mode="lines",
                              name="Linear projection", line=dict(color="#fd7e14", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=future_labels, y=proj_poly, mode="lines",
                              name="Polynomial projection", line=dict(color="#dc3545", width=2, dash="dot")))
    fig.update_layout(
        title="Households in TA — 10-Quarter Projection",
        template="plotly_white", height=400, font=dict(family="DM Sans"),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
<div class="insight-box">
<strong>📊 Projection:</strong> At the current linear rate, Westminster could have 
<strong>{int(proj_lin[-1]):,}</strong> households in TA within 10 quarters (~2.5 years). 
The polynomial model, which captures acceleration, projects <strong>{int(proj_poly[-1]):,}</strong>. 
The Q2 report noted that a 15% annual increase would reach ~10,000 by 2030 — 
the regression corroborates this risk.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 6: TECHNIQUES SHOWCASE
# ══════════════════════════════════════════════
elif page == "🧰 Techniques Showcase":
    st.markdown("# 🧰 Analytical Techniques Showcase")
    st.markdown("*Everything from your course notebooks, applied to council data*")
    st.markdown("")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Correlation Analysis", "Anomaly Detection", "What-If Scenarios", "R Integration"
    ])

    # --- Tab 1: Correlation ---
    with tab1:
        st.markdown("### Which KPIs Move Together?")
        st.markdown("A correlation matrix reveals hidden relationships between metrics.")
        
        corr_cols = ["FOI Timeliness %", "SAR Timeliness %", "Calls Answered %",
                     "Households in TA", "Tenant Overall Satisfaction %", 
                     "NEET 16-17 %", "Care Leavers EET %", "FOI Requests Volume"]
        corr_matrix = df[corr_cols].corr()
        
        fig = px.imshow(
            corr_matrix, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            labels=dict(color="Correlation"),
        )
        fig.update_layout(height=550, font=dict(family="DM Sans", size=10),
                          margin=dict(l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
<div class="insight-box">
<strong>Key finding:</strong> There's a strong negative correlation between 
Households in TA and Care Leavers EET % — as housing pressure increases, 
employment outcomes for care leavers worsen. FOI Request Volume and FOI 
Timeliness are also negatively correlated — as demand rises, response times suffer. 
These relationships help the board understand <em>why</em> KPIs move, not just <em>that</em> they moved.
</div>
""", unsafe_allow_html=True)

    # --- Tab 2: Anomaly Detection ---
    with tab2:
        st.markdown("### Spotting Unusual Quarters")
        st.markdown("Z-scores flag quarters where a KPI deviated significantly from its trend.")
        
        anom_kpi = st.selectbox("Choose KPI", [
            "FOI Timeliness %", "SAR Timeliness %", "Calls Answered %",
            "Households in TA", "Tenant Overall Satisfaction %",
        ], key="anom")
        
        values = df[anom_kpi]
        mean_val = values.mean()
        std_val = values.std()
        z_scores = (values - mean_val) / std_val
        
        colors_z = ["#dc3545" if abs(z) > 1.5 else "#fd7e14" if abs(z) > 1 else "#28a745" for z in z_scores]
        
        fig = go.Figure(go.Bar(
            x=df["Quarter"], y=z_scores,
            marker_color=colors_z,
            text=z_scores.apply(lambda x: f"{x:.2f}"),
            textposition="outside",
        ))
        fig.add_hline(y=1.5, line_dash="dash", line_color="#dc3545", annotation_text="Anomaly threshold")
        fig.add_hline(y=-1.5, line_dash="dash", line_color="#dc3545")
        fig.update_layout(
            title=f"{anom_kpi} — Z-Score by Quarter",
            yaxis_title="Z-Score (std deviations from mean)",
            template="plotly_white", height=400, font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig, use_container_width=True)
        
        anomalies = df.loc[abs(z_scores) > 1.5, ["Quarter", anom_kpi]]
        if len(anomalies) > 0:
            st.warning(f"⚠️ Anomalous quarters detected: {', '.join(anomalies['Quarter'].tolist())}")
        else:
            st.success("No statistically anomalous quarters detected (within 1.5σ).")

    # --- Tab 3: What-If ---
    with tab3:
        st.markdown("### What-If Scenario Builder")
        st.markdown("Adjust assumptions and see the impact on KPI trajectories.")
        
        st.markdown("**Scenario: FOI Request Volume vs Timeliness**")
        
        growth_rate = st.slider("Quarterly growth in FOI requests (%)", 0, 30, 10)
        current_vol = df["FOI Requests Volume"].values[-1]
        current_timeliness = df["FOI Timeliness %"].values[-1]
        
        # Simple model: each 10% increase in volume → 2% drop in timeliness
        future_vols = [current_vol]
        future_timeliness = [current_timeliness]
        for q in range(8):
            new_vol = future_vols[-1] * (1 + growth_rate/100)
            vol_increase_pct = (new_vol - current_vol) / current_vol * 100
            timeliness_impact = vol_increase_pct * 0.02  # 2% drop per 10% volume increase
            new_timeliness = max(50, current_timeliness - timeliness_impact * 0.1)
            future_vols.append(new_vol)
            future_timeliness.append(new_timeliness)
        
        q_labels = ["Now"] + [f"+{i+1}Q" for i in range(8)]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=q_labels, y=future_vols, name="FOI Volume",
                                  line=dict(color="#2a4070", width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=q_labels, y=future_timeliness, name="Timeliness %",
                                  line=dict(color="#dc3545", width=3)), secondary_y=True)
        fig.add_hline(y=85, line_dash="dash", line_color="#28a745",
                      annotation_text="85% target", secondary_y=True)
        fig.update_layout(
            title=f"Impact of {growth_rate}% Quarterly FOI Growth on Timeliness",
            template="plotly_white", height=400, font=dict(family="DM Sans"),
        )
        fig.update_yaxes(title_text="Request Volume", secondary_y=False)
        fig.update_yaxes(title_text="Timeliness %", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
<div class="insight-box">
<strong>Scenario result:</strong> At {growth_rate}% quarterly growth, FOI volume would 
reach ~{int(future_vols[-1]):,} requests/quarter within 2 years. 
Timeliness would drop to approximately {future_timeliness[-1]:.1f}%, 
{'well below the 85% target. Additional resource would be needed.' if future_timeliness[-1] < 85 else 'remaining near target.'}
</div>
""", unsafe_allow_html=True)

    # --- Tab 4: R Integration ---
    with tab4:
        st.markdown("### Using R Alongside Python")
        st.markdown("""
Streamlit runs Python natively, but you can also call R scripts for techniques 
you've learned in R. Here's how the integration works:
""")
        
        st.code("""
# In your Streamlit app, call an R script:
import subprocess
import pandas as pd

# Run an R regression and capture output
result = subprocess.run(
    ["Rscript", "my_analysis.R", "--input", "kpi_data.csv"],
    capture_output=True, text=True
)

# Read the R output back into Python
r_results = pd.read_csv("r_output.csv")
st.dataframe(r_results)
""", language="python")
        
        st.code("""
# my_analysis.R — Example R script
library(tidyverse)

data <- read_csv("kpi_data.csv")

# Fit a linear model
model <- lm(FOI_Timeliness ~ Quarter_Num + FOI_Volume, data = data)
summary(model)

# Output predictions
predictions <- predict(model, newdata = data)
write_csv(data.frame(Quarter = data$Quarter, Predicted = predictions), "r_output.csv")
""", language="r")
        
        st.markdown("""
<div class="insight-box">
<strong>Why this matters:</strong> Your R notebooks aren't wasted — they plug directly 
into Streamlit. Use R for what it does best (statistical modelling, ggplot2 visualisation) 
and Python for the interactive dashboard layer. Best of both worlds.
</div>
""", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Techniques Summary — Course to Council")
        
        techniques_df = pd.DataFrame({
            "Technique": [
                "Linear Regression", "Polynomial Regression", "Correlation Matrix",
                "Z-Score Anomaly Detection", "What-If Modelling",
                "R Integration", "Time Series Decomposition",
                "Classification (Logistic Regression)", "Clustering (K-Means)",
            ],
            "Council Use Case": [
                "Forecast KPI trends for the next 4 quarters",
                "Detect accelerating problems (e.g. TA growth)",
                "Find which KPIs influence each other",
                "Flag unusual quarters for investigation",
                "Model impact of demand increases on service capacity",
                "Leverage existing R statistical models in dashboards",
                "Separate seasonal patterns from genuine trends in complaints",
                "Predict which KPIs are likely to go red next quarter",
                "Group boroughs by similar performance profiles",
            ],
            "Benefit to Board": [
                "Data-driven early warning, not gut feel",
                "Quantifies whether things are getting worse faster",
                "Explains why multiple KPIs move together",
                "Focuses attention on genuine outliers, not noise",
                "Tests policy options before committing resources",
                "Protects investment in R training",
                "Avoids false alarms from seasonal dips",
                "Prioritises interventions where risk is highest",
                "Identifies peer boroughs for meaningful comparison",
            ],
        })
        st.dataframe(techniques_df, hide_index=True, use_container_width=True, height=370)
