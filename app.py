"""
app.py - Smart Fault Diagnosis System (Groq + Streamlit, light theme)
Run:  streamlit run app.py
"""

from __future__ import annotations
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from chatbot          import FaultChatbot
from data_processor    import (build_data_summary, compute_column_stats,
                                detect_anomalies, load_file, preprocess)
from diagnosis_engine  import DiagnosisEngine, DiagnosisResult
from report_generator  import (generate_csv_report, generate_json_report,
                                generate_markdown_summary, generate_pdf_report)
from utils              import (badge, health_score_color, health_score_label,
                                 init_session_defaults, now_str, severity_color,
                                 severity_icon, timestamp_slug)
from visualizations     import (anomaly_highlight_chart, correlation_heatmap,
                                 fault_history_chart, fault_pie, health_gauge,
                                 priority_bar, rolling_stats_chart, sensor_trend_chart)

# -- Page config --------------------------------------------------------------
st.set_page_config(
    page_title = "Smart Fault Diagnosis",
    page_icon  = "⚙️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# -- Light theme CSS ------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"], .stApp {
    font-family : 'Inter', sans-serif !important;
    background  : #F1F5F9 !important;
    color       : #0F172A !important;
}

.main .block-container {
    background    : #F1F5F9 !important;
    padding-top   : 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width     : 1280px !important;
}

[data-testid="stSidebar"] {
    background    : #FFFFFF !important;
    border-right  : 1px solid #E2E8F0 !important;
    box-shadow    : 2px 0 8px rgba(0,0,0,0.04) !important;
}
[data-testid="stSidebar"] * { color: #0F172A !important; }
[data-testid="stSidebar"] .stButton>button {
    background    : #2563EB !important;
    color         : #FFFFFF !important;
}
[data-testid="stSidebar"] .stButton>button:hover {
    background    : #1D4ED8 !important;
}

.card {
    background    : #FFFFFF;
    border        : 1px solid #E2E8F0;
    border-radius : 12px;
    padding       : 20px 24px;
    margin-bottom : 16px;
    box-shadow    : 0 1px 4px rgba(0,0,0,0.06);
}
.card-blue  { border-left: 4px solid #2563EB; }
.card-red   { border-left: 4px solid #DC2626; }
.card-green { border-left: 4px solid #16A34A; }
.card-amber { border-left: 4px solid #D97706; }

.kpi-box {
    background    : #FFFFFF;
    border        : 1px solid #E2E8F0;
    border-radius : 12px;
    padding       : 18px 20px;
    text-align    : center;
    box-shadow    : 0 1px 4px rgba(0,0,0,0.06);
}
.kpi-value { font-size: 2rem; font-weight: 700; line-height: 1.1; }
.kpi-label { font-size: 0.78rem; color: #64748B; margin-top: 4px; font-weight: 500; letter-spacing: 0.3px; }

.stButton>button {
    background    : #2563EB !important;
    color         : #FFFFFF !important;
    border        : none !important;
    border-radius : 8px !important;
    font-weight   : 600 !important;
    font-size     : 0.88rem !important;
    padding       : 0.45rem 1.2rem !important;
    transition    : all 0.18s ease !important;
    box-shadow    : 0 1px 3px rgba(37,99,235,0.3) !important;
}
.stButton>button:hover {
    background    : #1D4ED8 !important;
    box-shadow    : 0 4px 12px rgba(37,99,235,0.35) !important;
    transform     : translateY(-1px) !important;
}
.stButton>button:disabled {
    background    : #CBD5E1 !important;
    box-shadow    : none !important;
    transform     : none !important;
}

[data-baseweb="tab-list"] {
    background    : #FFFFFF !important;
    border        : 1px solid #E2E8F0 !important;
    border-radius : 10px !important;
    padding       : 5px !important;
    gap           : 3px !important;
    box-shadow    : 0 1px 3px rgba(0,0,0,0.05) !important;
}
[data-baseweb="tab"] {
    font-size     : 0.82rem !important;
    font-weight   : 500 !important;
    color         : #64748B !important;
    border-radius : 7px !important;
    padding       : 6px 14px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background    : #2563EB !important;
    color         : #FFFFFF !important;
    font-weight   : 600 !important;
}

[data-testid="metric-container"] {
    background    : #FFFFFF !important;
    border        : 1px solid #E2E8F0 !important;
    border-radius : 10px !important;
    padding       : 14px 18px !important;
    box-shadow    : 0 1px 3px rgba(0,0,0,0.05) !important;
}
[data-testid="metric-container"] label { color: #64748B !important; font-size: 0.8rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #0F172A !important; font-weight: 700 !important; }

[data-testid="stFileUploader"] {
    background    : #EFF6FF !important;
    border        : 2px dashed #93C5FD !important;
    border-radius : 10px !important;
}
[data-testid="stFileUploader"] * { color: #1E40AF !important; }

.stTextInput>div>div>input,
.stTextArea textarea,
.stSelectbox>div>div {
    background    : #FFFFFF !important;
    border        : 1px solid #CBD5E1 !important;
    border-radius : 8px !important;
    color         : #0F172A !important;
    font-size     : 0.9rem !important;
}
.stTextInput>div>div>input:focus,
.stTextArea textarea:focus {
    border-color  : #2563EB !important;
    box-shadow    : 0 0 0 2px rgba(37,99,235,0.15) !important;
}

[data-testid="stExpander"] {
    background    : #FFFFFF !important;
    border        : 1px solid #E2E8F0 !important;
    border-radius : 10px !important;
    box-shadow    : 0 1px 3px rgba(0,0,0,0.04) !important;
}
[data-testid="stExpander"] summary {
    color         : #0F172A !important;
    font-weight   : 500 !important;
}

[data-testid="stDataFrame"] {
    border        : 1px solid #E2E8F0 !important;
    border-radius : 10px !important;
    overflow      : hidden !important;
}

.stSuccess { background: #F0FDF4 !important; border: 1px solid #BBF7D0 !important; color: #15803D !important; border-radius: 8px !important; }
.stWarning { background: #FFFBEB !important; border: 1px solid #FDE68A !important; color: #92400E !important; border-radius: 8px !important; }
.stError   { background: #FEF2F2 !important; border: 1px solid #FECACA !important; color: #991B1B !important; border-radius: 8px !important; }
.stInfo    { background: #EFF6FF !important; border: 1px solid #BFDBFE !important; color: #1E40AF !important; border-radius: 8px !important; }

.chat-user {
    background    : #2563EB;
    color         : #FFFFFF;
    border-radius : 16px 16px 4px 16px;
    padding       : 10px 14px;
    margin        : 8px 0 8px 25%;
    font-size     : 0.88rem;
    line-height   : 1.55;
    box-shadow    : 0 1px 4px rgba(37,99,235,0.2);
}
.chat-bot {
    background    : #FFFFFF;
    color         : #0F172A;
    border        : 1px solid #E2E8F0;
    border-radius : 16px 16px 16px 4px;
    padding       : 10px 14px;
    margin        : 8px 25% 8px 0;
    font-size     : 0.88rem;
    line-height   : 1.55;
    box-shadow    : 0 1px 4px rgba(0,0,0,0.05);
}
.chat-welcome {
    background    : #EFF6FF;
    border        : 1px solid #BFDBFE;
    border-radius : 12px;
    padding       : 16px 20px;
    color         : #1E40AF;
    font-size     : 0.88rem;
    line-height   : 1.6;
    margin-bottom : 12px;
}

.sec-label {
    font-size     : 0.7rem;
    font-weight   : 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color         : #2563EB;
    margin-bottom : 8px;
}

.fault-item {
    background    : #FFFFFF;
    border        : 1px solid #FCA5A5;
    border-left   : 4px solid #DC2626;
    border-radius : 8px;
    padding       : 12px 16px;
    margin        : 6px 0;
    box-shadow    : 0 1px 3px rgba(0,0,0,0.04);
}
.fix-item {
    background    : #FFFFFF;
    border        : 1px solid #E2E8F0;
    border-radius : 8px;
    padding       : 12px 16px;
    margin        : 6px 0;
    box-shadow    : 0 1px 3px rgba(0,0,0,0.04);
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# -- Session state --------------------------------------------------------------
init_session_defaults({
    "df": None, "raw_text": "", "file_type": "", "filename": "",
    "data_summary": "", "statistics": "", "anomalies": {},
    "diagnosis": None, "fault_history": [], "chatbot": None,
})

# -- Helpers ----------------------------------------------------------------------
def get_chatbot() -> FaultChatbot:
    if st.session_state.chatbot is None:
        st.session_state.chatbot = FaultChatbot()
    return st.session_state.chatbot

def push_history(d: DiagnosisResult, fname: str):
    st.session_state.fault_history.append({
        "timestamp": now_str("%Y-%m-%d %H:%M"),
        "filename": fname,
        "health_score": d.health_score,
        "severity": d.severity,
        "faults": len(d.faults_detected),
    })

def _api_key_ok() -> bool:
    k = os.getenv("GROQ_API_KEY", "").strip()
    return bool(k) and k != "your_groq_api_key_here"

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 12px;text-align:center'>
      <div style='font-size:2.2rem'>⚙️</div>
      <div style='font-size:1rem;font-weight:700;color:#2563EB;letter-spacing:0.5px;margin-top:4px'>
        Fault Diagnosis AI
      </div>
      <div style='font-size:0.72rem;color:#94A3B8;margin-top:2px'>Industrial Monitoring System</div>
    </div>
    <hr style='border:none;border-top:1px solid #E2E8F0;margin:4px 0 16px'>
    """, unsafe_allow_html=True)

    # API key status only - never show the actual key value
    if _api_key_ok():
        st.markdown("""
        <div style='background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;
                    padding:8px 12px;margin-bottom:16px;display:flex;align-items:center;gap:8px'>
          <span style='font-size:0.85rem'>✅</span>
          <span style='font-size:0.8rem;color:#15803D;font-weight:500'>API key configured</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#FEF2F2;border:1px solid #FECACA;border-radius:8px;
                    padding:8px 12px;margin-bottom:4px'>
          <div style='font-size:0.82rem;color:#991B1B;font-weight:600'>⚠️ API key not set</div>
          <div style='font-size:0.76rem;color:#B91C1C;margin-top:3px'>
            Run: <code style='background:#FEE2E2;padding:1px 5px;border-radius:3px'>python setup_env.py</code>
          </div>
        </div>
        <div style='font-size:0.74rem;color:#64748B;margin-bottom:14px;padding:0 4px'>
          Free key at <a href='https://console.groq.com' target='_blank' style='color:#2563EB'>console.groq.com</a>
        </div>
        """, unsafe_allow_html=True)

    # File upload
    st.markdown('<div class="sec-label">📁 Upload Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload machine data",
        type=["csv","xlsx","xls","json","txt","log"],
        label_visibility="collapsed",
        help="Supports CSV, Excel, JSON, TXT/LOG files",
    )

    if uploaded:
        with st.spinner("Loading..."):
            df, raw_text, file_type = load_file(uploaded)
            if df is not None:
                df = preprocess(df)
            anoms = detect_anomalies(df) if df is not None else {
                "anomaly_columns":[], "anomaly_details":[], "anomaly_summary":"N/A"
            }
            st.session_state.update({
                "df": df, "raw_text": raw_text, "file_type": file_type,
                "filename": uploaded.name,
                "data_summary": build_data_summary(df, raw_text, file_type),
                "statistics": compute_column_stats(df) if df is not None else "",
                "anomalies": anoms,
            })
        st.success(f"✅ {uploaded.name}")
        if df is not None:
            st.caption(f"📊 {df.shape[0]:,} rows · {df.shape[1]} columns · "
                       f"{len(anoms['anomaly_columns'])} anomaly col(s)")

    st.markdown('<hr style="border:none;border-top:1px solid #E2E8F0;margin:12px 0">', unsafe_allow_html=True)

    # Diagnose button
    st.markdown('<div class="sec-label">🔬 Analysis</div>', unsafe_allow_html=True)
    can_diagnose = bool(st.session_state.filename) and _api_key_ok()
    if st.button("🚀 Run Fault Diagnosis", use_container_width=True, disabled=not can_diagnose):
        try:
            engine = DiagnosisEngine()
            with st.spinner("AI is analysing your machine data..."):
                diag = engine.diagnose(
                    data_summary    = st.session_state.data_summary,
                    statistics      = st.session_state.statistics,
                    anomaly_summary = st.session_state.anomalies.get("anomaly_summary",""),
                )
            if diag.error:
                st.error(f"❌ {diag.error}")
            else:
                st.session_state.diagnosis = diag
                push_history(diag, st.session_state.filename)
                st.success("✅ Diagnosis complete!")
                st.balloons()
        except EnvironmentError as exc:
            st.error(str(exc))

    if not _api_key_ok():
        st.caption("⚠️ Set API key first")
    elif not st.session_state.filename:
        st.caption("⚠️ Upload a file first")

    st.markdown('<hr style="border:none;border-top:1px solid #E2E8F0;margin:12px 0">', unsafe_allow_html=True)

    # Health score mini card
    d: DiagnosisResult | None = st.session_state.diagnosis
    if d and not d.error:
        sc    = d.health_score
        color = health_score_color(sc)
        lbl   = health_score_label(sc)
        st.markdown(f"""
        <div style='background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;
                    padding:14px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.06)'>
          <div style='font-size:2.4rem;font-weight:700;color:{color};line-height:1'>{sc}</div>
          <div style='font-size:0.75rem;color:#64748B;margin:2px 0 6px'>Health Score / 100</div>
          <div style='display:inline-block;background:{color}22;color:{color};
                      border-radius:20px;padding:2px 12px;font-size:0.8rem;font-weight:600'>{lbl}</div>
          <div style='margin-top:8px;font-size:0.82rem;color:#475569'>
            {severity_icon(d.severity)} {d.severity} severity
            &nbsp;·&nbsp; {len(d.faults_detected)} fault(s)
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:1px solid #E2E8F0;margin:12px 0">', unsafe_allow_html=True)

    if st.button("🗑️ Reset Session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown("""
    <div style='margin-top:20px;text-align:center;font-size:0.7rem;color:#CBD5E1;padding-bottom:8px'>
      Powered by Groq · LLaMA 3.3 70B<br>Smart Industrial AI v3.1
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# MAIN HEADER
# ==============================================================================
st.markdown("""
<div style='background:linear-gradient(135deg,#1E40AF 0%,#2563EB 60%,#3B82F6 100%);
            border-radius:14px;padding:22px 28px;margin-bottom:22px;
            box-shadow:0 4px 20px rgba(37,99,235,0.25)'>
  <h1 style='margin:0;color:#FFFFFF;font-size:1.55rem;font-weight:700;letter-spacing:-0.3px'>
    ⚙️ Smart Fault Diagnosis System
  </h1>
  <p style='margin:6px 0 0;color:#BFDBFE;font-size:0.85rem'>
    Upload industrial sensor data · AI detects faults · Get actionable maintenance insights
  </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["📊 Dashboard","🔬 Diagnosis","📈 Sensor Analytics","💬 AI Assistant","📋 History","⬇️ Export"])

# ==============================================================================
# TAB 1 - DASHBOARD
# ==============================================================================
with tabs[0]:
    diag: DiagnosisResult | None = st.session_state.diagnosis
    df: pd.DataFrame | None      = st.session_state.df

    if not st.session_state.filename:
        st.markdown("""
        <div style='text-align:center;padding:50px 20px 40px'>
          <div style='font-size:4rem;margin-bottom:12px'>🏭</div>
          <h2 style='color:#1E40AF;font-size:1.6rem;margin:0 0 8px'>Welcome to Fault Diagnosis AI</h2>
          <p style='color:#64748B;max-width:500px;margin:0 auto 32px;font-size:0.92rem;line-height:1.65'>
            Upload your machine sensor data, maintenance logs, or error reports
            using the sidebar panel, then click <strong>Run Fault Diagnosis</strong>.
          </p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, icon, title, desc in [
            (c1,"📂","Upload Data",    "CSV, Excel, JSON, TXT log files"),
            (c2,"🤖","AI Analysis",   "LLaMA 3.3 70B via Groq API"),
            (c3,"📊","Visual Reports","Interactive charts & dashboards"),
            (c4,"📄","Export",        "PDF, JSON, CSV, Markdown"),
        ]:
            col.markdown(f"""
            <div class='card' style='text-align:center;padding:20px 16px'>
              <div style='font-size:1.8rem;margin-bottom:8px'>{icon}</div>
              <div style='font-weight:600;color:#1E40AF;margin-bottom:4px;font-size:0.9rem'>{title}</div>
              <div style='font-size:0.78rem;color:#64748B'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        c1, c2, c3, c4 = st.columns(4)
        if diag and not diag.error:
            sc = diag.health_score
            c1.markdown(f"""
            <div class='kpi-box'>
              <div class='kpi-value' style='color:{health_score_color(sc)}'>{sc}<span style='font-size:1rem;color:#94A3B8'>/100</span></div>
              <div class='kpi-label'>Health Score</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""
            <div class='kpi-box'>
              <div class='kpi-value' style='color:{severity_color(diag.severity)}'>{diag.severity}</div>
              <div class='kpi-label'>Severity Level</div>
            </div>""", unsafe_allow_html=True)
            c3.markdown(f"""
            <div class='kpi-box'>
              <div class='kpi-value' style='color:#DC2626'>{len(diag.faults_detected)}</div>
              <div class='kpi-label'>Faults Detected</div>
            </div>""", unsafe_allow_html=True)
            c4.markdown(f"""
            <div class='kpi-box'>
              <div class='kpi-value' style='color:#D97706'>{len(diag.recommended_fixes)}</div>
              <div class='kpi-label'>Actions Required</div>
            </div>""", unsafe_allow_html=True)
        else:
            short_name = st.session_state.filename
            if len(short_name) > 18:
                short_name = short_name[:18] + "..."
            c1.metric("📂 File",    short_name)
            c2.metric("📊 Rows",    f"{df.shape[0]:,}" if df is not None else "—")
            c3.metric("📋 Columns", df.shape[1] if df is not None else "—")
            c4.metric("⚠️ Anomalies", len(st.session_state.anomalies.get("anomaly_columns",[])))

        st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

        if diag and not diag.error:
            col_g, col_p = st.columns([1,1])
            with col_g:
                st.plotly_chart(health_gauge(diag.health_score), use_container_width=True)
            with col_p:
                st.plotly_chart(fault_pie(diag.faults_detected), use_container_width=True)

            st.markdown(f"""
            <div class='card card-blue'>
              <div class='sec-label'>Executive Summary</div>
              <p style='margin:0;color:#334155;line-height:1.7;font-size:0.9rem'>{diag.summary}</p>
            </div>
            """, unsafe_allow_html=True)

            if diag.faults_detected:
                st.markdown('<div class="sec-label" style="margin-top:8px">Detected Faults</div>', unsafe_allow_html=True)
                top = diag.faults_detected[:3]
                cols = st.columns(len(top))
                for col, f in zip(cols, top):
                    conf  = f.get("confidence","Low")
                    bcol  = "#DC2626" if conf=="High" else "#D97706" if conf=="Medium" else "#16A34A"
                    desc  = f.get("description","")
                    short = desc[:120] + ("..." if len(desc) > 120 else "")
                    col.markdown(f"""
                    <div class='card card-red' style='height:100%'>
                      <div style='font-weight:600;color:#0F172A;margin-bottom:4px;font-size:0.88rem'>
                        {f.get('fault_name','Unknown')}
                      </div>
                      {badge(conf, bcol)}
                      <p style='margin:8px 0 0;font-size:0.8rem;color:#475569;line-height:1.5'>
                        {short}
                      </p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            if df is not None:
                st.markdown('<div class="sec-label">Data Preview</div>', unsafe_allow_html=True)
                st.dataframe(df.head(25), use_container_width=True, height=360)
                anom = st.session_state.anomalies.get("anomaly_details",[])
                if anom:
                    st.warning(f"⚠️  {len(anom)} column(s) show anomalies. Run diagnosis to analyse.")
                else:
                    st.info("✅ No statistical anomalies found. Run diagnosis for full AI analysis.")
            elif st.session_state.raw_text:
                st.markdown('<div class="sec-label">Log Preview</div>', unsafe_allow_html=True)
                st.text_area("", st.session_state.raw_text[:3000], height=340, disabled=True, label_visibility="collapsed")

# ==============================================================================
# TAB 2 - DIAGNOSIS REPORT
# ==============================================================================
with tabs[1]:
    diag = st.session_state.diagnosis

    if diag is None:
        st.info("⚙️ Upload a file and click **Run Fault Diagnosis** to see the report.")
    elif diag.error:
        st.error(f"❌ {diag.error}")
        if diag.raw_response:
            with st.expander("🔍 Raw model response (debug)"):
                st.code(diag.raw_response[:1500])
    else:
        col_h, col_score = st.columns([3,1])
        with col_h:
            st.markdown(f"""
            <div style='margin-bottom:4px'>
              <h2 style='margin:0;font-size:1.35rem;color:#0F172A;font-weight:700'>
                Fault Diagnosis Report
              </h2>
              <span style='font-size:0.8rem;color:#64748B'>
                📁 {st.session_state.filename} &nbsp;·&nbsp; 🕐 {now_str()}
              </span>
            </div>
            """, unsafe_allow_html=True)
        with col_score:
            sc = diag.health_score
            c  = health_score_color(sc)
            st.markdown(f"""
            <div class='kpi-box'>
              <div class='kpi-value' style='color:{c}'>{sc}<span style='font-size:0.9rem;color:#94A3B8'>/100</span></div>
              <div class='kpi-label'>{health_score_label(sc)} · {diag.severity}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin:12px 0'></div>", unsafe_allow_html=True)

        st.markdown('<div class="sec-label">Faults Detected</div>', unsafe_allow_html=True)
        if diag.faults_detected:
            for i, f in enumerate(diag.faults_detected, 1):
                conf  = f.get("confidence","Low")
                bcol  = "#DC2626" if conf=="High" else "#D97706" if conf=="Medium" else "#16A34A"
                with st.expander(f"🔴 {i}. {f.get('fault_name','Unknown')}", expanded=(i==1)):
                    st.markdown(badge(conf, bcol), unsafe_allow_html=True)
                    st.markdown(f"**Description:** {f.get('description','')}")
                    params = f.get("affected_parameters",[])
                    if params:
                        st.markdown("**Affected parameters:** " + " · ".join([f"`{p}`" for p in params]))
        else:
            st.success("✅ No faults detected in the uploaded data.")

        if diag.root_causes:
            st.markdown('<div class="sec-label" style="margin-top:16px">Root Cause Analysis</div>', unsafe_allow_html=True)
            for rc in diag.root_causes:
                with st.expander(f"🔍 {rc.get('cause','?')}"):
                    st.markdown(f"**Explanation:** {rc.get('explanation','')}")
                    factors = rc.get("contributing_factors",[])
                    if factors:
                        st.markdown("**Contributing factors:**")
                        for cf in factors:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;• {cf}")

        st.markdown("<div style='margin:12px 0'></div>", unsafe_allow_html=True)

        st.markdown('<div class="sec-label">Recommended Corrective Actions</div>', unsafe_allow_html=True)
        pmap = {"Immediate":"#DC2626","Within 24h":"#EA580C","Within 1 week":"#D97706","Scheduled":"#16A34A"}
        for fx in diag.recommended_fixes:
            p = fx.get("priority","Scheduled")
            c = pmap.get(p,"#6B7280")
            st.markdown(f"""
            <div class='fix-item' style='border-left:4px solid {c}'>
              <span style='background:{c}22;color:{c};border-radius:20px;padding:2px 10px;
                           font-size:0.74rem;font-weight:600'>{p}</span>
              <span style='font-weight:600;margin-left:8px;color:#0F172A;font-size:0.9rem'>
                {fx.get('action','')}
              </span>
              <div style='margin-top:5px;font-size:0.8rem;color:#64748B'>
                ⏱ {fx.get('estimated_downtime','?')} &nbsp;·&nbsp; 🛠 {fx.get('resources_needed','?')}
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin:12px 0'></div>", unsafe_allow_html=True)

        c_risk, c_pred = st.columns(2)

        with c_risk:
            st.markdown('<div class="sec-label">Risk Assessment</div>', unsafe_allow_html=True)
            ra = diag.risk_assessment
            if ra:
                rows = ""
                for key, lbl in [("overall_risk","Overall Risk"),("safety_risk","Safety"),
                                   ("production_impact","Production Impact"),
                                   ("financial_impact","Financial"),("mtbf_estimate","MTBF")]:
                    val = ra.get(key)
                    if val:
                        rows += (f"<div style='margin-bottom:8px'>"
                                 f"<span style='font-size:0.75rem;color:#64748B;font-weight:500'>{lbl.upper()}</span><br>"
                                 f"<span style='font-weight:600;color:#0F172A'>{val}</span></div>")
                st.markdown(f"<div class='card'>{rows}</div>", unsafe_allow_html=True)

        with c_pred:
            st.markdown('<div class="sec-label">Failure Prediction</div>', unsafe_allow_html=True)
            fp = diag.failure_prediction
            if fp:
                modes_html = ""
                if fp.get("failure_modes"):
                    modes_html = ("<div style='margin-top:8px;font-size:0.82rem'><b>Failure modes:</b><br>"
                                  + "<br>".join(f"• {m}" for m in fp.get('failure_modes',[])[:4]) + "</div>")
                st.markdown(f"""
                <div class='card card-amber'>
                  <div style='margin-bottom:8px'>
                    <span style='font-size:0.75rem;color:#64748B;font-weight:500'>TIME TO FAILURE</span><br>
                    <span style='font-weight:700;color:#D97706;font-size:1.1rem'>{fp.get('estimated_time_to_failure','?')}</span>
                  </div>
                  <div style='font-size:0.78rem;color:#64748B'>Confidence: <b>{fp.get('confidence','?')}</b></div>
                  {modes_html}
                </div>""", unsafe_allow_html=True)

        c_prev, c_sched = st.columns(2)
        with c_prev:
            st.markdown('<div class="sec-label" style="margin-top:12px">Preventive Actions</div>', unsafe_allow_html=True)
            for pa in diag.preventive_actions:
                st.markdown(f"✔️ {pa}")
        with c_sched:
            st.markdown('<div class="sec-label" style="margin-top:12px">Maintenance Schedule</div>', unsafe_allow_html=True)
            if diag.maintenance_schedule:
                st.dataframe(pd.DataFrame(diag.maintenance_schedule), use_container_width=True, hide_index=True)

        if diag.recommended_fixes:
            st.plotly_chart(priority_bar(diag.recommended_fixes), use_container_width=True)

# ==============================================================================
# TAB 3 - SENSOR ANALYTICS
# ==============================================================================
with tabs[2]:
    df = st.session_state.df

    if df is None:
        if st.session_state.raw_text:
            st.info("📝 Text/log file loaded — sensor charts are only available for tabular data (CSV/Excel/JSON).")
            with st.expander("View log content"):
                st.text_area("", st.session_state.raw_text[:4000], height=380, disabled=True, label_visibility="collapsed")
        else:
            st.info("📂 Upload a CSV, Excel, or JSON file to view sensor analytics.")
    else:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            st.warning("No numeric columns found for charting.")
        else:
            st.markdown('<div class="sec-label">Sensor Trend</div>', unsafe_allow_html=True)
            sel = st.multiselect("Select columns to plot",
                                 options=num_cols,
                                 default=num_cols[:min(5,len(num_cols))],
                                 key="trend_sel")
            if sel:
                st.plotly_chart(sensor_trend_chart(df, sel), use_container_width=True)

            st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)

            st.markdown('<div class="sec-label">Anomaly Inspection</div>', unsafe_allow_html=True)
            anom_cols = st.session_state.anomalies.get("anomaly_columns",[])
            anom_sel  = st.selectbox("Column to inspect",
                                     options=anom_cols if anom_cols else num_cols,
                                     key="anom_col")
            if anom_sel:
                st.plotly_chart(anomaly_highlight_chart(df, anom_sel), use_container_width=True)

            st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)

            c_roll, c_corr = st.columns(2)
            with c_roll:
                st.markdown('<div class="sec-label">Rolling Statistics</div>', unsafe_allow_html=True)
                r_col = st.selectbox("Column", num_cols, key="roll_col")
                win   = st.slider("Window size", 3, 50, 10, key="roll_win")
                st.plotly_chart(rolling_stats_chart(df, r_col, win), use_container_width=True)
            with c_corr:
                st.markdown('<div class="sec-label">Correlation Matrix</div>', unsafe_allow_html=True)
                if len(num_cols) >= 2:
                    st.plotly_chart(correlation_heatmap(df), use_container_width=True)
                else:
                    st.info("Need 2+ numeric columns for correlation.")

            anom_det = st.session_state.anomalies.get("anomaly_details",[])
            if anom_det:
                st.markdown('<div class="sec-label" style="margin-top:12px">Anomaly Summary Table</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(anom_det), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 4 - AI CHATBOT
# ==============================================================================
with tabs[3]:
    st.markdown("""
    <div style='margin-bottom:16px'>
      <h3 style='margin:0;font-size:1.15rem;color:#1E40AF;font-weight:700'>💬 AI Maintenance Assistant</h3>
      <p style='margin:4px 0 0;font-size:0.82rem;color:#64748B'>
        Ask anything about the diagnosed faults, root causes, or maintenance steps.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not _api_key_ok():
        st.warning("⚠️ Configure your Groq API key first. Run `python setup_env.py`")
    else:
        chatbot = get_chatbot()
        history = chatbot.get_history()
        diag    = st.session_state.diagnosis

        with st.container():
            if not history:
                st.markdown("""
                <div class='chat-welcome'>
                  👋 <b>Hello! I'm your Fault Diagnosis Assistant.</b><br><br>
                  After running a diagnosis, you can ask me:<br>
                  &nbsp;• <i>"What are the critical faults?"</i><br>
                  &nbsp;• <i>"How do I fix the bearing fault?"</i><br>
                  &nbsp;• <i>"Is it safe to keep running the machine?"</i><br>
                  &nbsp;• <i>"What maintenance should I do this week?"</i><br>
                  &nbsp;• <i>"Explain the health score."</i>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in history:
                    txt = msg["content"].replace("\n","<br>")
                    if msg["role"] == "user":
                        st.markdown(f'<div class="chat-user">{txt}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-bot">{txt}</div>', unsafe_allow_html=True)

        if diag and not getattr(diag,"error",True):
            st.markdown('<div class="sec-label" style="margin-top:12px">Quick Questions</div>', unsafe_allow_html=True)
            suggs = ["What are the critical faults?","Is it safe to run?",
                     "What needs immediate action?","Explain the health score","What caused this?"]
            cols  = st.columns(len(suggs))
            for i, (col, s) in enumerate(zip(cols, suggs)):
                if col.button(s[:28], key=f"s{i}", use_container_width=True):
                    with st.spinner("Thinking..."):
                        chatbot.chat(s, diag, st.session_state.data_summary[:600])
                    st.rerun()

        with st.form("chat_form", clear_on_submit=True):
            user_in = st.text_input("", placeholder="Type your question here...", label_visibility="collapsed")
            c1, c2  = st.columns([5,1])
            send    = c1.form_submit_button("Send ➤", use_container_width=True)
            clear   = c2.form_submit_button("Clear",  use_container_width=True)

        if send and user_in.strip():
            with st.spinner("Assistant is thinking..."):
                chatbot.chat(user_in.strip(), diag, st.session_state.data_summary[:600])
            st.rerun()

        if clear:
            chatbot.clear_history()
            st.rerun()

# ==============================================================================
# TAB 5 - HISTORY
# ==============================================================================
with tabs[4]:
    st.markdown('<div class="sec-label">Fault Analysis History</div>', unsafe_allow_html=True)
    hist = st.session_state.fault_history

    if not hist:
        st.info("No diagnoses recorded yet. Run your first analysis to see history here.")
    else:
        st.dataframe(pd.DataFrame(hist), use_container_width=True, hide_index=True)
        st.plotly_chart(fault_history_chart(hist), use_container_width=True)
        if st.button("🗑️ Clear History"):
            st.session_state.fault_history = []
            st.rerun()

# ==============================================================================
# TAB 6 - EXPORT
# ==============================================================================
with tabs[5]:
    st.markdown('<div class="sec-label">Download Reports</div>', unsafe_allow_html=True)
    diag     = st.session_state.diagnosis
    filename = st.session_state.filename or "machine"

    if not diag or diag.error:
        st.info("Run a successful diagnosis first to enable exports.")
    else:
        slug = timestamp_slug()
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown("""
            <div class='card' style='text-align:center;padding:20px 14px;min-height:160px'>
              <div style='font-size:1.8rem;margin-bottom:8px'>📄</div>
              <div style='font-weight:600;color:#0F172A;margin-bottom:4px'>PDF Report</div>
              <div style='font-size:0.78rem;color:#64748B;margin-bottom:14px'>Professional printable report</div>
            </div>""", unsafe_allow_html=True)
            pdf_b = generate_pdf_report(diag, filename)
            st.download_button("⬇️ Download PDF", data=pdf_b,
                               file_name=f"fault_report_{slug}.pdf",
                               mime="application/pdf", use_container_width=True)

        with c2:
            st.markdown("""
            <div class='card' style='text-align:center;padding:20px 14px;min-height:160px'>
              <div style='font-size:1.8rem;margin-bottom:8px'>🗂️</div>
              <div style='font-weight:600;color:#0F172A;margin-bottom:4px'>JSON Report</div>
              <div style='font-size:0.78rem;color:#64748B;margin-bottom:14px'>Structured machine-readable data</div>
            </div>""", unsafe_allow_html=True)
            json_b = generate_json_report(diag, filename)
            st.download_button("⬇️ Download JSON", data=json_b,
                               file_name=f"fault_report_{slug}.json",
                               mime="application/json", use_container_width=True)

        with c3:
            st.markdown("""
            <div class='card' style='text-align:center;padding:20px 14px;min-height:160px'>
              <div style='font-size:1.8rem;margin-bottom:8px'>📊</div>
              <div style='font-weight:600;color:#0F172A;margin-bottom:4px'>CSV Export</div>
              <div style='font-size:0.78rem;color:#64748B;margin-bottom:14px'>Import into Excel or BI tools</div>
            </div>""", unsafe_allow_html=True)
            csv_b = generate_csv_report(diag)
            st.download_button("⬇️ Download CSV", data=csv_b,
                               file_name=f"fault_report_{slug}.csv",
                               mime="text/csv", use_container_width=True)

        with c4:
            st.markdown("""
            <div class='card' style='text-align:center;padding:20px 14px;min-height:160px'>
              <div style='font-size:1.8rem;margin-bottom:8px'>📝</div>
              <div style='font-weight:600;color:#0F172A;margin-bottom:4px'>Markdown</div>
              <div style='font-size:0.78rem;color:#64748B;margin-bottom:14px'>For wikis, Notion, docs</div>
            </div>""", unsafe_allow_html=True)
            md_b = generate_markdown_summary(diag, filename).encode("utf-8")
            st.download_button("⬇️ Download MD", data=md_b,
                               file_name=f"fault_report_{slug}.md",
                               mime="text/markdown", use_container_width=True)

        st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)
        with st.expander("📋 Preview Markdown Report"):
            st.markdown(generate_markdown_summary(diag, filename))
