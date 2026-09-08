"""
Well2Surface Digital Twin - ISA-101 SCADA Control Cockpit
Asset: Baghewala Field Heavy Oil Operations (18° API)
"""

import json
from pathlib import Path
from typing import Any, Dict
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ---------------------------------------------------------
# COMPATIBILITY & HELPER FUNCTIONS
# ---------------------------------------------------------
def render_chart(fig: go.Figure) -> None:
    """Renders Plotly figures across Streamlit version API variations."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


@st.cache_data(ttl=3600)
def fetch_live_usd_to_inr(fallback_rate: float = 84.0) -> float:
    """
    Fetches real-time USD/INR spot exchange rates from an open API.
    Caches for 1 hour; falls back gracefully to offline baseline if disconnected.
    """
    url = "https://open.er-api.com/v6/latest/USD"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return float(data["rates"]["INR"])
    except Exception:
        return fallback_rate


# ---------------------------------------------------------
# PAGE SETUP & ISA-101 DARK THEME STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Well2Surface SCADA Cockpit | Baghewala",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ISA101_CSS = """
<style>
    .stApp {
        background-color: #121820;
        color: #ECEFF1;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #1A222D;
        border-right: 1px solid #222E40;
    }
    .metric-card {
        background: #222E40;
        border-radius: 6px;
        padding: 14px 18px;
        border: 1px solid #2C3B52;
        margin-bottom: 12px;
    }
    .metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        font-weight: 700;
        color: #00E5FF;
    }
    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #90A4AE;
    }
    .status-badge-healthy {
        background: rgba(0, 230, 118, 0.15);
        color: #00E676;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
        border: 1px solid #00E676;
    }
    .status-badge-alert {
        background: rgba(255, 23, 68, 0.18);
        color: #FF1744;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
        border: 1px solid #FF1744;
    }
    .mode-indicator {
        font-size: 11px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 3px;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 8px;
    }
    .mode-live {
        background: rgba(0, 229, 255, 0.2);
        color: #00E5FF;
        border: 1px solid #00E5FF;
    }
    .mode-sim {
        background: rgba(255, 171, 0, 0.2);
        color: #FFAB00;
        border: 1px solid #FFAB00;
    }
</style>
"""
st.markdown(ISA101_CSS, unsafe_allow_html=True)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
STATE_FILE = ROOT_DIR / "reports" / "latest_state.json"
TELEMETRY_CSV = ROOT_DIR / "reports" / "telemetry_log.csv"
REPORT_PDF = ROOT_DIR / "reports" / "cycle_summary.pdf"


# ---------------------------------------------------------
# PERSISTENCE LOADER & PHYSICS ENGINE
# ---------------------------------------------------------
def load_persisted_state() -> Dict[str, Any]:
    """Reads latest SCADA simulation dump from disk."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "well_id": "BGW-JH-07",
        "timestamp_utc": "Offline Baseline",
        "production_days": 16.5,
        "reservoir_temperature_c": 113.01,
        "dynamic_viscosity_cp": 2.21,
        "spm_actual": 5.0,
        "spm_safe": 8.0,
        "vfd_hz": 30.0,
        "pprl_lbs": 20092.6,
        "rod_stretch_in": 1.88,
        "diagnostic_state": "Normal Operation",
        "profit_rate_dj_dt": 1791.56,
        "cutoff_triggered": False,
        "cum_sor": 3.18,
    }


def generate_dyno_curves(stroke_m: float, rod_stretch_m: float, visc_cp: float, spm: float, is_float: bool):
    """Reconstructs Surface and Downhole Dynamometer Cards."""
    theta = np.linspace(0, 2 * np.pi, 120)
    surf_pos = (stroke_m / 2.0) * (1.0 - np.cos(theta))
    surf_load = 14000.0 + 5000.0 * np.sin(theta) + 1200.0 * np.sin(2 * theta)

    down_stroke_m = max(0.4, stroke_m - rod_stretch_m)
    down_pos = (down_stroke_m / 2.0) * (1.0 - np.cos(theta))

    if is_float:
        down_load = np.where(
            np.sin(theta) < 0,
            -250.0 + 800.0 * np.sin(theta),
            8500.0 + 4000.0 * np.sin(theta),
        )
    else:
        down_load = np.where(
            np.sin(theta) < 0,
            3500.0 + 1000.0 * np.sin(theta),
            11000.0 + 2000.0 * np.sin(theta),
        )
    return surf_pos, surf_load, down_pos, down_load


def simulate_synthetic_physics(t_days: float, spm_target: float, auto_throttle: bool) -> Dict[str, Any]:
    """Synthetic multi-physics simulator for interactive what-if exploration."""
    t_steam = 180.0
    t_res = 46.0
    alpha = 0.045

    # Coupled Exponential Heat Decay & Andrade Dynamic Viscosity
    temp = t_res + (t_steam - t_res) * np.exp(-alpha * t_days)
    a_const, b_const = -14.28, 5820.4
    visc = np.exp(a_const + b_const / (temp + 273.15))

    # Dynamic Safe SPM Floor (Prevents compressive rod float)
    spm_safe = max(2.0, min(8.0, 7.5 * (66.0 / visc) ** 0.35))
    spm_exec = min(spm_target, spm_safe) if auto_throttle else spm_target
    vfd_hz = spm_exec * 6.0

    stroke_length = 2.5
    rod_stretch_m = min(1.2, 0.25 * (visc / 100.0) ** 0.4)
    pprl = 12000.0 + (spm_exec / 6.0) * 4500.0 + (visc * 2.5)

    is_floating = spm_exec > spm_safe and visc > 300.0
    dyno_state = "Rod Floating / Slack" if is_floating else "Normal Operation"

    surf_pos, surf_load, down_pos, down_load = generate_dyno_curves(
        stroke_length, rod_stretch_m, visc, spm_exec, is_floating
    )

    # Lifecycle Economics: dJ/dt = Poil*qo - Celec*Pmotor - Cwear*sigma
    q_oil = max(0.5, 45.0 * np.exp(-0.035 * t_days) * (spm_exec / 5.0))
    p_motor = 22.0 * (spm_exec / 5.0) ** 1.8
    revenue = q_oil * 75.0
    op_cost = p_motor * 0.12 * 24.0 + 150.0
    dj_dt = revenue - op_cost

    cum_oil = max(1.0, 45.0 * ((1 - np.exp(-0.035 * t_days)) / 0.035) * (spm_exec / 5.0))
    cum_sor = 1200.0 / (cum_oil + 1e-4)

    return {
        "well_id": "BGW-JH-07",
        "timestamp_utc": "Simulated Live Buffer",
        "production_days": t_days,
        "reservoir_temperature_c": temp,
        "dynamic_viscosity_cp": visc,
        "spm_safe": spm_safe,
        "spm_actual": spm_exec,
        "vfd_hz": vfd_hz,
        "pprl_lbs": pprl,
        "rod_stretch_in": rod_stretch_m * 39.37,
        "dyno_state": dyno_state,
        "profit_rate_dj_dt": dj_dt,
        "cum_sor": cum_sor,
        "cutoff_triggered": dj_dt <= 0.0,
        "is_floating": is_floating,
        "surf_pos": surf_pos,
        "surf_load": surf_load,
        "down_pos": down_pos,
        "down_load": down_load,
    }


# ---------------------------------------------------------
# SIDEBAR NAVIGATION & TELEMETRY CONTROLS
# ---------------------------------------------------------
st.sidebar.markdown("### 🎛️ Well Supervisory Console")
st.sidebar.markdown("**Asset:** `Baghewala-07` (Jodhpur Sandstone)")

cockpit_screen = st.sidebar.radio(
    "Cockpit Navigation",
    ["Screen 1: Well-to-Surface Cockpit", "Screen 2: EOR Cycle Economics"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Operational Mode")
data_mode = st.sidebar.radio(
    "Telemetry Ingestion Source",
    ["Live SCADA Playback", "Synthetic Physics Simulator"],
    index=0,
)

# Multi-Currency Toggle & Live Auto-Fetching
st.sidebar.markdown("---")
st.sidebar.markdown("#### Financial Valuation Unit")
currency_option = st.sidebar.radio(
    "Reporting Currency",
    ["USD ($)", "INR (₹)"],
    index=0,
    horizontal=True,
)
is_inr = "INR" in currency_option
currency_symbol = "₹" if is_inr else "$"

if is_inr:
    live_fx = fetch_live_usd_to_inr(fallback_rate=84.0)
    fx_rate = st.sidebar.number_input(
        "Exchange Rate (₹ per USD)",
        min_value=50.0,
        max_value=120.0,
        value=round(live_fx, 2),
        step=0.25,
        format="%.2f",
        help="Auto-fetched via live forex feed. Can be adjusted manually.",
    )
else:
    fx_rate = 1.0

# Route Data Based on Active Mode
if data_mode == "Live SCADA Playback":
    st.sidebar.markdown('<span class="mode-indicator mode-live">● Connected to SCADA Engine</span>', unsafe_allow_html=True)
    if st.sidebar.button("🔄 Poll Latest Engine State"):
        st.rerun()

    raw_state = load_persisted_state()
    stroke_m = 2.5
    rod_stretch_m = raw_state.get("rod_stretch_in", 1.88) / 39.37
    is_float = "Floating" in raw_state.get("diagnostic_state", "Normal")

    surf_pos, surf_load, down_pos, down_load = generate_dyno_curves(
        stroke_m,
        rod_stretch_m,
        raw_state.get("dynamic_viscosity_cp", 2.21),
        raw_state.get("spm_actual", 5.0),
        is_float,
    )

    sim_data = {
        "well_id": raw_state.get("well_id", "BGW-JH-07"),
        "timestamp_utc": raw_state.get("timestamp_utc", "Offline"),
        "production_days": raw_state.get("production_days", 16.5),
        "reservoir_temperature_c": raw_state.get("reservoir_temperature_c", 113.01),
        "dynamic_viscosity_cp": raw_state.get("dynamic_viscosity_cp", 2.21),
        "spm_actual": raw_state.get("spm_actual", 5.0),
        "spm_safe": raw_state.get("spm_safe", 8.0),
        "vfd_hz": raw_state.get("vfd_hz", 30.0),
        "pprl_lbs": raw_state.get("pprl_lbs", 20092.6),
        "rod_stretch_in": raw_state.get("rod_stretch_in", 1.88),
        "dyno_state": raw_state.get("diagnostic_state", "Normal Operation"),
        "profit_rate_dj_dt": raw_state.get("profit_rate_dj_dt", 1791.56),
        "cum_sor": raw_state.get("cum_sor", 3.18),
        "cutoff_triggered": raw_state.get("cutoff_triggered", False),
        "is_floating": is_float,
        "surf_pos": surf_pos,
        "surf_load": surf_load,
        "down_pos": down_pos,
        "down_load": down_load,
    }
    st.sidebar.caption(f"State Source: `latest_state.json`\n\nUTC Stamp: `{sim_data['timestamp_utc']}`")
else:
    st.sidebar.markdown('<span class="mode-indicator mode-sim">⚙ Offline Predictive Modeling</span>', unsafe_allow_html=True)
    sim_days = st.sidebar.slider("Puff Production Elapsed (Days)", 0.5, 60.0, 12.0, 0.5)
    sim_spm = st.sidebar.slider("Manual Setpoint (SPM)", 1.0, 8.5, 5.2, 0.1)
    auto_throttle_active = st.sidebar.toggle("VFD Auto-Throttle Engine", value=True)
    sim_data = simulate_synthetic_physics(sim_days, sim_spm, auto_throttle_active)


# ---------------------------------------------------------
# SCREEN 1: WELL-TO-SURFACE COCKPIT
# ---------------------------------------------------------
if cockpit_screen == "Screen 1: Well-to-Surface Cockpit":
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Reservoir Temperature</div>
                <div class="metric-val">{sim_data['reservoir_temperature_c']:.1f} °C</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_kpi2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Crude Viscosity</div>
                <div class="metric-val">{sim_data['dynamic_viscosity_cp']:.1f} cP</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_kpi3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">VFD Target Output</div>
                <div class="metric-val">{sim_data['vfd_hz']:.1f} Hz</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_kpi4:
        status_badge = "status-badge-alert" if sim_data["is_floating"] else "status-badge-healthy"
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Downhole Health State</div>
                <div style="margin-top:6px;"><span class="{status_badge}">{sim_data['dyno_state']}</span></div>
            </div>""",
            unsafe_allow_html=True,
        )

    left_panel, right_panel = st.columns([1, 2])

    # Left: Subsurface & Surface 2D Schematic
    with left_panel:
        st.markdown("**Subsurface Profile (Depth: 1,200 m)**")
        fig_well = go.Figure()

        fig_well.add_trace(go.Scatter(
            x=[-0.6, 0.6, 0.6, -0.6, -0.6],
            y=[0, 0, -1200, -1200, 0],
            fill="toself",
            fillcolor="#1E2836",
            line=dict(color="#455A64", width=2),
            name="Casing Envelope",
            hoverinfo="none",
        ))

        fig_well.add_trace(go.Scatter(
            x=[0, 0],
            y=[0, -1180],
            mode="lines",
            line=dict(color="#00E5FF", width=4),
            name="Sucker Rod String",
        ))

        fig_well.add_trace(go.Scatter(
            x=[-0.5, 0.5, 0.5, -0.5],
            y=[-1100, -1100, -1200, -1200],
            fill="toself",
            fillcolor="rgba(213, 0, 249, 0.25)",
            line=dict(color="#D500F9", dash="dot"),
            name="CSS Drainage Zone",
        ))

        fig_well.update_layout(
            paper_bgcolor="#121820",
            plot_bgcolor="#121820",
            height=680,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(visible=False, range=[-1.2, 1.2]),
            yaxis=dict(title="True Vertical Depth (m)", range=[-1250, 50], gridcolor="#222E40"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#90A4AE", size=10)),
        )
        render_chart(fig_well)

    # Right: Dynamometer Studio, Thermal Decay, & VFD Console
    with right_panel:
        st.markdown("**Dynamometer Studio: Surface Card vs. Downhole Reconstruction**")
        fig_dyno = go.Figure()

        fig_dyno.add_trace(go.Scatter(
            x=sim_data["surf_pos"],
            y=sim_data["surf_load"],
            mode="lines",
            line=dict(color="#00E5FF", width=2.5),
            name="Surface Card (PRL)",
        ))

        downhole_color = "#FF1744" if sim_data["is_floating"] else "#00E676"
        fig_dyno.add_trace(go.Scatter(
            x=sim_data["down_pos"],
            y=sim_data["down_load"],
            mode="lines",
            line=dict(color=downhole_color, width=3),
            name="Downhole Plunger Card",
        ))

        fig_dyno.update_layout(
            paper_bgcolor="#222E40",
            plot_bgcolor="#1A2433",
            height=260,
            margin=dict(l=40, r=20, t=20, b=30),
            xaxis=dict(title="Plunger Stroke Position (m)", gridcolor="#2C3B52", color="#ECEFF1"),
            yaxis=dict(title="Axial Force (lbf)", gridcolor="#2C3B52", color="#ECEFF1"),
            legend=dict(x=0.02, y=0.95, bgcolor="rgba(0,0,0,0.5)", font=dict(color="#ECEFF1")),
        )
        render_chart(fig_dyno)

        st.markdown("**Thermal Decay vs. Andrade Dynamic Viscosity Surge**")
        t_arr = np.linspace(0, 60, 100)
        temp_curve = 46.0 + (180.0 - 46.0) * np.exp(-0.045 * t_arr)
        visc_curve = np.exp(-14.28 + 5820.4 / (temp_curve + 273.15))

        fig_thermal = make_subplots(specs=[[{"secondary_y": True}]])
        fig_thermal.add_trace(
            go.Scatter(x=t_arr, y=temp_curve, name="Temp (°C)", line=dict(color="#D500F9", width=2)),
            secondary_y=False,
        )
        fig_thermal.add_trace(
            go.Scatter(x=t_arr, y=visc_curve, name="Viscosity (cP)", line=dict(color="#FFAB00", width=2, dash="dot")),
            secondary_y=True,
        )
        fig_thermal.add_vline(x=sim_data["production_days"], line=dict(color="#00E5FF", dash="dash"))

        fig_thermal.update_layout(
            paper_bgcolor="#222E40",
            plot_bgcolor="#1A2433",
            height=200,
            margin=dict(l=40, r=40, t=10, b=20),
            xaxis=dict(title="Elapsed Production Time (Days)", gridcolor="#2C3B52", color="#ECEFF1"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.2, font=dict(color="#ECEFF1", size=10)),
        )
        fig_thermal.update_yaxes(title_text="Temp (°C)", gridcolor="#2C3B52", color="#D500F9", secondary_y=False)
        fig_thermal.update_yaxes(title_text="Viscosity (cP)", color="#FFAB00", secondary_y=True)
        render_chart(fig_thermal)

        c_g1, c_g2 = st.columns([1, 1])
        with c_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sim_data["spm_actual"],
                title={"text": f"Operating SPM (Safe Limit: {sim_data['spm_safe']:.1f})", "font": {"size": 13, "color": "#ECEFF1"}},
                gauge={
                    "axis": {"range": [0, 10], "tickcolor": "#ECEFF1"},
                    "bar": {"color": "#00E5FF"},
                    "bgcolor": "#1A2433",
                    "steps": [
                        {"range": [0, 2.0], "color": "#455A64"},
                        {"range": [2.0, sim_data["spm_safe"]], "color": "rgba(0, 230, 118, 0.3)"},
                        {"range": [sim_data["spm_safe"], 10], "color": "rgba(255, 23, 68, 0.4)"},
                    ],
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#222E40",
                height=180,
                margin=dict(l=25, r=25, t=45, b=15),
            )
            render_chart(fig_gauge)

        with c_g2:
            st.markdown(
                f"""<div style="background:#1A2433; padding:16px; border-radius:6px; border:1px solid #2C3B52; height:180px;">
                    <div style="font-size:11px; color:#90A4AE; text-transform:uppercase;">VFD Console Status</div>
                    <div style="font-size:14px; font-weight:600; color:#ECEFF1; margin-top:4px;">
                        Mode: {'<span style="color:#00E5FF;">LIVE SCADA SYNC</span>' if data_mode == 'Live SCADA Playback' else '<span style="color:#FFAB00;">EXPLORATORY SIM</span>'}
                    </div>
                    <div style="font-size:12px; color:#CFD8DC; margin-top:10px; line-height:1.7;">
                        • Peak Load (PPRL): <b>{sim_data['pprl_lbs']:.0f} lbf</b><br>
                        • Elastic Stretch (ΔL): <b>{sim_data['rod_stretch_in']:.2f} in</b><br>
                        • VFD Inverter Target: <b>{sim_data['vfd_hz']:.1f} Hz</b>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------
# SCREEN 2: EOR CYCLE ECONOMICS
# ---------------------------------------------------------
elif cockpit_screen == "Screen 2: EOR Cycle Economics":
    st.markdown(f"### 📈 EOR Lifecycle Economics & Steam Cutoff Predictor ({currency_option.split()[0]})")

    col_e1, col_e2 = st.columns([1, 2])
    with col_e1:
        dj_dt_usd = sim_data["profit_rate_dj_dt"]
        dj_dt_disp = dj_dt_usd * fx_rate
        gauge_color = "#00E676" if dj_dt_usd > 100 else ("#FFAB00" if dj_dt_usd > 0 else "#FF1744")

        min_range = -500.0 * fx_rate
        max_range = 2000.0 * fx_rate
        warn_thresh = 200.0 * fx_rate

        fig_ec_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=dj_dt_disp,
            number={
                "prefix": f"{currency_symbol} ",
                "suffix": "/day",
                "valueformat": ",.0f" if is_inr else ",.2f",
            },
            title={"text": "Marginal Profit Rate (dJ/dt)", "font": {"size": 15, "color": "#ECEFF1"}},
            gauge={
                "axis": {"range": [min_range, max_range], "tickcolor": "#ECEFF1"},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [min_range, 0], "color": "rgba(255, 23, 68, 0.4)"},
                    {"range": [0, warn_thresh], "color": "rgba(255, 171, 0, 0.3)"},
                    {"range": [warn_thresh, max_range], "color": "rgba(0, 230, 118, 0.3)"},
                ],
                "threshold": {"line": {"color": "#FF1744", "width": 4}, "thickness": 0.75, "value": 0},
            },
        ))
        fig_ec_gauge.update_layout(paper_bgcolor="#222E40", height=280, margin=dict(l=20, r=20, t=40, b=20))
        render_chart(fig_ec_gauge)

        if sim_data["cutoff_triggered"]:
            st.error(f"🚨 ECONOMIC CUTOFF TRIGGERED: dJ/dt ≤ 0 {currency_symbol}/day. Terminate Puff production and trigger next Huff steam cycle.")
        elif dj_dt_usd < 200:
            st.warning(f"⚠️ Approaching economic exhaustion limit (< {currency_symbol}{warn_thresh:,.0f}/day). Plan steam boiler schedule.")
        else:
            st.success(f"✅ Profitable EOR operational envelope ({currency_symbol}{dj_dt_disp:,.0f}/day).")

    with col_e2:
        t_seq = np.linspace(1, 60, 60)
        c_oil = 45.0 * ((1 - np.exp(-0.035 * t_seq)) / 0.035) * (sim_data["spm_actual"] / 5.0)
        sor_seq = 1200.0 / (c_oil + 1e-4)

        fig_sor = go.Figure()
        fig_sor.add_trace(go.Scatter(x=t_seq, y=sor_seq, mode="lines", line=dict(color="#00E5FF", width=2.5), name="Cumulative SOR"))
        fig_sor.add_hline(y=4.5, line=dict(color="#FF1744", dash="dash"), annotation_text="Economic Ceiling (SOR = 4.5)")
        fig_sor.add_vline(x=sim_data["production_days"], line=dict(color="#00E676", dash="dot"), annotation_text="Current Day")

        fig_sor.update_layout(
            title="Cumulative Steam-Oil Ratio (SOR) Lifecycle Curve",
            paper_bgcolor="#222E40",
            plot_bgcolor="#1A2433",
            height=280,
            margin=dict(l=40, r=20, t=40, b=30),
            xaxis=dict(title="Production Days", gridcolor="#2C3B52", color="#ECEFF1"),
            yaxis=dict(title="SOR (m³ steam / m³ oil)", gridcolor="#2C3B52", color="#ECEFF1"),
        )
        render_chart(fig_sor)

    # Historical Telemetry Log Table & Export Action Bar
    if TELEMETRY_CSV.exists():
        st.markdown("---")
        st.markdown(f"#### SCADA Telemetry Stream History ({currency_option.split()[0]})")
        try:
            hist_df = pd.read_csv(TELEMETRY_CSV)
            if is_inr and "profit_rate_dj_dt" in hist_df.columns:
                hist_df[f"profit_rate_dj_dt ({currency_symbol})"] = (hist_df["profit_rate_dj_dt"] * fx_rate).round(0)
            st.dataframe(hist_df, width=1200, height=180)

            # Export Action Bar
            col_csv, col_pdf, col_spacer = st.columns([1.5, 2, 3])

            with col_csv:
                csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Telemetry CSV",
                    data=csv_bytes,
                    file_name=f"baghewala_telemetry_{currency_option.split()[0].lower()}.csv",
                    mime="text/csv",
                    help="Download complete SCADA telemetry history in CSV format",
                )

            with col_pdf:
                if REPORT_PDF.exists():
                    with open(REPORT_PDF, "rb") as pdf_f:
                        st.download_button(
                            label="📄 Download Engineering Report (PDF)",
                            data=pdf_f.read(),
                            file_name="Well2Surface_Cycle_Summary.pdf",
                            mime="application/pdf",
                            help="Download compiled MiKTeX engineering summary report",
                        )
                else:
                    st.button(
                        "📄 Engineering PDF (Local TeX Required)",
                        disabled=True,
                        help="Run python main.py locally with pdflatex installed to compile the PDF report",
                    )
        except Exception:
            pass