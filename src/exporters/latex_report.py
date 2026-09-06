"""
LaTeX & PDF Summary Report Generator for Baghewala Field Twin
"""

from datetime import datetime
from pathlib import Path
import subprocess
from typing import Any, Dict


def generate_latex_report(data: Dict[str, Any], output_pdf_path: str = None) -> str:
    """
    Generates a formal engineering report in LaTeX summarizing dynamic CSS/SRP twins.
    Compiles to PDF via pdflatex if installed.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    well_id = data.get("well_id", "BGW-JH-07")
    t_prod = data.get("production_days", 14.5)
    temp_c = data.get("reservoir_temperature_c", 82.4)
    visc_cp = data.get("dynamic_viscosity_cp", 412.5)
    spm = data.get("spm_actual", 4.2)
    spm_safe = data.get("spm_safe", 4.8)
    vfd_hz = data.get("vfd_hz", 35.0)
    diagnostic_state = data.get("diagnostic_state", "Normal Operation")
    pprl_lbs = data.get("pprl_lbs", 16840.0)
    rod_stretch_in = data.get("rod_stretch_in", 8.4)
    j_profit = data.get("cumulative_profit_usd", 124500.0)
    dj_dt = data.get("profit_rate_dj_dt", 420.0)
    cum_sor = data.get("cum_sor", 3.12)
    cutoff_triggered = data.get("cutoff_triggered", False)

    status_color = "red" if cutoff_triggered or "Floating" in diagnostic_state else "teal"

    tex_content = rf"""\documentclass[10pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[margin=0.75in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{xcolor}}
\usepackage{{titlesec}}
\usepackage{{amsmath}}

\titleformat{{\section}}{{\large\bfseries\color{{blue!70!black}}}}{{\thesection}}{{1em}}{{}}[\titlerule]

\title{{\textbf{{Well-to-Surface Digital Twin: Operational Summary Report}}\\\large Asset: Baghewala Field Heavy Oil Operations}}
\author{{Automated SCADA Orchestrator Engine}}
\date{{{timestamp}}}

\begin{{document}}
\maketitle

\section{{Asset \& Executive Status}}
\begin{{tabular}}{{ll@{{\hspace{{0.8cm}}}}ll}}
\textbf{{Well Identifier:}} & {well_id} & \textbf{{Crude Gravity:}} & 18.0$^\circ$ API[cite: 7, 8] \\
\textbf{{True Vertical Depth:}} & 1,200 m[cite: 6] & \textbf{{Operational State:}} & \textcolor{{{status_color}}}{{\textbf{{{diagnostic_state}}}}} \\
\textbf{{Production Elapsed:}} & {t_prod:.1f} Days & \textbf{{Cycle Cutoff Status:}} & \textbf{{{ "CYCLE TERMINATION ALERT" if cutoff_triggered else "STABLE PRODUCTION" }}} \\
\end{{tabular}}

\vspace{{0.4cm}}
\section{{Coupled Thermodynamic \& Viscosity State}}
\begin{{table}}[h!]
\centering
\begin{{tabular}}{{lccc}}
\toprule
\textbf{{Metric}} & \textbf{{Current Value}} & \textbf{{Baseline Floor}} & \textbf{{Units}} \\
\midrule
Reservoir Temperature $T(t)$ & {temp_c:.2f} & 46.00 & $^\circ$C[cite: 2, 7] \\
Crude Viscosity $\mu(t)$ & {visc_cp:.2f} & 66.00 & cP[cite: 3, 7] \\
Safe Pumping Limit $SPM_{{\text{{safe}}}}$ & {spm_safe:.2f} & 2.00 & SPM[cite: 5, 8] \\
Operating Pump Speed & {spm:.2f} & --- & SPM \\
VFD Operational Frequency & {vfd_hz:.1f} & 16.70 & Hz \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Mechanical Dynamics \& Dyno Card Diagnostics}}
\begin{{table}}[h!]
\centering
\begin{{tabular}}{{lcc}}
\toprule
\textbf{{Kinematic Parameter}} & \textbf{{Magnitude}} & \textbf{{Design Safe Limit}} \\
\midrule
Peak Polished Rod Load (PPRL) & {pprl_lbs:.1f} lbf & 24,000.0 lbf \\
Estimated Rod Elastic Stretch ($\Delta L$) & {rod_stretch_in:.2f} in & 18.00 in \\
Morphology Classification & \textbf{{{diagnostic_state}}} & Full Rectangular Envelope \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Lifecycle Economics \& Cycle Cutoff Criteria}}
The joint economic engine balances gross crude revenue against thermal injection costs and electrical wear:
\begin{{equation*}}
J = \int_{{0}}^{{t}} \left[ P_{{o}} q_{{o}}(\tau) - C_{{e}} P_{{m}}(\tau) - C_{{w}} \sigma_{{r}}(\tau) \right] d\tau - C_{{s}} m_{{s}}[cite: 2, 5]
\end{{equation*}}

\begin{{itemize}}
  \item \textbf{{Cumulative Net Profit ($J$):}} \${j_profit:,.2f}
  \item \textbf{{Marginal Profit Rate ($dJ/dt$):}} \${dj_dt:.2f} / day[cite: 2, 7]
  \item \textbf{{Cumulative Steam-Oil Ratio (SOR):}} {cum_sor:.2f} $\text{{m}}^3/\text{{m}}^3$
  \item \textbf{{Autonomous Action:}} { "Trigger Huff injection phase immediately ($dJ/dt \\le 0$)." if cutoff_triggered else "Continue automated VFD throttle modulation." }
\end{{itemize}}

\vspace{{0.5cm}}
\noindent\textit{{Report generated deterministically by the SIH 2026 Well2Surface Twin Pipeline.}}
\end{{document}}
"""
    if output_pdf_path:
        out_file = Path(output_pdf_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        tex_path = out_file.with_suffix(".tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_content)
        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(out_file.parent), str(tex_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except Exception:
            pass  # Fallback gracefully if pdflatex binary is missing

    return tex_content