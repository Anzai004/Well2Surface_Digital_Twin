```markdown
# Well2Surface Digital Twin

[![CI Unit Tests & Benchmarks](https://img.shields.io/badge/pytest-134%2F134%20passed-00E676?style=flat-square)](https://github.com/Anzai004/Well2Surface_Digital_Twin)
[![Live Streamlit Cockpit](https://img.shields.io/badge/Streamlit%20Cloud-Live%20Cockpit-00E5FF?style=flat-square)](https://well2surface-twin.streamlit.app)
[![Compliance](https://img.shields.io/badge/HMI-ANSI%2FISA--101-blue?style=flat-square)](https://github.com/Anzai004/Well2Surface_Digital_Twin)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.14-FFD43B?style=flat-square)](https://github.com/Anzai004/Well2Surface_Digital_Twin)

A closed-loop cyber-physical digital twin designed for ultra-heavy crude operations in the Baghewala Field (Jodhpur Sandstone formation, Rajasthan). The platform resolves the historical operational disconnect between Cyclic Steam Stimulation (CSS) thermal dissipation and Sucker Rod Pump (SRP) artificial lift by coupling downhole elastic wave mechanics, Andrade viscosity surges, Beggs & Brill multiphase hydraulics and autonomous Variable Frequency Drive (VFD) speed limits.

---

## The Operational Challenge

Heavy oil production at the Baghewala asset presents severe thermodynamic and mechanical lift bottlenecks:
* **Native Reservoir State:** The reservoir sits at a native temperature of 46°C with dynamic crude oil viscosities exceeding 10,000 cP for 17° to 19° API crude with high asphaltene content.
* **The Cooling Phase Dilemma:** After high-pressure steam injection (180°C) shuts in and production begins, the near-wellbore region undergoes steep exponential cooling. As temperature drops back toward baseline, fluid viscosity spikes by several orders of magnitude.
* **The Decoupled Management Conflict:** Pumping units are traditionally operated on fixed time schedules or static stroke rates regardless of current subterranean thermal conditions.
* **Mechanical Failure Cascade:** When crude viscosity surges, annular fluid drag force ($F_{\text{drag}}$) overtakes the submerged buoyant weight of the sucker rod string. Downward rod velocity lags behind the surface beam, creating axial compression and mechanical slack. When the pump reverses on the upstroke, severe impact loading causes rod buckling, tubing wear, stripped barrels and catastrophic parted rod strings.
* **Thermal Energy Inefficiency:** Continuing production past the point of thermal depletion wastes electrical power for negligible fluid lift, causing the Steam-Oil Ratio (SOR) to spike.

```text
Conventional Disconnected Silos:
[CSS Steaming Schedule]  -->  (Unaware of downhole mechanical stress)
[Surface SRP Beam Pump]  -->  (Unaware of exponential viscosity surge)
                                        |
                                        v
                 Rod Floating -> Mechanical Slack -> Parted Rod Strings

Well2Surface Integrated Closed Loop:
[Thermal Decay T(t)] -> [Dynamic Viscosity mu(t)] -> [1D Gibbs Wave PDE Solver]
                                                             |
                                                             v
       [Autonomous VFD Throttle: SPM <= SPM_safe] <--> [Real-time dJ/dt <= 0 Cutoff]

```

---

## Core Computational Modules

### 1. Coupled Subsurface Thermodynamics & Viscosity

Captures the thermal enthalpy delivered during the Huff stage and models the subsequent exponential dissipation during production:

* **Heat Injection ($Q_{\text{inj}}$):** Evaluates sensible and latent steam enthalpy accounting for steam quality $x_{\text{qual}} \in [0, 1]$:
$$Q_{\text{inj}} = m_s \left[ x_{\text{qual}} h_{\text{fg}} + C_w (T_{\text{steam}} - T_{\text{res}}) \right]$$


* **Lumped Thermal Decay:** Solves the cooling ODE to predict temperature $T(t)$ over elapsed production time $t$:
$$T(t) = T_{\text{res}} + (T_{\text{steam}} - T_{\text{res}}) e^{-\alpha t}$$


* **Andrade Dynamic Viscosity:** Directly links cooling to dynamic fluid viscosity $\mu(t)$ in centipoise calibrated to laboratory PVT core data:
$$\mu(t) = \exp\left(A + \frac{B}{T(t) + 273.15}\right) \quad (A = -14.28,\, B = 5820.4)$$



### 2. 1D Gibbs Damped Wave Mechanics

Reconstructs subsurface dynamometer cards at the pump plunger ($x = L$) from surface polished rod load and position vectors:

* **Governing Elastic Wave PDE:**
$$\frac{\partial^2 u(x,t)}{\partial t^2} = a^2 \frac{\partial^2 u(x,t)}{\partial x^2} - c(t) \frac{\partial u(x,t)}{\partial t}$$


where $a = \sqrt{E / \rho_{\text{steel}}} \approx 5{,}000\text{ m/s}$ is acoustic velocity in steel, and $c(t)$ is viscosity-dependent fluid damping:
$$c(t) = \frac{2 \pi \mu(t)}{\rho_{\text{steel}} A_{\text{rod}} \ln(D_{\text{tubing}} / D_{\text{rod}})}$$


* **Discretization:** Discretized using a semi-implicit finite difference mesh with 50 spatial nodes over 1,200 m True Vertical Depth (TVD).
* **Boundary Conditions:** Measured surface displacement $u(0,t) = x_{\text{surface}}(t)$ as a Dirichlet top boundary, paired with a ghost-node Neumann bottom boundary reconstructing downhole plunger load $F_{\text{plunger}} = E A_{\text{rod}} \frac{\partial u}{\partial x}\big\vert{}_{x=L}$.

### 3. Dynamic Non-Floating Speed Limit ($SPM_{\text{safe}}$)

To prevent rod float, downward gravitational force must exceed laminar viscous drag ($W_{\text{rod,buoyant}} > F_{\text{drag}}$):

$$SPM_{\text{safe}}(t) \le K \cdot \frac{(\rho_{\text{steel}} - \rho_{\text{fluid}}) g}{\mu(t)}$$

The engine dynamically caps pumping speed, automatically throttling the surface Variable Frequency Drive as crude cools and thickens.

### 4. Downhole Tensor Diagnostics

Converts continuous downhole force-position vectors ($F_{\text{plunger}}$ vs $u_{\text{plunger}}$) into normalized $224 \times 224$ single-channel computer vision tensors. A heuristic classification tree diagnoses the pump state across five conditions:

* **Normal Operation:** Symmetrical rectangular envelope with proper valve opening and closing.
* **Rod Floating / Slack:** Axial compressive loads ($F_{\text{plunger}} < 0\text{ lbf}$) detected during the downstroke.
* **Fluid Pound:** Sudden steep load drop during downstroke due to incomplete barrel liquid fillage.
* **Gas Locking:** Hyperbolic delay in traveling valve opening caused by entrained gas compression.
* **Pump Unsetting:** Abnormal load hysteresis and shifted mechanical baseline.

### 5. Lifecycle Economics & Cutoff Trigger ($dJ/dt \le 0$)

Maximizes the cumulative net margin $J$ across the entire Huff-Soak-Puff production sequence:

$$J = \int_{0}^{t_{\text{cycle}}} \left[ P_{\text{oil}} q_o(t) - C_{\text{elec}} P_{\text{motor}}(t) - C_{\text{wear}} \sigma_{\text{rod}}(t) \right] dt - C_{\text{steam}} m_s$$

The system monitors the real-time marginal profit rate. The exact day $dJ/dt \le 0$ or the cumulative Steam-Oil Ratio exceeds economic viability, the twin issues an automated cutoff command to cease artificial lift and cycle back to steam injection.

---

## ANSI/ISA-101 SCADA Dashboard

An interactive operations cockpit deployed live on Streamlit Community Cloud:

**Live URL:** [https://well2surface-twin.streamlit.app](https://well2surface-twin.streamlit.app)

* **Design Palette:** Built in an industrial dark theme (`#121820` obsidian background and `#222E40` slate cards) to reduce glare in continuous control room monitoring.
* **Subsurface Visualization:** 2D interactive profile depicting the 1,200 m wellbore, casing envelope, sucker rod string and thermal boundary layer.
* **Dynamometer Studio:** WebGL dual-trace overlay of measured Surface Dyno Cards against reconstructed Downhole Plunger Cards.
* **Thermal & Dynamic Limits:** Synchronized dual-axis tracking of exponential cooling curves against Andrade viscosity surges and live $SPM_{\text{safe}}$ limits.
* **Dual Operation Modes:** Seamless toggling between Live SCADA Playback (ingesting serialized states from `reports/latest_state.json`) and an offline Synthetic Physics Simulator for exploratory what-if analysis.
* **Multi-Currency Localization:** Real-time toggling between USD ($) and INR (₹) with automated live forex fetching from open currency endpoints and offline local fallbacks.

---

## Repository Structure

```text
Well2Surface_Digital_Twin/
├── config/
│   ├── field_params.yaml          # Petrophysical, fluid and wellbore parameters
│   └── operational_limits.yaml    # Rod yield caps, SPM floors and thermal limits
├── data/
│   └── raw/                       # Telemetry drop location (.gitkeep isolated)
├── reports/                       # Simulation dumps, LaTeX PDFs and CSV logs
├── src/
│   ├── dashboard/
│   │   └── app.py                 # ANSI/ISA-101 Streamlit SCADA cockpit
│   ├── diagnostics/
│   │   └── card_classifier.py     # 224x224 dyno rasterizer and 5-state classifier
│   ├── exporters/
│   │   ├── latex_report.py        # Automated MiKTeX engineering PDF compiler
│   │   └── persistence.py         # JSON state encoder and time-series CSV logger
│   ├── optimization/
│   │   ├── economic_objective.py  # Cumulative margin J and dJ/dt cutoff trigger
│   │   └── joint_optimizer.py     # Differential Evolution + SLSQP 4-var optimizer
│   └── physics/
│       ├── css_thermal.py         # Heat injection Q_inj, decay T(t) and viscosity
│       ├── multiphase_flow.py     # Beggs & Brill pressure traverse and holdup
│       └── srp_dynamics.py        # 1D Gibbs wave PDE solver and rod kinematics
├── tests/
│   ├── test_cfl_benchmarks.py     # CFL mesh stability proofs and <200 ms latency tests
│   ├── test_css_thermal.py        # Thermal decay asymptotic limit validations
│   ├── test_multiphase_flow.py    # Beggs & Brill flow-regime transition checks
│   ├── test_optimization.py       # Non-convex optimizer convergence tests
│   └── test_srp_dynamics.py       # Elastic stretch, PPRL and drag unit tests
├── main.py                        # Complete pipeline execution orchestrator
├── requirements.txt               # Pinned Python package dependencies
├── LICENSE                        # Apache License 2.0
└── README.md

```

---

## Quickstart & Installation

### Prerequisites

* Python 3.10, 3.11, 3.12 or 3.14
* Git
* Optional for automated PDF compilation: A local LaTeX engine (such as MiKTeX or TeX Live) providing `pdflatex` in your system path

### Setup Instructions

```bash
# 1. Clone the repository
git clone [https://github.com/Anzai004/Well2Surface_Digital_Twin.git](https://github.com/Anzai004/Well2Surface_Digital_Twin.git)
cd Well2Surface_Digital_Twin

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install required package dependencies
pip install -r requirements.txt

```

### Running the End-to-End Orchestrator

Execute the complete six-stage cyber-physical pipeline:

```bash
python main.py

```

This runs synthetic ingestion, solves thermodynamic decay, discretizes the 1D Gibbs wave mechanics PDE, rasterizes dyno tensors, computes Beggs & Brill multiphase lift, evaluates economic cutoff conditions and exports state files to `reports/`.

### Launching the SCADA Cockpit Locally

```bash
python -m streamlit run src/dashboard/app.py

```

Open your local browser to `http://localhost:8501` to interact with the wellbore schematic, dynamic dyno card loops and VFD controllers.

---

## Verification & Test Suite

The repository contains an automated validation suite covering CFL mesh stability, wave solver execution latency, high-damping numerical bounds, edge conditions and unit physics across all modules:

```bash
python -m pytest tests/ -v

```

```text
tests/test_cfl_benchmarks.py ..............................   [ 22%]
tests/test_css_thermal.py ................                    [ 34%]
tests/test_multiphase_flow.py ................................[ 61%]
tests/test_optimization.py ...................................[ 89%]
tests/test_srp_dynamics.py ..............                     [100%]
============================== 134 passed in 15.63s ==============================

```

### Numerical Rigor Highlights

* **Courant-Friedrichs-Lewy (CFL) Condition:** Validates that the numerical wave solver enforces:
$$\Delta t \le \frac{\Delta x}{a}$$


preventing non-physical oscillations and numerical divergence across spatial node counts from 10 to 200 nodes.
* **Latency Budget Compliance:** Asserts that single-stroke 1D Gibbs wave equation solves over 1,200 m rod strings execute in under 200 ms, satisfying real-time SCADA loop requirements.
* **Extreme Damping Bounds:** Verifies numerical stability under near-native cold crude conditions up to $\mu = 15{,}000\text{ cP}$, confirming zero NaN propagation and proper suppression of displacement amplitudes.
* **Boundary Edge Hardening:** Verifies physical behavior at boundary conditions, including zero stroke length ($S = 0$), non-positive fluid viscosities and extreme fluid resistance limits.

---

## Field IIoT Deployment Roadmap

While the digital twin is demonstrated using an interactive software simulator and serialized state replays, the underlying architecture is built for industrial field deployment:

* **Wellhead Edge Layer:** Industrial edge hardware (Siemens MindConnect Nano or Advantech UNO) installed in the motor control cabinet executes the compiled C++ wave solver locally in under 100 ms.
* **Hardware Fail-Safe Interlocks:** If the load cell detects compressive forces ($F_{\text{plunger}} \le 0\text{ lbf}$) or loads exceeding 90% of the rod tensile yield strength, the edge gateway issues a direct hardware override to throttle the VFD to safe idle (2.0 SPM) within 100 ms.
* **Industrial Messaging:** Telemetry transmission runs over MQTT with Sparkplug B schemas secured via TLS 1.3 mutual authentication (mTLS) across Private Industrial 5G and LTE networks.
* **Enterprise Fleet Scalability:** Decoupled YAML configuration schemas enable onboarding over 200 wells across the Baghewala asset, with containerized microservices auto-scaling on Kubernetes.

---

## License & Authorship

Developed for the Smart India Hackathon (SIH) 2026.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

```

```
