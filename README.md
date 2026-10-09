# TippingPoint ML Engine

Standalone operational bottleneck prediction and discrete-event simulation engine for hospital emergency and inpatient bed management.

---

## 📦 Package Architecture

```
ml_engine/
├── __init__.py      # Package exports and public API
├── dataset.py       # Synthetic operational dataset generator
├── predictor.py     # Scikit-learn Random Forest model & Root Cause Analysis
├── simulator.py     # SimPy discrete-event simulation engine
└── test_ml.py       # CLI test & verification suite
test_ml.py           # Top-level CLI entrypoint
```

---

## 🚀 Key Modules

### 1. `ml_engine.dataset` (`HospitalDataGenerator`)
Simulates realistic hospital operational dynamics including:
- **Temporal & Diurnal Factors**: Hour-of-day peaks, weekend surges.
- **Seasonal Multipliers**: Monsoon disease vector outbreaks (dengue, malaria) and respiratory flu season indices.
- **Operational Metrics**: Hourly arrival inflow, doctor and nurse rosters, nurse-to-patient ratios, total and ICU bed occupancy, pending discharge backlogs, triage wait times, ED boarding counts, and ambulance diversion triggers.
- **Clinical Targets**: Continuous composite risk scores (0-100%), time-to-tipping-point (hours), discrete severity levels (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`), and primary bottleneck labels.

```python
from ml_engine.dataset import generate_dataset

df = generate_dataset(n_samples=2000, seed=42)
print(df.head())
```

---

### 2. `ml_engine.predictor` (`TippingPointPredictor`)
Multi-target predictive and explainability engine:
- **Risk Score Regressor**: Predicts probability / operational risk index (0-100%) using Random Forest.
- **Time-to-Tipping Regressor**: Predicts projected hours remaining before hospital capacity breach.
- **Severity Level Classifier**: Categorizes operational stress (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`).
- **Root Cause Analysis (RCA)**: Evaluates local operational deviations against clinical benchmark baselines, computing percentage impact attribution and prescribing concrete operational counter-measures (e.g. float nurse deployment, discharge lounge activation, overflow ward opening).

```python
from ml_engine.predictor import TippingPointPredictor

predictor = TippingPointPredictor()
predictor.train(df)

result = predictor.predict({
    "inflow_rate_per_hr": 35.0,
    "occupied_beds": 225,
    "bed_occupancy_rate": 0.90,
    "icu_occupied_beds": 30,
    "icu_occupancy_rate": 0.86,
    "doctors_on_duty": 10,
    "nurses_on_duty": 22,
    "nurse_to_patient_ratio": 0.098,
    "pending_discharges": 38,
    "avg_triage_wait_min": 55.0,
    "ed_boarding_count": 18,
    "monsoon_index": 0.85,
    "flu_index": 0.30,
    "hour_of_day": 14,
    "is_weekend": 0,
})

print(f"Risk Score: {result.risk_score}% | Severity: {result.severity_level}")
print(f"Time to Tipping: {result.time_to_tipping_hours}h")
for rc in result.root_causes:
    print(f"[{rc.severity}] {rc.factor} ({rc.contribution_pct}%): {rc.recommendation}")
```

---

### 3. `ml_engine.simulator` (`HospitalSimulator`)
Discrete-event simulation engine built with **SimPy**:
- Simulates patient journeys through Triage $\rightarrow$ Doctor Consultation $\rightarrow$ Bed Assignment / ED Boarding $\rightarrow$ Inpatient Stay $\rightarrow$ Discharge Clearance.
- Evaluates What-If scenarios:
  - Patient arrival surges ($+X\%$ surge).
  - Staffing adjustments (doctor & nurse count changes).
  - Discharge speedup factors (e.g. 50% accelerated discharge via transit lounge).
- Tracks whether and when the hospital tips, peak and average occupancy, and boarding queues.

```python
from ml_engine.simulator import HospitalSimulator, SimulationScenario

baseline = SimulationScenario(name="Baseline", surge_multiplier=1.0)
surge = SimulationScenario(name="Monsoon Surge (+50%)", surge_multiplier=1.50)
mitigated = SimulationScenario(
    name="Surge + Interventions",
    surge_multiplier=1.50,
    doctors_count=12,
    nurses_count=34,
    discharge_speedup_factor=0.45,
)

comparison_df = HospitalSimulator.compare_scenarios([baseline, surge, mitigated])
print(comparison_df)
```

---

## 💻 CLI Verification Suite (`test_ml.py`)

Run tests directly from terminal:

```bash
# Run all tests sequentially (Dataset, Training, Multi-scenario Inference & RCA, Simulation)
python test_ml.py

# Or run individual modules:
python test_ml.py --dataset --samples 1500
python test_ml.py --train
python test_ml.py --predict
python test_ml.py --simulate
```
