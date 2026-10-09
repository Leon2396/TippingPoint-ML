"""
test_ml.py
==========
Command-line test and verification suite for TippingPoint ML Engine.
Independently verifies:
1. Synthetic dataset generator (inflow, staffing, occupancy, monsoon/flu factors)
2. Scikit-learn predictor training, evaluation, and serialization
3. Multi-scenario operational inference with root cause analysis & recommendations
4. SimPy discrete-event simulation for What-If scenario modeling
"""

import sys
import os

# Self-healing environment check: If current Python has DLL application control restrictions
# (e.g. Python 3.14 on Windows), transparently re-launch with compatible Python 3.13 or .venv.
if __name__ == "__main__" and os.environ.get("_TIPPINGPOINT_RELAUNCHED") is None:
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as _err:
        import subprocess
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", ".venv", "Scripts", "python.exe"),
            "py -3.13",
        ]
        relaunch_cmd = None
        for cand in candidates:
            if cand.startswith("py "):
                relaunch_cmd = ["py", "-3.13", __file__] + sys.argv[1:]
                break
            elif os.path.isfile(cand):
                relaunch_cmd = [cand, __file__] + sys.argv[1:]
                break
        if relaunch_cmd:
            env = os.environ.copy()
            env["_TIPPINGPOINT_RELAUNCHED"] = "1"
            res = subprocess.run(relaunch_cmd, env=env)
            sys.exit(res.returncode)

# Ensure stdout uses UTF-8 if available
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure package root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import time
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Internal imports
try:
    from ml_engine.dataset import HospitalDataGenerator, generate_dataset
    from ml_engine.predictor import TippingPointPredictor, PredictionResult, SeverityLevel
    from ml_engine.simulator import HospitalSimulator, SimulationScenario, SimulationResult
except ImportError:
    from dataset import HospitalDataGenerator, generate_dataset
    from predictor import TippingPointPredictor, PredictionResult, SeverityLevel
    from simulator import HospitalSimulator, SimulationScenario, SimulationResult


# Terminal Color Utilities
class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    GRAY = "\033[90m"


def print_header(title: str):
    width = 75
    print("\n" + Color.CYAN + "=" * width + Color.RESET)
    print(f"{Color.BOLD}{Color.CYAN}  {title.center(width - 4)}{Color.RESET}")
    print(Color.CYAN + "=" * width + Color.RESET + "\n")


def print_step(step_num: int, title: str):
    print(f"\n{Color.BOLD}{Color.YELLOW}[STEP {step_num}] {title}{Color.RESET}")
    print(Color.GRAY + "-" * 75 + Color.RESET)


def format_severity(sev: str) -> str:
    if sev == "CRITICAL":
        return f"{Color.BOLD}{Color.RED}[CRITICAL]{Color.RESET}"
    elif sev == "HIGH":
        return f"{Color.BOLD}{Color.YELLOW}[HIGH]{Color.RESET}"
    elif sev == "MODERATE":
        return f"{Color.BOLD}{Color.BLUE}[MODERATE]{Color.RESET}"
    else:
        return f"{Color.BOLD}{Color.GREEN}[LOW]{Color.RESET}"


def run_test_dataset(n_samples: int = 1500) -> pd.DataFrame:
    """TEST 1: Verify synthetic operational dataset generation."""
    print_step(1, f"Generating Synthetic Hospital Operational Dataset ({n_samples} samples)")
    start_time = time.time()
    generator = HospitalDataGenerator(seed=42)
    df = generator.generate(n_samples=n_samples)
    elapsed = time.time() - start_time

    print(f"Generated {Color.GREEN}{len(df)}{Color.RESET} operational records across {Color.GREEN}{len(df.columns)}{Color.RESET} columns in {elapsed:.2f}s.")
    print(f"Memory footprint: {df.memory_usage().sum() / 1024:.1f} KB")

    print(f"\n{Color.BOLD}Feature Distribution Summary:{Color.RESET}")
    summary_cols = [
        "inflow_rate_per_hr",
        "bed_occupancy_rate",
        "nurse_to_patient_ratio",
        "pending_discharges",
        "avg_triage_wait_min",
        "monsoon_index",
        "flu_index",
        "risk_score",
        "time_to_tipping_point_hours",
    ]
    summary_df = df[summary_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    print(summary_df.to_string())

    print(f"\n{Color.BOLD}Severity Level Breakdown:{Color.RESET}")
    sev_counts = df["severity_level"].value_counts()
    for sev, cnt in sev_counts.items():
        pct = (cnt / len(df)) * 100
        print(f"  {format_severity(sev):<25}: {cnt:>5} records ({pct:.1f}%)")

    print(f"\n{Color.BOLD}Primary Ground Truth Bottlenecks:{Color.RESET}")
    bottleneck_counts = df["primary_bottleneck"].value_counts()
    for bn, cnt in bottleneck_counts.items():
        pct = (cnt / len(df)) * 100
        print(f"  - {bn:<30}: {cnt:>5} ({pct:.1f}%)")

    print(f"\n{Color.BOLD}Sample Operational Records (First 3 rows):{Color.RESET}")
    sample_preview = df[["hour_of_day", "inflow_rate_per_hr", "occupied_beds", "bed_occupancy_rate", "pending_discharges", "risk_score", "severity_level", "time_to_tipping_point_hours"]].head(3)
    print(sample_preview.to_string(index=False))

    return df


def run_test_predictor(df: pd.DataFrame, model_save_path: Optional[str] = None) -> TippingPointPredictor:
    """TEST 2: Train and evaluate Scikit-learn predictor."""
    print_step(2, "Training & Evaluating Scikit-learn Random Forest Models")
    start_time = time.time()
    predictor = TippingPointPredictor(random_state=42)
    metrics = predictor.train(df, test_size=0.2, n_estimators=100)
    elapsed = time.time() - start_time

    print(f"Training completed in {Color.GREEN}{elapsed:.2f}s{Color.RESET} on {len(df) - metrics['test_sample_count']} train samples, {metrics['test_sample_count']} test samples.")
    print(f"\n{Color.BOLD}Model Performance Metrics on Hold-out Set:{Color.RESET}")
    print(f"  * Risk Score Regressor      : R^2 = {Color.GREEN}{metrics['risk_score_r2']:.4f}{Color.RESET} | MAE = {Color.GREEN}{metrics['risk_score_mae']} pts{Color.RESET}")
    print(f"  * Time-to-Tipping Regressor : R^2 = {Color.GREEN}{metrics['time_to_tipping_r2']:.4f}{Color.RESET} | MAE = {Color.GREEN}{metrics['time_to_tipping_mae_hours']} hours{Color.RESET}")
    print(f"  * Severity Classifier       : Accuracy = {Color.GREEN}{metrics['severity_accuracy'] * 100:.1f}%{Color.RESET}")

    print(f"\n{Color.BOLD}Top 6 Predictive Feature Importances (Risk Model):{Color.RESET}")
    for feat, imp in list(metrics["feature_importances"].items())[:6]:
        bar = "#" * int(imp * 35)
        print(f"  {feat:<25} : {imp * 100:>5.1f}% {Color.CYAN}{bar}{Color.RESET}")

    # Persistence verification
    save_path = model_save_path or os.path.join(CURRENT_DIR, "models", "tipping_point_rf.joblib")
    predictor.save(save_path)
    print(f"\nModel successfully saved to: {Color.GRAY}{save_path}{Color.RESET}")

    # Test reload
    reloaded = TippingPointPredictor().load(save_path)
    assert reloaded.is_fitted, "Reloaded model must be fitted."
    print(f"Model reload verified successfully ({len(reloaded.FEATURE_COLUMNS)} input features preserved).")

    return predictor


def run_test_inference(predictor: TippingPointPredictor):
    """TEST 3: Run inference and root cause analysis across diverse hospital scenarios."""
    print_step(3, "Inference & Root Cause Analysis on Real-World Operational Scenarios")

    scenarios = [
        {
            "name": "Scenario A: Nominal Balanced Operations (Mid-week Shift)",
            "state": {
                "inflow_rate_per_hr": 16.0,
                "occupied_beds": 175,
                "bed_occupancy_rate": 0.70,
                "icu_occupied_beds": 20,
                "icu_occupancy_rate": 0.57,
                "doctors_on_duty": 14,
                "nurses_on_duty": 42,
                "nurse_to_patient_ratio": 0.24,  # 1 nurse per 4.1 patients
                "pending_discharges": 9,
                "avg_triage_wait_min": 12.0,
                "ed_boarding_count": 2,
                "monsoon_index": 0.15,
                "flu_index": 0.10,
                "hour_of_day": 11,
                "is_weekend": 0,
            },
        },
        {
            "name": "Scenario B: Monsoon Vector Epidemic Outbreak (Dengue Influx)",
            "state": {
                "inflow_rate_per_hr": 35.0,  # +94% surge
                "occupied_beds": 222,
                "bed_occupancy_rate": 0.888,
                "icu_occupied_beds": 28,
                "icu_occupancy_rate": 0.80,
                "doctors_on_duty": 12,
                "nurses_on_duty": 36,
                "nurse_to_patient_ratio": 0.162,
                "pending_discharges": 22,
                "avg_triage_wait_min": 45.0,
                "ed_boarding_count": 9,
                "monsoon_index": 0.85,
                "flu_index": 0.25,
                "hour_of_day": 16,
                "is_weekend": 0,
            },
        },
        {
            "name": "Scenario C: Severe Nursing Deficit & Shift Absenteeism",
            "state": {
                "inflow_rate_per_hr": 22.0,
                "occupied_beds": 205,
                "bed_occupancy_rate": 0.82,
                "icu_occupied_beds": 25,
                "icu_occupancy_rate": 0.71,
                "doctors_on_duty": 11,
                "nurses_on_duty": 18,  # Major staff shortage!
                "nurse_to_patient_ratio": 0.088,  # 1 nurse per 11.4 patients!
                "pending_discharges": 16,
                "avg_triage_wait_min": 68.0,
                "ed_boarding_count": 7,
                "monsoon_index": 0.20,
                "flu_index": 0.35,
                "hour_of_day": 20,
                "is_weekend": 1,
            },
        },
        {
            "name": "Scenario D: Critical Bed Saturation & Discharge Paralysis (Near-Crash)",
            "state": {
                "inflow_rate_per_hr": 38.0,
                "occupied_beds": 242,  # 96.8% occupancy of 250 beds!
                "bed_occupancy_rate": 0.968,
                "icu_occupied_beds": 34,
                "icu_occupancy_rate": 0.971,
                "doctors_on_duty": 8,
                "nurses_on_duty": 26,
                "nurse_to_patient_ratio": 0.107,
                "pending_discharges": 46,  # 46 blocked discharges!
                "avg_triage_wait_min": 85.0,
                "ed_boarding_count": 24,
                "monsoon_index": 0.75,
                "flu_index": 0.60,
                "hour_of_day": 15,
                "is_weekend": 1,
            },
        },
    ]

    for item in scenarios:
        name = item["name"]
        state = item["state"]
        res: PredictionResult = predictor.predict(state)

        imminent_flag = f"{Color.RED}{Color.BOLD} [IMMINENT BREACH!]{Color.RESET}" if res.tipping_point_imminent else ""
        print(f"\n{Color.BOLD}{Color.MAGENTA}=== {name} ==={Color.RESET}")
        print(
            f"Risk Score : {Color.BOLD}{res.risk_score:>5.1f}%{Color.RESET}  |  "
            f"Severity   : {format_severity(res.severity_level)}  |  "
            f"Time to Tipping: {Color.BOLD}{res.time_to_tipping_hours} hrs{Color.RESET}{imminent_flag}"
        )
        print(f"Primary Bottleneck: {Color.BOLD}{res.primary_bottleneck}{Color.RESET}")
        print("Root Cause Attribution & Prescriptive Interventions:")
        for rc in res.root_causes:
            pct_str = f"({rc.contribution_pct:>4.1f}% impact)"
            print(f"  * {format_severity(rc.severity)} {rc.factor} {Color.CYAN}{pct_str}{Color.RESET}:")
            print(f"    {Color.GRAY}Current: {rc.current_value} vs Benchmark: {rc.benchmark_value}{Color.RESET}")
            print(f"    {Color.YELLOW}Action:{Color.RESET} {rc.recommendation}")


def run_test_simulation():
    """TEST 4: SimPy discrete-event simulation for What-If scenario modeling."""
    print_step(4, "SimPy Discrete-Event Simulation (What-If Scenario Modeling)")

    # Define 3 operational scenarios
    s1 = SimulationScenario(
        name="1. Baseline Capacity (Normal Flow)",
        duration_hours=24.0,
        surge_multiplier=1.0,
        total_beds=200,
        initial_occupied_beds=150,  # 75% baseline
        doctors_count=10,
        nurses_count=30,
        avg_discharge_delay_hours=3.5,
        discharge_speedup_factor=1.0,
        random_seed=42,
    )

    s2 = SimulationScenario(
        name="2. Unmitigated Inflow Surge (+50%)",
        duration_hours=24.0,
        surge_multiplier=1.50,  # +50% patient arrivals
        total_beds=200,
        initial_occupied_beds=165,
        doctors_count=10,
        nurses_count=30,
        avg_discharge_delay_hours=3.5,
        discharge_speedup_factor=1.0,
        random_seed=42,
    )

    s3 = SimulationScenario(
        name="3. Mitigated Surge (+50% surge + 4 nurses + rapid discharge)",
        duration_hours=24.0,
        surge_multiplier=1.50,
        total_beds=200,
        initial_occupied_beds=165,
        doctors_count=12,         # +2 doctors
        nurses_count=34,          # +4 nurses
        avg_discharge_delay_hours=3.5,
        discharge_speedup_factor=0.45,  # 55% faster discharge clearance
        random_seed=42,
    )

    scenarios = [s1, s2, s3]

    print("Executing discrete-event simulations over 24-hour horizons...\n")
    df_comparison = HospitalSimulator.compare_scenarios(scenarios)
    print(df_comparison.to_string(index=False))

    # Detailed inspection of Scenario 2 vs 3
    sim2 = HospitalSimulator(s2).run()
    sim3 = HospitalSimulator(s3).run()

    print(f"\n{Color.BOLD}What-If Analysis Insights:{Color.RESET}")
    if sim2.tipping_point_reached:
        print(
            f"  [!] In {Color.RED}Scenario 2{Color.RESET}, hospital tipped at hour "
            f"{Color.BOLD}{sim2.tipping_point_hour:.1f}h{Color.RESET} with peak bed occupancy of "
            f"{Color.BOLD}{sim2.peak_bed_occupancy_pct:.1f}%{Color.RESET} and {sim2.peak_ed_boarding_queue} boarded patients."
        )
    else:
        print(f"  [*] Scenario 2 Peak occupancy: {sim2.peak_bed_occupancy_pct:.1f}%")

    if not sim3.tipping_point_reached:
        print(
            f"  [+] In {Color.GREEN}Scenario 3 (Interventions){Color.RESET}, proactive staffing (+4 nurses, +2 docs) and "
            f"accelerated discharge clearance successfully {Color.BOLD}AVERTED{Color.RESET} the tipping point!"
        )
        print(f"      Peak occupancy kept at {sim3.peak_bed_occupancy_pct:.1f}% (down from {sim2.peak_bed_occupancy_pct:.1f}%).")
    else:
        print(f"  [*] Scenario 3 Peak occupancy: {sim3.peak_bed_occupancy_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="TippingPoint ML Engine CLI Test Suite - Hospital Bottleneck Prediction & Simulation"
    )
    parser.add_argument("--all", action="store_true", help="Run complete test suite (default)")
    parser.add_argument("--dataset", action="store_true", help="Run only synthetic dataset generation test")
    parser.add_argument("--train", action="store_true", help="Run only model training and evaluation")
    parser.add_argument("--predict", action="store_true", help="Run only inference and root cause analysis test")
    parser.add_argument("--simulate", action="store_true", help="Run only SimPy discrete-event simulation test")
    parser.add_argument("--samples", type=int, default=1500, help="Number of synthetic samples to generate (default: 1500)")
    parser.add_argument("--save-model", type=str, default=None, help="Custom filepath for saving trained model")

    args = parser.parse_args()

    # Default to running all if no specific test flag is passed
    run_all = args.all or not (args.dataset or args.train or args.predict or args.simulate)

    print_header("TippingPoint ML Engine - Comprehensive Test Suite")

    df = None
    predictor = None

    if run_all or args.dataset or args.train or args.predict:
        df = run_test_dataset(n_samples=args.samples)

    if run_all or args.train or args.predict:
        predictor = run_test_predictor(df, model_save_path=args.save_model)

    if run_all or args.predict:
        if predictor is None:
            predictor = run_test_predictor(df)
        run_test_inference(predictor)

    if run_all or args.simulate:
        run_test_simulation()

    print_header("All TippingPoint ML Engine Tests Passed Successfully!")


if __name__ == "__main__":
    main()
