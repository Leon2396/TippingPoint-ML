"""
predictor.py
============
Scikit-learn based predictive engine for TippingPoint.
Predicts:
- Operational Risk Score (0 - 100%)
- Time-to-Tipping-Point (in hours)
- Severity Level ('LOW', 'MODERATE', 'HIGH', 'CRITICAL')
- Root Cause Analysis (multi-factor clinical & operational attribution with actionable interventions)
"""

from dataclasses import dataclass, asdict
from enum import Enum
import os
from typing import Dict, List, Any, Optional, Union, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RootCauseFactor:
    """Detailed attribution for an operational bottleneck factor."""
    factor: str
    metric_name: str
    current_value: float
    benchmark_value: float
    contribution_pct: float
    severity: str
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionResult:
    """Consolidated prediction output for a hospital operational snapshot."""
    risk_score: float
    severity_level: str
    time_to_tipping_hours: float
    tipping_point_imminent: bool
    primary_bottleneck: str
    root_causes: List[RootCauseFactor]
    feature_contributions: Dict[str, float]
    raw_inputs: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["root_causes"] = [rc.to_dict() for rc in self.root_causes]
        return res


class TippingPointPredictor:
    """Random Forest predictor for hospital tipping points and bottleneck root cause analysis."""

    FEATURE_COLUMNS = [
        "inflow_rate_per_hr",
        "occupied_beds",
        "bed_occupancy_rate",
        "icu_occupied_beds",
        "icu_occupancy_rate",
        "doctors_on_duty",
        "nurses_on_duty",
        "nurse_to_patient_ratio",
        "pending_discharges",
        "avg_triage_wait_min",
        "ed_boarding_count",
        "monsoon_index",
        "flu_index",
        "hour_of_day",
        "is_weekend",
    ]

    BENCHMARKS = {
        "bed_occupancy_rate": {"safe": 0.75, "critical": 0.90, "name": "Bed Occupancy Rate"},
        "nurse_to_patient_ratio": {"safe": 0.20, "critical": 0.12, "name": "Nurse-to-Patient Ratio", "inverted": True},
        "pending_discharges": {"safe": 12.0, "critical": 35.0, "name": "Pending Discharge Backlog"},
        "inflow_rate_per_hr": {"safe": 18.0, "critical": 32.0, "name": "Emergency Inflow Rate"},
        "ed_boarding_count": {"safe": 4.0, "critical": 15.0, "name": "ED Boarding Queue"},
        "avg_triage_wait_min": {"safe": 15.0, "critical": 60.0, "name": "Triage Wait Time"},
        "monsoon_index": {"safe": 0.20, "critical": 0.65, "name": "Monsoon Disease Vector Index"},
        "flu_index": {"safe": 0.20, "critical": 0.60, "name": "Respiratory Flu Index"},
    }

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.risk_model: Optional[RandomForestRegressor] = None
        self.time_model: Optional[RandomForestRegressor] = None
        self.severity_model: Optional[RandomForestClassifier] = None
        self.is_fitted: bool = False
        self.metrics_: Dict[str, Any] = {}

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        n_estimators: int = 120,
        max_depth: int = 14
    ) -> Dict[str, Any]:
        """Train Random Forest models for risk score, time-to-tipping, and severity level.

        Parameters
        ----------
        df : pd.DataFrame
            Operational historical data containing FEATURE_COLUMNS and targets
            ('risk_score', 'time_to_tipping_point_hours', 'severity_level').
        test_size : float
            Fraction of dataset reserved for evaluation.
        n_estimators : int
            Number of decision trees in Random Forest ensembles.
        max_depth : int
            Maximum depth of trees to prevent overfitting.

        Returns
        -------
        Dict[str, Any]
            Training and validation performance metrics.
        """
        missing_features = [col for col in self.FEATURE_COLUMNS if col not in df.columns]
        if missing_features:
            raise ValueError(f"Training DataFrame is missing required features: {missing_features}")

        X = df[self.FEATURE_COLUMNS]
        y_risk = df["risk_score"]
        y_time = df["time_to_tipping_point_hours"]
        y_sev = df["severity_level"]

        X_train, X_test, y_r_train, y_r_test, y_t_train, y_t_test, y_s_train, y_s_test = train_test_split(
            X, y_risk, y_time, y_sev, test_size=test_size, random_state=self.random_state
        )

        # 1. Fit Risk Score Regressor
        self.risk_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=4,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.risk_model.fit(X_train, y_r_train)
        pred_risk = self.risk_model.predict(X_test)
        risk_mae = float(mean_absolute_error(y_r_test, pred_risk))
        risk_r2 = float(r2_score(y_r_test, pred_risk))

        # 2. Fit Time-to-Tipping Regressor
        self.time_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=4,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.time_model.fit(X_train, y_t_train)
        pred_time = self.time_model.predict(X_test)
        time_mae = float(mean_absolute_error(y_t_test, pred_time))
        time_r2 = float(r2_score(y_t_test, pred_time))

        # 3. Fit Severity Classifier
        self.severity_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=4,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.severity_model.fit(X_train, y_s_train)
        pred_sev = self.severity_model.predict(X_test)
        sev_acc = float(accuracy_score(y_s_test, pred_sev))

        # Feature importances from risk model
        importances = dict(
            sorted(
                zip(self.FEATURE_COLUMNS, [float(v) for v in self.risk_model.feature_importances_]),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        self.metrics_ = {
            "risk_score_mae": round(risk_mae, 3),
            "risk_score_r2": round(risk_r2, 4),
            "time_to_tipping_mae_hours": round(time_mae, 3),
            "time_to_tipping_r2": round(time_r2, 4),
            "severity_accuracy": round(sev_acc, 4),
            "feature_importances": importances,
            "test_sample_count": len(X_test),
        }

        self.is_fitted = True
        return self.metrics_

    def predict(
        self,
        features: Union[Dict[str, Any], pd.DataFrame, pd.Series]
    ) -> Union[PredictionResult, List[PredictionResult]]:
        """Predict risk score, time-to-tipping, severity level, and root causes.

        Parameters
        ----------
        features : Dict, Series, or DataFrame
            Input operational metrics matching FEATURE_COLUMNS.

        Returns
        -------
        PredictionResult or List[PredictionResult]
            Structured prediction results.
        """
        if not self.is_fitted:
            raise RuntimeError("Predictor model has not been trained. Call train() or load() first.")

        # Format input to DataFrame
        if isinstance(features, dict):
            df_in = pd.DataFrame([features])
            is_single = True
        elif isinstance(features, pd.Series):
            df_in = pd.DataFrame([features.to_dict()])
            is_single = True
        elif isinstance(features, pd.DataFrame):
            df_in = features.copy()
            is_single = (len(features) == 1)
        else:
            raise TypeError(f"Unsupported features type: {type(features)}")

        # Fill any missing non-critical features if possible
        for col in self.FEATURE_COLUMNS:
            if col not in df_in.columns:
                if col == "monsoon_index":
                    df_in[col] = 0.2
                elif col == "flu_index":
                    df_in[col] = 0.2
                elif col == "hour_of_day":
                    df_in[col] = 12
                elif col == "is_weekend":
                    df_in[col] = 0
                elif col == "icu_occupancy_rate" and "icu_occupied_beds" in df_in:
                    df_in[col] = df_in["icu_occupied_beds"] / 35.0
                elif col == "nurse_to_patient_ratio" and "nurses_on_duty" in df_in and "occupied_beds" in df_in:
                    df_in[col] = df_in["nurses_on_duty"] / np.maximum(df_in["occupied_beds"], 1)
                else:
                    raise ValueError(f"Missing mandatory feature in input: '{col}'")

        X_eval = df_in[self.FEATURE_COLUMNS]

        raw_risk = self.risk_model.predict(X_eval)
        raw_time = self.time_model.predict(X_eval)
        raw_sev = self.severity_model.predict(X_eval)

        results: List[PredictionResult] = []

        for idx in range(len(df_in)):
            row = df_in.iloc[idx]
            risk_val = float(np.clip(round(raw_risk[idx], 1), 0.0, 100.0))
            time_val = float(np.clip(round(raw_time[idx], 1), 0.2, 24.0))
            sev_val = str(raw_sev[idx])

            # Calibrate severity boundary if continuous risk indicates edge case
            if risk_val >= 84.0 and sev_val != "CRITICAL":
                sev_val = "CRITICAL"
            elif risk_val < 38.0 and sev_val != "LOW":
                sev_val = "LOW"

            imminent = bool(time_val <= 4.0 and risk_val >= 68.0)

            # Perform Root Cause Analysis
            root_causes, contributions, primary_cause = self._analyze_root_causes(row, risk_val)

            pred = PredictionResult(
                risk_score=risk_val,
                severity_level=sev_val,
                time_to_tipping_hours=time_val,
                tipping_point_imminent=imminent,
                primary_bottleneck=primary_cause,
                root_causes=root_causes,
                feature_contributions=contributions,
                raw_inputs=row.to_dict(),
            )
            results.append(pred)

        return results[0] if is_single else results

    def _analyze_root_causes(
        self, row: pd.Series, risk_score: float
    ) -> Tuple[List[RootCauseFactor], Dict[str, float], str]:
        """Compute local feature attribution against safe operating baselines."""
        stress_scores: Dict[str, float] = {}

        # 1. Bed occupancy stress
        occ = float(row.get("bed_occupancy_rate", 0.75))
        occ_stress = max(0.0, (occ - 0.75) / (0.95 - 0.75)) * 40.0
        stress_scores["bed_occupancy"] = occ_stress

        # 2. Staffing stress (nurse ratio)
        ratio = float(row.get("nurse_to_patient_ratio", 0.20))
        staff_stress = max(0.0, (0.20 - ratio) / (0.20 - 0.08)) * 25.0
        stress_scores["staff_shortage"] = staff_stress

        # 3. Discharge backlog stress
        discharges = float(row.get("pending_discharges", 10.0))
        disch_stress = max(0.0, (discharges - 12.0) / (38.0 - 12.0)) * 18.0
        stress_scores["discharge_backlog"] = disch_stress

        # 4. Inflow surge stress
        inflow = float(row.get("inflow_rate_per_hr", 18.0))
        inflow_stress = max(0.0, (inflow - 18.0) / (35.0 - 18.0)) * 14.0
        stress_scores["inflow_surge"] = inflow_stress

        # 5. Seasonal epidemic stress
        monsoon = float(row.get("monsoon_index", 0.2))
        flu = float(row.get("flu_index", 0.2))
        seasonal_stress = max(0.0, (monsoon * 0.5 + flu * 0.5 - 0.25) / 0.75) * 12.0
        stress_scores["seasonal_outbreak"] = seasonal_stress

        # 6. ED Boarding bottleneck
        boarding = float(row.get("ed_boarding_count", 2.0))
        board_stress = max(0.0, (boarding - 4.0) / (18.0 - 4.0)) * 12.0
        stress_scores["ed_boarding"] = board_stress

        total_stress = sum(stress_scores.values())
        contributions: Dict[str, float] = {}

        if total_stress > 0.01:
            for k, v in stress_scores.items():
                contributions[k] = round((v / total_stress) * 100.0, 1)
        else:
            contributions = {k: 0.0 for k in stress_scores}

        # Build detailed RootCauseFactor objects for significant contributors
        root_causes: List[RootCauseFactor] = []

        if occ >= 0.85:
            sev = "CRITICAL" if occ >= 0.92 else "HIGH"
            root_causes.append(RootCauseFactor(
                factor="Inpatient Bed Saturation",
                metric_name="bed_occupancy_rate",
                current_value=round(occ * 100, 1),
                benchmark_value=75.0,
                contribution_pct=contributions.get("bed_occupancy", 0.0),
                severity=sev,
                recommendation=(
                    f"Bed capacity at {round(occ * 100, 1)}%. Trigger bed management escalation tier 2: "
                    "authorize step-down transfers, convert 12 day-surgery beds to inpatient overflow."
                ),
            ))

        if ratio < 0.16:
            sev = "CRITICAL" if ratio < 0.12 else "HIGH"
            root_causes.append(RootCauseFactor(
                factor="Nursing Staff Deficit",
                metric_name="nurse_to_patient_ratio",
                current_value=round(ratio, 3),
                benchmark_value=0.20,
                contribution_pct=contributions.get("staff_shortage", 0.0),
                severity=sev,
                recommendation=(
                    f"Nurse-to-patient ratio degraded to 1:{round(1/max(ratio, 0.001), 1)} (standard 1:5). "
                    "Deploy on-call float pool nurses immediately to inpatient units."
                ),
            ))

        if discharges >= 24:
            sev = "CRITICAL" if discharges >= 35 else "HIGH"
            root_causes.append(RootCauseFactor(
                factor="Delayed Discharge Backlog",
                metric_name="pending_discharges",
                current_value=discharges,
                benchmark_value=12.0,
                contribution_pct=contributions.get("discharge_backlog", 0.0),
                severity=sev,
                recommendation=(
                    f"{int(discharges)} patients medically fit awaiting discharge. "
                    "Dispatch mobile pharmacy liaisons and activate rapid transit discharge lounge."
                ),
            ))

        if inflow >= 26:
            sev = "HIGH" if inflow < 36 else "CRITICAL"
            root_causes.append(RootCauseFactor(
                factor="Emergency Inflow Surge",
                metric_name="inflow_rate_per_hr",
                current_value=round(inflow, 1),
                benchmark_value=18.0,
                contribution_pct=contributions.get("inflow_surge", 0.0),
                severity=sev,
                recommendation=(
                    f"Arrival rate surging at {round(inflow, 1)} pts/hr (+{round((inflow-18)/18*100)}% over baseline). "
                    "Open secondary fast-track triage pod to divert minor acuity cases."
                ),
            ))

        if boarding >= 8:
            root_causes.append(RootCauseFactor(
                factor="Emergency Department Boarding Gridlock",
                metric_name="ed_boarding_count",
                current_value=float(boarding),
                benchmark_value=4.0,
                contribution_pct=contributions.get("ed_boarding", 0.0),
                severity="HIGH" if boarding < 16 else "CRITICAL",
                recommendation=(
                    f"{int(boarding)} admitted patients boarded in ED corridors. "
                    "Activate full-capacity protocol and pull boarded patients to inpatient hallway beds."
                ),
            ))

        if monsoon >= 0.55 or flu >= 0.55:
            surge_type = "Monsoon Vector Spikes (Dengue/Malaria)" if monsoon > flu else "Seasonal Flu / Respiratory"
            root_causes.append(RootCauseFactor(
                factor=f"Seasonal Epidemic Surge: {surge_type}",
                metric_name="monsoon_index" if monsoon > flu else "flu_index",
                current_value=round(max(monsoon, flu), 2),
                benchmark_value=0.20,
                contribution_pct=contributions.get("seasonal_outbreak", 0.0),
                severity="HIGH",
                recommendation=(
                    "Implement seasonal clinical pathways; establish rapid blood count testing "
                    "and designated fever observation cohort units."
                ),
            ))

        # Sort root causes by contribution percentage
        root_causes.sort(key=lambda x: x.contribution_pct, reverse=True)

        if not root_causes or risk_score < 38.0:
            primary_cause = "NORMAL_OPERATIONS"
            if not root_causes:
                root_causes.append(RootCauseFactor(
                    factor="Stable Operations",
                    metric_name="nominal_state",
                    current_value=1.0,
                    benchmark_value=1.0,
                    contribution_pct=100.0,
                    severity="LOW",
                    recommendation="Hospital running within nominal safety margins. Continue routine shift monitoring.",
                ))
        else:
            primary_cause = root_causes[0].factor.upper().replace(" ", "_")

        return root_causes, contributions, primary_cause

    def save(self, filepath: str) -> str:
        """Persist trained model weights and metadata to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model. Call train() first.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "risk_model": self.risk_model,
            "time_model": self.time_model,
            "severity_model": self.severity_model,
            "metrics": self.metrics_,
            "feature_columns": self.FEATURE_COLUMNS,
            "random_state": self.random_state,
        }
        joblib.dump(payload, filepath)
        return filepath

    def load(self, filepath: str) -> "TippingPointPredictor":
        """Load trained models and configuration from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")

        payload = joblib.load(filepath)
        self.risk_model = payload["risk_model"]
        self.time_model = payload["time_model"]
        self.severity_model = payload["severity_model"]
        self.metrics_ = payload.get("metrics", {})
        self.FEATURE_COLUMNS = payload.get("feature_columns", self.FEATURE_COLUMNS)
        self.random_state = payload.get("random_state", 42)
        self.is_fitted = True
        return self


if __name__ == "__main__":
    from .dataset import generate_dataset

    print("Generating training data...")
    df_train = generate_dataset(n_samples=500, seed=42)
    predictor = TippingPointPredictor()
    metrics = predictor.train(df_train)
    print("Model Training Metrics:", metrics)

    test_case = {
        "inflow_rate_per_hr": 38.5,
        "occupied_beds": 238,
        "bed_occupancy_rate": 0.952,
        "icu_occupied_beds": 33,
        "icu_occupancy_rate": 0.943,
        "doctors_on_duty": 8,
        "nurses_on_duty": 24,
        "nurse_to_patient_ratio": 0.101,
        "pending_discharges": 42,
        "avg_triage_wait_min": 78.0,
        "ed_boarding_count": 22,
        "monsoon_index": 0.85,
        "flu_index": 0.65,
        "hour_of_day": 14,
        "is_weekend": 1,
    }

    res = predictor.predict(test_case)
    print("\nPrediction Test Case:")
    print(f"Risk Score: {res.risk_score}% | Severity: {res.severity_level} | Time to Tipping: {res.time_to_tipping_hours}h")
    print("Primary Bottleneck:", res.primary_bottleneck)
    for rc in res.root_causes:
        print(f" - [{rc.severity}] {rc.factor} ({rc.contribution_pct}%): {rc.recommendation}")
