"""
dataset.py
==========
Synthetic hospital operational data generator for TippingPoint ML Engine.
Simulates real-world operational factors including patient inflow, doctor/nurse
staffing, bed occupancy, pending discharge bottlenecks, and seasonal monsoon/flu spikes.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class HospitalConfig:
    """Configuration baseline for a representative tertiary/district hospital."""
    total_beds: int = 250
    icu_total_beds: int = 35
    baseline_inflow_rate: float = 18.0  # arrivals per hour baseline
    baseline_doctors: int = 14
    baseline_nurses: int = 45
    safe_nurse_patient_ratio: float = 0.20  # 1 nurse per 5 occupied beds
    tipping_occupancy_threshold: float = 0.90  # 90% bed occupancy is clinical tipping point


class HospitalDataGenerator:
    """Generates synthetic multi-factor hospital operational records with realistic

    non-linear interactions between demand, capacity, and seasonal stress.
    """

    def __init__(self, config: Optional[HospitalConfig] = None, seed: Optional[int] = 42):
        self.config = config or HospitalConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self, n_samples: int = 2000) -> pd.DataFrame:
        """Generate a DataFrame of synthetic hospital operational observations.

        Parameters
        ----------
        n_samples : int
            Number of operational snapshots to simulate (e.g. hourly or shift logs).

        Returns
        -------
        pd.DataFrame
            DataFrame containing operational metrics, risk scores, time-to-tipping,
            severity classifications, and primary bottlenecks.
        """
        rng = self.rng
        cfg = self.config

        # 1. Temporal & Diurnal cycle factors
        hour_of_day = rng.integers(0, 24, size=n_samples)
        day_of_week = rng.integers(0, 7, size=n_samples)  # 0=Monday, 6=Sunday
        is_weekend = (day_of_week >= 5).astype(int)

        # Diurnal inflow pattern: peak between 10:00 and 14:00, secondary peak 18:00-21:00, trough 02:00-06:00
        diurnal_factor = 0.7 + 0.45 * np.sin(np.pi * (hour_of_day - 6) / 12) ** 2
        diurnal_factor = np.clip(diurnal_factor, 0.4, 1.45)

        # Weekend effect: slightly lower scheduled admissions but higher trauma/emergency inflow
        weekend_factor = np.where(is_weekend == 1, 1.08, 1.0)

        # 2. Seasonal Factors (Monsoon & Flu)
        # In tropical/subtropical regions, monsoon drives dengue, malaria, gastroenteritis, leptospirosis.
        # Flu seasons drive acute respiratory distress, elderly decompensation, and ICU admissions.
        monsoon_index = rng.beta(a=1.5, b=3.5, size=n_samples)  # Skewed, sporadic severe monsoon storms
        # Inject occasional monsoon outbreak clusters
        outbreak_mask = rng.random(n_samples) < 0.18
        monsoon_index[outbreak_mask] = np.clip(monsoon_index[outbreak_mask] + rng.uniform(0.35, 0.65, size=np.sum(outbreak_mask)), 0.0, 1.0)

        flu_index = rng.beta(a=1.8, b=3.2, size=n_samples)
        flu_spike_mask = rng.random(n_samples) < 0.15
        flu_index[flu_spike_mask] = np.clip(flu_index[flu_spike_mask] + rng.uniform(0.30, 0.55, size=np.sum(flu_spike_mask)), 0.0, 1.0)

        seasonal_inflow_multiplier = 1.0 + (0.50 * monsoon_index) + (0.35 * flu_index)

        # 3. Patient Inflow Rate (Arrivals per hour)
        base_rate = cfg.baseline_inflow_rate * diurnal_factor * weekend_factor * seasonal_inflow_multiplier
        # Add Poisson-like stochastic dispersion
        inflow_rate_per_hr = np.round(rng.normal(loc=base_rate, scale=base_rate * 0.15), 1)
        inflow_rate_per_hr = np.clip(inflow_rate_per_hr, 4.0, 55.0)

        # 4. Staffing Levels (Doctors and Nurses)
        # Shift variations: night shifts (22:00-06:00) have fewer doctors and nurses
        is_night_shift = ((hour_of_day >= 22) | (hour_of_day < 7)).astype(int)
        shift_doctor_mult = np.where(is_night_shift == 1, 0.65, 1.0)
        shift_nurse_mult = np.where(is_night_shift == 1, 0.75, 1.0)

        # Staff absenteeism / sick leaves (amplified during flu season)
        staff_absenteeism_pct = rng.uniform(0.0, 0.25, size=n_samples) + (flu_index * 0.15)
        staff_absenteeism_pct = np.clip(staff_absenteeism_pct, 0.0, 0.45)

        doctors_on_duty = np.round(cfg.baseline_doctors * shift_doctor_mult * (1.0 - staff_absenteeism_pct))
        doctors_on_duty = np.clip(doctors_on_duty, 3, cfg.baseline_doctors + 4).astype(int)

        nurses_on_duty = np.round(cfg.baseline_nurses * shift_nurse_mult * (1.0 - staff_absenteeism_pct))
        nurses_on_duty = np.clip(nurses_on_duty, 12, cfg.baseline_nurses + 8).astype(int)

        # 5. Bed Occupancy & ICU Dynamics
        # Base occupancy influenced by cumulative load, seasonal surges, and pending discharge blockades
        base_occupancy_ratio = rng.uniform(0.68, 0.88, size=n_samples) + (0.08 * monsoon_index) + (0.06 * flu_index)
        base_occupancy_ratio = np.clip(base_occupancy_ratio, 0.50, 0.99)

        total_beds = np.full(n_samples, cfg.total_beds, dtype=int)
        occupied_beds = np.round(total_beds * base_occupancy_ratio).astype(int)
        occupied_beds = np.clip(occupied_beds, int(cfg.total_beds * 0.45), cfg.total_beds)
        bed_occupancy_rate = np.round(occupied_beds / total_beds, 3)

        # ICU Capacity
        icu_total_beds = np.full(n_samples, cfg.icu_total_beds, dtype=int)
        icu_base_ratio = np.clip(base_occupancy_ratio + (0.10 * flu_index) + rng.normal(0, 0.05, n_samples), 0.40, 1.0)
        icu_occupied_beds = np.round(icu_total_beds * icu_base_ratio).astype(int)
        icu_occupied_beds = np.clip(icu_occupied_beds, 10, cfg.icu_total_beds)
        icu_occupancy_rate = np.round(icu_occupied_beds / icu_total_beds, 3)

        # Nurse to Patient Ratio
        nurse_to_patient_ratio = np.round(nurses_on_duty / np.maximum(occupied_beds, 1), 3)

        # 6. Pending Discharges (Administrative, Transport, Billing, Step-down Bottlenecks)
        # Higher occupancy and fewer doctors cause discharge paperwork backlogs
        discharge_pressure = (occupied_beds / cfg.total_beds) * 25.0 + (1.0 - (doctors_on_duty / cfg.baseline_doctors)) * 15.0
        pending_discharges = np.round(rng.normal(loc=discharge_pressure, scale=5.0))
        pending_discharges = np.clip(pending_discharges, 2, 60).astype(int)

        # 7. Operational Wait Times & ED Bottlenecks
        # Triage wait explodes non-linearly when inflow exceeds nurse triage throughput
        triage_capacity_per_hr = nurses_on_duty * 0.85
        load_ratio = inflow_rate_per_hr / np.maximum(triage_capacity_per_hr, 1.0)
        avg_triage_wait_min = np.round(12.0 * np.exp(np.maximum(0, load_ratio - 0.75) * 1.6) + rng.normal(0, 3.0, n_samples), 1)
        avg_triage_wait_min = np.clip(avg_triage_wait_min, 5.0, 150.0)

        # ED Boarding: patients admitted but waiting for an inpatient bed to open
        available_beds = total_beds - occupied_beds
        ed_boarding_count = np.where(
            available_beds < 15,
            np.round((15 - available_beds) * 1.8 + (pending_discharges * 0.3) + rng.normal(0, 2, n_samples)),
            np.round(rng.exponential(scale=2.0, size=n_samples))
        ).astype(int)
        ed_boarding_count = np.clip(ed_boarding_count, 0, 45)

        # Ambulance diversion is triggered when occupancy > 94% or boarding count > 15
        ambulance_diversion_active = (
            (bed_occupancy_rate >= 0.94) | (ed_boarding_count >= 16) | (avg_triage_wait_min >= 90.0)
        ).astype(int)

        # 8. Deterministic Calculation of Tipping Point Risk Score (0 - 100%)
        # Multi-factor operational stress indexing
        # Component weights:
        # - Bed occupancy stress: up to 40 pts
        # - Staffing deficit: up to 25 pts
        # - Discharge bottleneck: up to 15 pts
        # - Inflow surge: up to 12 pts
        # - Seasonal / Outbreak vulnerability: up to 8 pts

        bed_stress = np.clip((bed_occupancy_rate - 0.72) / (0.96 - 0.72), 0.0, 1.0) * 40.0
        
        # Safe ratio is 0.20 (1 nurse per 5 patients). Deficit triggers when below 0.18
        staff_deficit = np.clip((cfg.safe_nurse_patient_ratio - nurse_to_patient_ratio) / (cfg.safe_nurse_patient_ratio - 0.08), 0.0, 1.0) * 25.0

        discharge_stress = np.clip((pending_discharges - 12.0) / 38.0, 0.0, 1.0) * 15.0

        inflow_stress = np.clip((inflow_rate_per_hr - cfg.baseline_inflow_rate) / (40.0 - cfg.baseline_inflow_rate), 0.0, 1.0) * 12.0

        seasonal_stress = (monsoon_index * 0.5 + flu_index * 0.5) * 8.0

        raw_risk = bed_stress + staff_deficit + discharge_stress + inflow_stress + seasonal_stress
        # Add realistic sensor/reporting stochastic jitter
        jitter = rng.normal(0, 1.8, size=n_samples)
        risk_score = np.round(np.clip(raw_risk + jitter, 0.0, 100.0), 1)

        # 9. Time-to-Tipping-Point (in hours)
        # How many hours until the hospital reaches complete bed exhaustion or paralysis?
        # Net bed consumption rate = (Inflow * admission_pct) - (effective discharge rate)
        # Average admission rate: 22% of ED arrivals need inpatient beds
        hourly_admissions = inflow_rate_per_hr * 0.22
        # Discharges happen at roughly 8-14 beds/hour, hindered by pending discharge backlog
        discharge_efficiency = np.clip(1.0 - (pending_discharges / 70.0), 0.25, 1.0)
        hourly_discharges = (occupied_beds * 0.045) * discharge_efficiency
        net_intake_rate = hourly_admissions - hourly_discharges  # beds per hour

        available_capacity = np.maximum(0, total_beds - occupied_beds)

        # Calculate projected time to bed exhaustion or operational saturation
        time_to_tipping = np.where(
            bed_occupancy_rate >= 0.95,
            # Already at or exceeding critical threshold
            np.clip(rng.uniform(0.3, 1.5, size=n_samples), 0.2, 2.5),
            np.where(
                net_intake_rate > 0.5,
                np.clip(available_capacity / net_intake_rate + rng.normal(0, 0.4, size=n_samples), 0.5, 24.0),
                # If stable, tipping point is far away (> 24 hours horizon)
                np.clip(24.0 - (risk_score / 100.0) * 12.0 + rng.normal(0, 1.0, size=n_samples), 8.0, 24.0)
            )
        )
        time_to_tipping_point = np.round(np.clip(time_to_tipping, 0.5, 24.0), 1)

        # 10. Severity Level Categorization
        severity_level = np.select(
            [
                risk_score >= 82.0,
                risk_score >= 62.0,
                risk_score >= 38.0,
            ],
            [
                "CRITICAL",
                "HIGH",
                "MODERATE",
            ],
            default="LOW"
        )

        # 11. Tipping Point Occurred Flag
        tipping_point_occurred = (
            (bed_occupancy_rate >= cfg.tipping_occupancy_threshold) |
            (avg_triage_wait_min >= 75.0) |
            (ed_boarding_count >= 18) |
            (risk_score >= 85.0)
        ).astype(int)

        # 12. Primary Bottleneck Identification (Ground Truth Cause)
        bottleneck_candidates = np.column_stack([
            bed_stress,
            staff_deficit,
            discharge_stress,
            inflow_stress
        ])
        candidate_labels = np.array([
            "BED_CAPACITY_EXHAUSTION",
            "CRITICAL_STAFF_SHORTAGE",
            "DISCHARGE_PROCESS_BACKLOG",
            "INFLOW_SURGE_OVERLOAD"
        ])

        top_candidate_idx = np.argmax(bottleneck_candidates, axis=1)
        primary_bottleneck = np.where(
            risk_score < 38.0,
            "NORMAL_OPERATIONS",
            candidate_labels[top_candidate_idx]
        )

        df = pd.DataFrame({
            # Temporal features
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            # Seasonal drivers
            "monsoon_index": np.round(monsoon_index, 3),
            "flu_index": np.round(flu_index, 3),
            # Operational capacity & demand
            "inflow_rate_per_hr": inflow_rate_per_hr,
            "total_beds": total_beds,
            "occupied_beds": occupied_beds,
            "bed_occupancy_rate": bed_occupancy_rate,
            "icu_total_beds": icu_total_beds,
            "icu_occupied_beds": icu_occupied_beds,
            "icu_occupancy_rate": icu_occupancy_rate,
            "doctors_on_duty": doctors_on_duty,
            "nurses_on_duty": nurses_on_duty,
            "nurse_to_patient_ratio": nurse_to_patient_ratio,
            "pending_discharges": pending_discharges,
            "avg_triage_wait_min": avg_triage_wait_min,
            "ed_boarding_count": ed_boarding_count,
            "ambulance_diversion_active": ambulance_diversion_active,
            # Target labels & Ground Truth
            "risk_score": risk_score,
            "time_to_tipping_point_hours": time_to_tipping_point,
            "severity_level": severity_level,
            "tipping_point_occurred": tipping_point_occurred,
            "primary_bottleneck": primary_bottleneck,
        })

        return df


def generate_dataset(n_samples: int = 2000, seed: int = 42, config: Optional[HospitalConfig] = None) -> pd.DataFrame:
    """Convenience function to generate a synthetic hospital operations dataset."""
    generator = HospitalDataGenerator(config=config, seed=seed)
    return generator.generate(n_samples=n_samples)


if __name__ == "__main__":
    df = generate_dataset(n_samples=10)
    print(f"Generated synthetic dataset with shape: {df.shape}")
    print(df.head(5))
