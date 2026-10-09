"""
simulator.py
============
SimPy discrete-event simulation engine for TippingPoint.
Enables What-If scenario analysis:
- Patient surges (+% arrival surges from seasonal outbreaks or mass casualty events)
- Staffing changes (doctor/nurse rosters, float pools, absenteeism)
- Operational interventions (expedited discharge lounges, bed re-allocation)
- Tracking tipping point breaches (bed saturation, queue explosions, ED boarding)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import simpy


@dataclass
class SimulationScenario:
    """Configuration parameters for a hospital What-If operational scenario."""
    name: str = "Baseline"
    duration_hours: float = 24.0
    total_beds: int = 200
    initial_occupied_beds: int = 150
    base_inflow_per_hr: float = 18.0
    surge_multiplier: float = 1.0  # 1.0 = baseline, 1.30 = +30% surge, 1.50 = +50%
    doctors_count: int = 10
    nurses_count: int = 30
    admission_rate: float = 0.24   # ~24% of emergency arrivals require inpatient beds
    avg_triage_time_min: float = 8.0
    avg_doctor_exam_min: float = 22.0
    avg_inpatient_los_hours: float = 16.0  # Scaled mean stay for observation/acute beds
    avg_discharge_delay_hours: float = 3.5  # Billing/transport delay before bed turnover
    discharge_speedup_factor: float = 1.0   # < 1.0 accelerates discharge (e.g. 0.5 = 50% faster)
    tipping_occupancy_threshold: float = 0.90  # 90% bed occupancy considered tipping point
    tipping_max_boarding_count: int = 12       # >12 boarded patients in ED triggers tipping point
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationResult:
    """Summary and timeseries metrics resulting from a simulation run."""
    scenario_name: str
    duration_hours: float
    total_arrivals: int
    total_admissions: int
    total_discharges: int
    peak_bed_occupancy_pct: float
    avg_bed_occupancy_pct: float
    peak_ed_boarding_queue: int
    peak_ed_wait_min: float
    avg_triage_wait_min: float
    avg_doctor_wait_min: float
    avg_bed_wait_min: float
    tipping_point_reached: bool
    tipping_point_hour: Optional[float]
    tipping_reasons: List[str]
    timeline: pd.DataFrame = field(repr=False)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "duration_hours": self.duration_hours,
            "total_arrivals": self.total_arrivals,
            "total_admissions": self.total_admissions,
            "total_discharges": self.total_discharges,
            "peak_bed_occupancy_pct": round(self.peak_bed_occupancy_pct, 1),
            "avg_bed_occupancy_pct": round(self.avg_bed_occupancy_pct, 1),
            "peak_ed_boarding_queue": self.peak_ed_boarding_queue,
            "peak_ed_wait_min": round(self.peak_ed_wait_min, 1),
            "avg_triage_wait_min": round(self.avg_triage_wait_min, 1),
            "avg_doctor_wait_min": round(self.avg_doctor_wait_min, 1),
            "avg_bed_wait_min": round(self.avg_bed_wait_min, 1),
            "tipping_point_reached": self.tipping_point_reached,
            "tipping_point_hour": round(self.tipping_point_hour, 1) if self.tipping_point_hour is not None else None,
            "tipping_reasons": self.tipping_reasons,
        }


class HospitalSimulator:
    """SimPy-driven discrete-event simulation engine for hospital capacity bottlenecks."""

    def __init__(self, scenario: Optional[SimulationScenario] = None):
        self.scenario = scenario or SimulationScenario()

    def run(self, scenario: Optional[SimulationScenario] = None) -> SimulationResult:
        """Execute the discrete-event simulation for the specified scenario.

        Parameters
        ----------
        scenario : SimulationScenario, optional
            Scenario configuration. Defaults to initial scenario.

        Returns
        -------
        SimulationResult
            Aggregated operational performance and timeline metrics.
        """
        sc = scenario or self.scenario
        rng = np.random.default_rng(sc.random_seed)

        env = simpy.Environment()

        # Shared hospital resources
        # Triage allocated nurses (approx 20% of nurses on triage duty, rest inpatient)
        triage_nurses_count = max(2, int(sc.nurses_count * 0.22))
        triage_res = simpy.Resource(env, capacity=triage_nurses_count)
        doctor_res = simpy.Resource(env, capacity=sc.doctors_count)
        bed_res = simpy.Resource(env, capacity=sc.total_beds)

        # Pre-occupy initial beds
        initial_pre_occupied = min(sc.initial_occupied_beds, sc.total_beds)
        held_bed_requests: List[simpy.Request] = []
        for _ in range(initial_pre_occupied):
            req = bed_res.request()
            held_bed_requests.append(req)

        # Pre-seed initial inpatients discharging over the first simulation window
        def pre_seeded_inpatient_release(env, bed_req, scheduled_discharge_delay_hours):
            yield env.timeout(scheduled_discharge_delay_hours)
            # Discharge administrative delay
            effective_delay = sc.avg_discharge_delay_hours * sc.discharge_speedup_factor
            delay = float(rng.exponential(scale=max(0.2, effective_delay)))
            yield env.timeout(delay)
            bed_res.release(bed_req)
            nonlocal_counts["discharges"] += 1

        for i, req in enumerate(held_bed_requests):
            # Stagger initial discharges across the simulation duration
            sched_hours = float(rng.uniform(0.5, sc.duration_hours * 0.9))
            env.process(pre_seeded_inpatient_release(env, req, sched_hours))

        # Operational metrics tracking
        nonlocal_counts = {
            "arrivals": 0,
            "admissions": 0,
            "discharges": 0,
            "triage_waits": [],
            "doctor_waits": [],
            "bed_waits": [],
            "tipping_point_hour": None,
            "tipping_reasons": set(),
        }

        timeline_records = []

        # Monitor process tracking state periodically (every 0.5 hours)
        def monitor_process(env):
            interval = 0.5  # half-hour intervals
            while True:
                now_hr = env.now
                occupied = bed_res.count
                occupancy_pct = (occupied / sc.total_beds) * 100.0
                boarding_queue = len(bed_res.queue)
                triage_queue = len(triage_res.queue)
                doctor_queue = len(doctor_res.queue)

                is_tipped = False
                reasons = []

                if occupancy_pct >= (sc.tipping_occupancy_threshold * 100.0):
                    is_tipped = True
                    reasons.append(f"Bed Occupancy ({round(occupancy_pct, 1)}% >= {int(sc.tipping_occupancy_threshold*100)}%)")

                if boarding_queue >= sc.tipping_max_boarding_count:
                    is_tipped = True
                    reasons.append(f"ED Boarding Queue ({boarding_queue} pts >= {sc.tipping_max_boarding_count})")

                if triage_queue >= (triage_nurses_count * 4):
                    is_tipped = True
                    reasons.append(f"Triage Queue Gridlock ({triage_queue} waiting)")

                if is_tipped:
                    for r in reasons:
                        nonlocal_counts["tipping_reasons"].add(r)
                    if nonlocal_counts["tipping_point_hour"] is None:
                        nonlocal_counts["tipping_point_hour"] = now_hr

                timeline_records.append({
                    "hour": round(now_hr, 1),
                    "occupied_beds": occupied,
                    "bed_occupancy_pct": round(occupancy_pct, 1),
                    "ed_boarding_queue": boarding_queue,
                    "triage_waiting_count": triage_queue,
                    "doctor_waiting_count": doctor_queue,
                    "is_tipped": is_tipped,
                })

                yield env.timeout(interval)

        env.process(monitor_process(env))

        # Patient Journey Process
        def patient_process(env, patient_id):
            arrival_time = env.now
            nonlocal_counts["arrivals"] += 1

            # 1. Triage Phase
            triage_req_start = env.now
            with triage_res.request() as req:
                yield req
                wait_triage = (env.now - triage_req_start) * 60.0  # in minutes
                nonlocal_counts["triage_waits"].append(wait_triage)
                # Triage duration in hours
                triage_duration_hr = max(2.0, rng.normal(sc.avg_triage_time_min, 2.0)) / 60.0
                yield env.timeout(triage_duration_hr)

            # 2. Doctor Examination Phase
            doctor_req_start = env.now
            with doctor_res.request() as req:
                yield req
                wait_doctor = (env.now - doctor_req_start) * 60.0  # in minutes
                nonlocal_counts["doctor_waits"].append(wait_doctor)
                # Doctor consultation duration in hours
                exam_duration_hr = max(5.0, rng.normal(sc.avg_doctor_exam_min, 6.0)) / 60.0
                yield env.timeout(exam_duration_hr)

            # 3. Admission vs. Outpatient Discharge Decision
            needs_admission = rng.random() < sc.admission_rate

            if needs_admission:
                nonlocal_counts["admissions"] += 1
                bed_req_start = env.now

                # Request Inpatient Hospital Bed
                with bed_res.request() as bed_req:
                    yield bed_req
                    wait_bed_min = (env.now - bed_req_start) * 60.0
                    nonlocal_counts["bed_waits"].append(wait_bed_min)

                    # Inpatient length of stay
                    los_hours = max(2.0, rng.exponential(scale=sc.avg_inpatient_los_hours))
                    yield env.timeout(los_hours)

                    # Administrative / Discharge Delay before freeing bed
                    discharge_delay = max(
                        0.2,
                        rng.exponential(scale=sc.avg_discharge_delay_hours * sc.discharge_speedup_factor)
                    )
                    yield env.timeout(discharge_delay)
                    nonlocal_counts["discharges"] += 1
            else:
                # Treated and discharged from emergency observation
                ed_obs_delay = max(0.5, rng.exponential(scale=1.5))
                yield env.timeout(ed_obs_delay)

        # Inflow Arrival Generator
        def arrival_generator(env):
            patient_counter = 0
            while env.now < sc.duration_hours:
                # Hourly diurnal variation
                hour_of_day = int(env.now % 24)
                diurnal_mult = 0.75 + 0.5 * np.sin(np.pi * (hour_of_day - 6) / 12) ** 2
                current_rate = sc.base_inflow_per_hr * sc.surge_multiplier * diurnal_mult
                current_rate = max(2.0, current_rate)

                # Inter-arrival time (exponential)
                inter_arrival_hr = rng.exponential(1.0 / current_rate)
                yield env.timeout(inter_arrival_hr)

                patient_counter += 1
                env.process(patient_process(env, patient_counter))

        env.process(arrival_generator(env))

        # Execute simulation for specified duration
        env.run(until=sc.duration_hours)

        # Aggregate Results
        df_timeline = pd.DataFrame(timeline_records)

        peak_occ = float(df_timeline["bed_occupancy_pct"].max()) if not df_timeline.empty else 0.0
        avg_occ = float(df_timeline["bed_occupancy_pct"].mean()) if not df_timeline.empty else 0.0
        peak_boarding = int(df_timeline["ed_boarding_queue"].max()) if not df_timeline.empty else 0

        avg_triage_w = float(np.mean(nonlocal_counts["triage_waits"])) if nonlocal_counts["triage_waits"] else 0.0
        avg_doc_w = float(np.mean(nonlocal_counts["doctor_waits"])) if nonlocal_counts["doctor_waits"] else 0.0
        avg_bed_w = float(np.mean(nonlocal_counts["bed_waits"])) if nonlocal_counts["bed_waits"] else 0.0

        all_waits = nonlocal_counts["triage_waits"] + nonlocal_counts["doctor_waits"]
        peak_ed_wait = float(np.percentile(all_waits, 95)) if all_waits else 0.0

        tipping_reached = nonlocal_counts["tipping_point_hour"] is not None
        tipping_reasons = sorted(list(nonlocal_counts["tipping_reasons"]))

        return SimulationResult(
            scenario_name=sc.name,
            duration_hours=sc.duration_hours,
            total_arrivals=nonlocal_counts["arrivals"],
            total_admissions=nonlocal_counts["admissions"],
            total_discharges=nonlocal_counts["discharges"],
            peak_bed_occupancy_pct=peak_occ,
            avg_bed_occupancy_pct=avg_occ,
            peak_ed_boarding_queue=peak_boarding,
            peak_ed_wait_min=peak_ed_wait,
            avg_triage_wait_min=avg_triage_w,
            avg_doctor_wait_min=avg_doc_w,
            avg_bed_wait_min=avg_bed_w,
            tipping_point_reached=tipping_reached,
            tipping_point_hour=nonlocal_counts["tipping_point_hour"],
            tipping_reasons=tipping_reasons,
            timeline=df_timeline,
        )

    @classmethod
    def compare_scenarios(cls, scenarios: List[SimulationScenario]) -> pd.DataFrame:
        """Run multiple What-If scenarios and return a side-by-side comparative table."""
        records = []
        for sc in scenarios:
            sim = cls(scenario=sc)
            result = sim.run()
            summ = result.summary_dict()
            records.append({
                "Scenario": sc.name,
                "Surge Multiplier": f"{sc.surge_multiplier:.2f}x",
                "Staff (Docs/Nurses)": f"{sc.doctors_count}/{sc.nurses_count}",
                "Discharge Delay": f"{sc.avg_discharge_delay_hours * sc.discharge_speedup_factor:.1f}h",
                "Peak Occupancy %": f"{summ['peak_bed_occupancy_pct']}%",
                "Avg Occupancy %": f"{summ['avg_bed_occupancy_pct']}%",
                "Peak Boarding": summ["peak_ed_boarding_queue"],
                "Avg Bed Wait": f"{summ['avg_bed_wait_min']} min",
                "Tipping Point?": "YES (BREACHED)" if summ["tipping_point_reached"] else "NO (SAFE)",
                "Tipping Hour": f"{summ['tipping_point_hour']}h" if summ["tipping_point_hour"] is not None else "N/A",
            })
        return pd.DataFrame(records)


if __name__ == "__main__":
    baseline = SimulationScenario(name="Baseline Operations", surge_multiplier=1.0)
    surge = SimulationScenario(name="Monsoon Surge (+40%)", surge_multiplier=1.4)
    mitigated = SimulationScenario(
        name="Surge + Interventions (+40% surge, +4 nurses, expedited discharge)",
        surge_multiplier=1.4,
        nurses_count=34,
        doctors_count=12,
        discharge_speedup_factor=0.5,
    )

    df_comp = HospitalSimulator.compare_scenarios([baseline, surge, mitigated])
    print("\nScenario Comparison Table:")
    print(df_comp.to_string(index=False))
