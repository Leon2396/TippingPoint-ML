"""
TippingPoint ML Engine
======================
Standalone, isolated operational bottleneck prediction and discrete-event
simulation engine for hospital emergency and inpatient care units.

Modules:
- dataset: Synthetic operational data generator with seasonal and clinical factors.
- predictor: Scikit-learn Random Forest model predicting risk scores, time-to-tipping,
  severity levels, and local root cause explanations.
- simulator: SimPy discrete-event simulation engine for What-If scenario modeling.
"""

from .dataset import HospitalDataGenerator, generate_dataset
from .predictor import TippingPointPredictor, PredictionResult, SeverityLevel
from .simulator import HospitalSimulator, SimulationScenario, SimulationResult

__version__ = "1.0.0"

__all__ = [
    "HospitalDataGenerator",
    "generate_dataset",
    "TippingPointPredictor",
    "PredictionResult",
    "SeverityLevel",
    "HospitalSimulator",
    "SimulationScenario",
    "SimulationResult",
]
