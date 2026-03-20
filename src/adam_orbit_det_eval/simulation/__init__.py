"""
simulation — synthetic truth-state validation of LOOO bias detection.

Public API
----------
observatory_map:
    FakeObservatory, ObservatoryMap

bias_models:
    BiasModel (ABC), CompoundBias, ConstantBias, TimingBias, ClockDrift,
    MagnitudeDependentBias, ColorDependentBias, DCRBias, RefractionModelError,
    SeasonalBias, StepChangeBias, NightlyDrift, FieldRotationBias,
    PlateScaleBias, CatalogEpochBias, TrailingBias, ReportingTruncation,
    WrongSiteBias, CTEBias

noise_model:
    NoiseModel

synthetic_obs:
    generate_synthetic_observations

dataset:
    SimulationConfig, SimulationDataset

evaluate:
    evaluate_recovery, StationRecovery
"""

from .bias_models import (
    BiasModel,
    CatalogEpochBias,
    ClockDrift,
    ColorDependentBias,
    CompoundBias,
    ConstantBias,
    CTEBias,
    DCRBias,
    FieldRotationBias,
    MagnitudeDependentBias,
    NightlyDrift,
    PlateScaleBias,
    RefractionModelError,
    ReportingTruncation,
    SeasonalBias,
    StepChangeBias,
    TimingBias,
    TrailingBias,
    WrongSiteBias,
)
from .dataset import SimulationConfig, SimulationDataset
from .evaluate import StationRecovery, evaluate_recovery
from .noise_model import NoiseModel
from .observatory_map import FakeObservatory, ObservatoryMap
from .synthetic_obs import generate_synthetic_observations

__all__ = [
    # observatory_map
    "FakeObservatory",
    "ObservatoryMap",
    # bias_models
    "BiasModel",
    "CatalogEpochBias",
    "ClockDrift",
    "ColorDependentBias",
    "CompoundBias",
    "ConstantBias",
    "CTEBias",
    "DCRBias",
    "FieldRotationBias",
    "MagnitudeDependentBias",
    "NightlyDrift",
    "PlateScaleBias",
    "RefractionModelError",
    "ReportingTruncation",
    "SeasonalBias",
    "StepChangeBias",
    "TimingBias",
    "TrailingBias",
    "WrongSiteBias",
    # noise_model
    "NoiseModel",
    # synthetic_obs
    "generate_synthetic_observations",
    # dataset
    "SimulationConfig",
    "SimulationDataset",
    # evaluate
    "StationRecovery",
    "evaluate_recovery",
]
