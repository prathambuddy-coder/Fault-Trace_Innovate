"""Feature Extraction and Signal Processing for MetroPT-3 Telemetry.

Computes physical differentials, operational modes, sliding statistical windows,
and domain-specific engineered indicators for industrial compressor fault detection.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Sequence
import numpy as np


FEATURE_NAMES = [
    # Primary sensors
    "TP2",                  # Discharge pressure (bar)
    "TP3",                  # Line pressure (bar)
    "H1",                   # Separator discharge pressure (bar)
    "DV_pressure",          # Dryer purge pressure (bar)
    "Reservoirs",           # Reservoir pressure (bar)
    "Oil_temperature",      # Oil temperature (°C)
    "Motor_current",        # Motor current (A)
    # Digitals
    "COMP",                 # Intake closed (0 = open, 1 = closed)
    "DV_eletric",           # Load valve active (1 = loaded, 0 = unloaded)
    "Towers",               # Dryer tower active (1 or 0)
    "LPS",                  # Low pressure switch active (1 = tripped)
    "Oil_level",            # Oil level normal (1 = ok, 0 = low)
    # Physical Differentials
    "diff_discharge_line",  # TP2 - TP3
    "diff_separator_line",  # H1 - TP3
    "diff_reservoir_line",  # Reservoirs - TP3
    # Machine Mode (One-hot encoded)
    "mode_loaded",          # COMP == 0 and DV_eletric == 1
    "mode_unloaded",        # not loaded and Motor_current >= 1.0
    "mode_off",             # not loaded and Motor_current < 1.0
    # Domain Failure Indicators
    "purge_leak_indicator", # DV_pressure elevated while loaded
    "oil_temp_rise_rate",   # Rate of oil temperature change per hour
    "line_press_slope",     # Rate of line pressure change per minute
    "current_load_ratio",   # Motor current relative to nominal loaded current (~5.5A)
]


@dataclass(frozen=True)
class SensorRow:
    timestamp: str
    tp2: float
    tp3: float
    h1: float
    dv_pressure: float
    reservoirs: float
    oil_temperature: float
    motor_current: float
    comp: float
    dv_eletric: float
    towers: float
    mpg: float
    lps: float
    pressure_switch: float
    oil_level: float
    caudal_impulses: float

    @property
    def is_loaded(self) -> bool:
        return self.comp == 0.0 and self.dv_eletric == 1.0

    @property
    def is_unloaded(self) -> bool:
        return not self.is_loaded and self.motor_current >= 1.0

    @property
    def is_off(self) -> bool:
        return not self.is_loaded and self.motor_current < 1.0


class FeatureExtractor:
    """Extracts machine learning feature vectors from streaming or batched telemetry."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self._history: list[SensorRow] = []

    def push(self, row: SensorRow) -> np.ndarray:
        """Pushes one sample and returns the 1D feature vector."""
        self._history.append(row)
        if len(self._history) > self.window_size * 4:
            self._history.pop(0)
        return self._extract_features(row)

    def _extract_features(self, row: SensorRow) -> np.ndarray:
        # Differentials
        diff_dl = row.tp2 - row.tp3
        diff_sl = row.h1 - row.tp3
        diff_rl = row.reservoirs - row.tp3

        # Modes
        m_loaded = 1.0 if row.is_loaded else 0.0
        m_unloaded = 1.0 if row.is_unloaded else 0.0
        m_off = 1.0 if row.is_off else 0.0

        # Purge indicator: purge pressure elevated during loaded operation
        purge_leak = float(row.dv_pressure > 0.25 and row.is_loaded)

        # Dynamic slopes over history
        oil_slope = 0.0
        line_slope = 0.0
        if len(self._history) >= 2:
            dt_s = max(10.0, 10.0 * (len(self._history) - 1))
            delta_oil = row.oil_temperature - self._history[0].oil_temperature
            delta_p = row.tp3 - self._history[0].tp3
            oil_slope = (delta_oil / dt_s) * 3600.0   # °C per hour
            line_slope = (delta_p / dt_s) * 60.0     # bar per min

        # Current ratio relative to typical ~5.5 A loaded rating
        current_ratio = row.motor_current / 5.5

        features = [
            row.tp2,
            row.tp3,
            row.h1,
            row.dv_pressure,
            row.reservoirs,
            row.oil_temperature,
            row.motor_current,
            row.comp,
            row.dv_eletric,
            row.towers,
            row.lps,
            row.oil_level,
            diff_dl,
            diff_sl,
            diff_rl,
            m_loaded,
            m_unloaded,
            m_off,
            purge_leak,
            oil_slope,
            line_slope,
            current_ratio,
        ]
        return np.array(features, dtype=np.float32)

    @staticmethod
    def parse_csv_row(row_dict: dict[str, str]) -> SensorRow:
        return SensorRow(
            timestamp=row_dict.get("timestamp", ""),
            tp2=float(row_dict.get("TP2", 0.0)),
            tp3=float(row_dict.get("TP3", 0.0)),
            h1=float(row_dict.get("H1", 0.0)),
            dv_pressure=float(row_dict.get("DV_pressure", 0.0)),
            reservoirs=float(row_dict.get("Reservoirs", 0.0)),
            oil_temperature=float(row_dict.get("Oil_temperature", 0.0)),
            motor_current=float(row_dict.get("Motor_current", 0.0)),
            comp=float(row_dict.get("COMP", 1.0)),
            dv_eletric=float(row_dict.get("DV_eletric", 0.0)),
            towers=float(row_dict.get("Towers", 0.0)),
            mpg=float(row_dict.get("MPG", 0.0)),
            lps=float(row_dict.get("LPS", 0.0)),
            pressure_switch=float(row_dict.get("Pressure_switch", 0.0)),
            oil_level=float(row_dict.get("Oil_level", 1.0)),
            caudal_impulses=float(row_dict.get("Caudal_impulses", 1.0)),
        )


def stream_dataset(csv_path: str | Path, max_rows: int | None = None) -> Generator[SensorRow, None, None]:
    """Streams parsed SensorRow objects from the MetroPT-3 CSV."""
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            yield FeatureExtractor.parse_csv_row(row)
            count += 1
            if max_rows is not None and count >= max_rows:
                break

