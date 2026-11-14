"""Historical Incident Intelligence Store and Similarity Retrieval.

Stores past maintenance incidents across machine fleets, enabling case-based reasoning,
symptom comparison, and retrieval of verified historical remediation actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np


@dataclass(frozen=True)
class HistoricalIncident:
    incident_id: str
    machine_id: str
    timestamp: str
    observed_symptoms: str
    sensor_conditions: dict[str, float]
    identified_fault: str
    root_cause: str
    maintenance_action: str
    resolution_status: str
    severity: str
    similarity_score: float = 0.0


# Verified historical incidents from previous CAU fleet logs (CAU-1 to CAU-6)
# Kept strictly separate from the current CAU-7 evaluation dataset.
HISTORICAL_INCIDENTS_DB: list[dict[str, Any]] = [
    {
        "incident_id": "INC-2019-084",
        "machine_id": "CAU-3",
        "timestamp": "2019-08-14T04:22:10Z",
        "observed_symptoms": "Compressor stayed loaded continuously (>25 min) without reaching cut-out pressure. Dryer purge pressure elevated at 0.78 bar during loaded phase.",
        "sensor_conditions": {
            "TP2": 9.85,
            "TP3": 6.82,
            "DV_pressure": 0.78,
            "Motor_current": 3.84,
            "COMP": 0.0,
            "DV_eletric": 1.0,
        },
        "identified_fault": "dryer_purge_leak",
        "root_cause": "Desiccant dryer purge exhaust solenoid valve seat contaminated with particulate debris, preventing tight closure during compression cycle.",
        "maintenance_action": "Disassembled purge solenoid valve, cleaned brass seat, replaced internal Viton seal, and verified pressure drop below 0.05 bar.",
        "resolution_status": "verified_resolved",
        "severity": "high",
    },
    {
        "incident_id": "INC-2019-112",
        "machine_id": "CAU-2",
        "timestamp": "2019-10-03T11:45:00Z",
        "observed_symptoms": "Line pressure decay rate exceeded 0.15 bar/min while unloaded. Cut-in cycle triggered every 90 seconds under low plant demand.",
        "sensor_conditions": {
            "TP2": 7.40,
            "TP3": 7.35,
            "DV_pressure": 0.02,
            "Motor_current": 1.15,
            "COMP": 1.0,
            "DV_eletric": 0.0,
        },
        "identified_fault": "downstream_air_leak",
        "root_cause": "Pneumatic quick-connect fitting on auxiliary air receiver manifold cracked due to structural vibration fatigue.",
        "maintenance_action": "Replaced manifold coupling fitting with vibration-damped SS union and performed ultrasonic leak inspection.",
        "resolution_status": "verified_resolved",
        "severity": "medium",
    },
    {
        "incident_id": "INC-2020-019",
        "machine_id": "CAU-5",
        "timestamp": "2020-01-22T14:10:30Z",
        "observed_symptoms": "Oil temperature climbed steadily past 88°C under normal ambient conditions (16°C). Temperature trend was +4.5°C/hr over a 4-hour window.",
        "sensor_conditions": {
            "TP2": 9.70,
            "TP3": 8.90,
            "DV_pressure": 0.01,
            "Motor_current": 3.75,
            "Oil_temperature": 89.4,
            "COMP": 0.0,
            "DV_eletric": 1.0,
        },
        "identified_fault": "oil_cooler_fouled",
        "root_cause": "External radiator fin bank of air-blast oil cooler clogged with industrial dust and lint, reducing thermal dissipation efficiency by ~40%.",
        "maintenance_action": "Cleaned heat exchanger core with compressed nitrogen lance and biodegradable fin wash; verified operating oil temperature dropped to 68°C.",
        "resolution_status": "verified_resolved",
        "severity": "high",
    },
    {
        "incident_id": "INC-2020-031",
        "machine_id": "CAU-1",
        "timestamp": "2020-02-18T08:30:00Z",
        "observed_symptoms": "Differential pressure between separator and main line exceeded 1.8 bar during peak loaded run. Motor current elevated by 12% above nominal.",
        "sensor_conditions": {
            "TP2": 10.45,
            "TP3": 8.60,
            "H1": 10.40,
            "DV_pressure": 0.01,
            "Motor_current": 4.25,
            "COMP": 0.0,
            "DV_eletric": 1.0,
        },
        "identified_fault": "oil_filter_blocked",
        "root_cause": "Oil separator cartridge saturated with oxidized varnish and particulate deposits beyond 2,000-hour service interval.",
        "maintenance_action": "Replaced air/oil separator element and spin-on oil filter; refilled with fresh synthetic ISO VG 46 compressor lubricant.",
        "resolution_status": "verified_resolved",
        "severity": "medium",
    },
    {
        "incident_id": "INC-2020-044",
        "machine_id": "CAU-4",
        "timestamp": "2020-03-05T16:00:00Z",
        "observed_symptoms": "Rapid pressure drop and continuous loaded run observed during factory shift changeover. Both line and reservoir pressures stabilized at 7.2 bar.",
        "sensor_conditions": {
            "TP2": 9.20,
            "TP3": 7.20,
            "DV_pressure": 0.02,
            "Motor_current": 3.90,
            "COMP": 0.0,
            "DV_eletric": 1.0,
        },
        "identified_fault": "high_air_demand",
        "root_cause": "Three pneumatic blast cabinets operated simultaneously on Section 4 assembly line, temporarily exceeding compressor volumetric capacity.",
        "maintenance_action": "No machine repair needed. Staggered blast cabinet schedules and adjusted intermediate receiver regulator setpoint.",
        "resolution_status": "benign_verified",
        "severity": "low",
    },
    {
        "incident_id": "INC-2020-062",
        "machine_id": "CAU-3",
        "timestamp": "2020-04-09T02:15:00Z",
        "observed_symptoms": "Dryer tower transfer valve failed to switch over at 120s interval. Purge pressure spiked to 2.1 bar, triggering safety release whistle.",
        "sensor_conditions": {
            "TP2": 9.60,
            "TP3": 8.10,
            "DV_pressure": 2.10,
            "Motor_current": 3.70,
            "COMP": 0.0,
            "DV_eletric": 1.0,
        },
        "identified_fault": "dryer_tower_switching_failure",
        "root_cause": "Pneumatic pilot actuator on shuttle valve Tower 2 seized due to moisture ingress in the pilot line.",
        "maintenance_action": "Rebuilt shuttle valve actuator, replaced desiccant bed in Tower 2 pre-filter, and calibrated switching timer.",
        "resolution_status": "verified_resolved",
        "severity": "high",
    },
]


class HistoricalIncidentIntelligence:
    """Retrieves and matches historical maintenance cases based on symptom and sensor condition similarity."""

    def __init__(self, incidents: list[dict[str, Any]] | None = None) -> None:
        self.incidents = incidents or HISTORICAL_INCIDENTS_DB

    def search_similar(
        self,
        current_sensors: dict[str, float],
        symptom_keywords: Sequence[str] = (),
        top_k: int = 2,
    ) -> list[HistoricalIncident]:
        """Calculates distance between current sensor conditions and historical incidents."""
        scored_incidents: list[tuple[float, dict[str, Any]]] = []

        for inc in self.incidents:
            conds = inc["sensor_conditions"]
            # Distance across key matching signals
            dist_sq = 0.0
            weight_sum = 0.0

            # DV_pressure (purge) - heavily weighted for leak discrimination
            if "DV_pressure" in current_sensors and "DV_pressure" in conds:
                diff = abs(current_sensors["DV_pressure"] - conds["DV_pressure"])
                dist_sq += 3.0 * (diff ** 2)
                weight_sum += 3.0

            # TP3 (line pressure)
            if "TP3" in current_sensors and "TP3" in conds:
                diff = abs(current_sensors["TP3"] - conds["TP3"])
                dist_sq += 1.0 * (diff ** 2)
                weight_sum += 1.0

            # Oil_temperature
            if "Oil_temperature" in current_sensors and "Oil_temperature" in conds:
                diff = abs(current_sensors["Oil_temperature"] - conds["Oil_temperature"]) / 10.0
                dist_sq += 1.5 * (diff ** 2)
                weight_sum += 1.5

            # Machine mode agreement
            curr_loaded = current_sensors.get("COMP", 1.0) == 0.0 and current_sensors.get("DV_eletric", 0.0) == 1.0
            inc_loaded = conds.get("COMP", 1.0) == 0.0 and conds.get("DV_eletric", 0.0) == 1.0
            mode_penalty = 0.0 if curr_loaded == inc_loaded else 2.0

            # Keyword bonus
            kw_bonus = 0.0
            for kw in symptom_keywords:
                if kw.lower() in inc["observed_symptoms"].lower():
                    kw_bonus += 0.5

            norm_dist = np.sqrt(dist_sq / max(weight_sum, 1.0)) + mode_penalty
            # Convert distance to similarity score in [0.0, 1.0]
            similarity = float(np.exp(-0.75 * norm_dist) + kw_bonus)
            similarity = min(0.99, max(0.10, similarity))

            scored_incidents.append((similarity, inc))

        # Sort descending by similarity
        scored_incidents.sort(key=lambda x: x[0], reverse=True)

        results: list[HistoricalIncident] = []
        for sim, inc in scored_incidents[:top_k]:
            results.append(
                HistoricalIncident(
                    incident_id=inc["incident_id"],
                    machine_id=inc["machine_id"],
                    timestamp=inc["timestamp"],
                    observed_symptoms=inc["observed_symptoms"],
                    sensor_conditions=inc["sensor_conditions"],
                    identified_fault=inc["identified_fault"],
                    root_cause=inc["root_cause"],
                    maintenance_action=inc["maintenance_action"],
                    resolution_status=inc["resolution_status"],
                    severity=inc["severity"],
                    similarity_score=round(sim, 3),
                )
            )

        return results

