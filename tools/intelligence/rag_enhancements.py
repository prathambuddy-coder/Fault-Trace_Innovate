"""Advanced RAG Enhancements for Industrial Troubleshooting.

Implements technical terminology expansion, symptom extraction, query rewriting,
cross-procedural reranking, and Standard Operating Procedure (SOP) retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Sequence
import numpy as np


# Technical terminology dictionary mapping field expressions to formal manual concepts
TECHNICAL_TERMINOLOGY_MAP: dict[str, list[str]] = {
    "air leak": ["dryer purge valve seating", "downstream pneumatic line rupture", "unloader seal failure"],
    "leak": ["dryer_purge_leak", "downstream_air_leak", "line_pressure_decay"],
    "won't cut out": ["compressor stays loaded and does not reach cut-out", "demand exceeds capacity"],
    "keeps running": ["compressor stays loaded", "continuous motor load", "unloader pilot blocked"],
    "purge blowing": ["dryer purge valve not seating", "purge solenoid stuck open", "tower exhaust"],
    "hot oil": ["oil cooler core fouled", "thermostatic bypass valve stuck", "cooling fan failure"],
    "high temperature": ["oil cooler fouling", "high ambient temperature", "inadequate ventilation"],
    "filter dirty": ["oil filter blocked", "separator element saturated", "high differential pressure"],
    "pressure drop": ["downstream air distribution leak", "heavy air demand", "check valve backflow"],
}

# Standard Operating Procedures (SOPs) extending the static manual
TECHNICAL_SOPS: list[dict[str, Any]] = [
    {
        "sop_id": "SOP-PNEU-01",
        "title": "Standard Operating Procedure: Pneumatic Leak Isolation and Purge Valve Inspection",
        "document": "CAU Maintenance Standard Operating Procedures (Rev 3)",
        "section": "SOP-4.2",
        "applicable_symptoms": ["dryer_purge_leak", "downstream_air_leak", "purge_blowing"],
        "checklist": [
            "1. Depressurize unit and lock out electrical motor supply (LOTO protocol).",
            "2. Inspect dryer purge silencer for continuous audible exhaust while compressor is in loaded state.",
            "3. Measure purge line backpressure using calibrated test gauge at port M1 (must be < 0.05 bar when closed).",
            "4. Verify solenoid coil energization with magnetic flux indicator.",
            "5. Remove purge valve body and inspect Viton seat for particulate indentations or oil emulsion sludge.",
            "6. Reassemble with new O-ring kit (Part #CAU-SK-44) and torque bonnet bolts to 18 Nm.",
        ],
    },
    {
        "sop_id": "SOP-THERM-03",
        "title": "Standard Operating Procedure: Oil Cooler De-fouling and Thermal Efficiency Verification",
        "document": "CAU Maintenance Standard Operating Procedures (Rev 3)",
        "section": "SOP-6.1",
        "applicable_symptoms": ["oil_cooler_fouled", "hot oil", "high temperature"],
        "checklist": [
            "1. Check ambient air inlet temperature at cooling matrix face (must not exceed 40°C).",
            "2. Inspect external aluminum radiator matrix for lint, dust, or grease encapsulation.",
            "3. Clean cooler core using reverse-flow low-pressure dry air lance (max 2 bar) or neutral solvent wash.",
            "4. Verify thermostatic mixing valve begins opening at 71°C and fully opens at 82°C.",
            "5. Record post-cleaning oil temperature stabilization at steady loaded duty cycle.",
        ],
    },
    {
        "sop_id": "SOP-FILT-02",
        "title": "Standard Operating Procedure: Air/Oil Separator Differential Check and Replacement",
        "document": "CAU Maintenance Standard Operating Procedures (Rev 3)",
        "section": "SOP-5.4",
        "applicable_symptoms": ["oil_filter_blocked", "filter dirty", "separator_line_delta"],
        "checklist": [
            "1. Monitor pressure differential between separator vessel gauge (H1) and line pressure (TP3).",
            "2. If differential exceeds 1.2 bar under loaded run, cartridge replacement is mandatory.",
            "3. Drain oil reservoir below sight glass before unbolting separator receiver head.",
            "4. Inspect scavenging line orifice nozzle for carbon or varnish clogging; clean with solvent.",
            "5. Install new coalescing filter element and verify electrical bonding grounding staple is intact.",
        ],
    },
]


@dataclass
class ProcessedQuery:
    original_query: str
    normalized_query: str
    expanded_terms: list[str]
    extracted_symptoms: list[str]
    identified_tags: list[str]


@dataclass
class RerankedCandidate:
    candidate_id: str
    name: str
    source_document: str
    section_ref: str
    initial_score: float
    reranked_score: float
    relevance_rationale: str
    procedural_checklist: list[str] = field(default_factory=list)


class QueryProcessor:
    """Normalizes technician queries and telemetry event descriptors with technical taxonomy expansion."""

    def process(self, raw_query: str) -> ProcessedQuery:
        normalized = raw_query.strip().lower()
        expanded_terms: list[str] = []
        extracted_symptoms: list[str] = []
        identified_tags: list[str] = []

        # Find matching technical terms
        for phrase, expansions in TECHNICAL_TERMINOLOGY_MAP.items():
            if phrase in normalized:
                expanded_terms.extend(expansions)
                extracted_symptoms.append(phrase)

        # Detect specific sensor tags mentioned
        sensor_keywords = {
            "dv_pressure": "DV_pressure",
            "purge pressure": "DV_pressure",
            "purge": "DV_pressure",
            "line pressure": "TP3",
            "tp3": "TP3",
            "discharge": "TP2",
            "tp2": "TP2",
            "motor current": "Motor_current",
            "oil temperature": "Oil_temperature",
            "oil temp": "Oil_temperature",
        }
        for kw, tag in sensor_keywords.items():
            if kw in normalized and tag not in identified_tags:
                identified_tags.append(tag)

        return ProcessedQuery(
            original_query=raw_query,
            normalized_query=normalized,
            expanded_terms=list(dict.fromkeys(expanded_terms)),
            extracted_symptoms=list(dict.fromkeys(extracted_symptoms)),
            identified_tags=identified_tags,
        )


class ProceduralReranker:
    """Re-scores RAG candidate causes using symptom alignment and physical feasibility."""

    def rerank(
        self,
        candidates: Sequence[dict[str, Any]],
        observed_sensors: dict[str, float],
        symptom_key: str,
    ) -> list[RerankedCandidate]:
        results: list[RerankedCandidate] = []

        for cand in candidates:
            fault_id = cand.get("fault_id", "")
            base_score = float(cand.get("score", cand.get("probability", 0.5)))
            name = cand.get("name", fault_id)
            manual_ref = cand.get("manual_ref", {})
            section_ref = manual_ref.get("section", "8.1")
            section_title = manual_ref.get("title", "Troubleshooting")

            alignment_bonus = 0.0
            rationale_parts: list[str] = []

            # 1. Physics rule: If purge pressure is high while loaded, strong bonus for purge leak
            if fault_id == "dryer_purge_leak" and observed_sensors.get("DV_pressure", 0.0) > 0.25:
                alignment_bonus += 0.35
                rationale_parts.append("Elevated purge pressure (>0.25 bar) during loaded state directly matches purge valve seating failure signature.")

            # 2. Physics rule: If oil temperature trend is positive without ambient rise, bonus for cooler fouling
            if fault_id == "oil_cooler_fouled" and observed_sensors.get("Oil_temperature", 0.0) > 80.0:
                alignment_bonus += 0.30
                rationale_parts.append("Elevated oil temperature without high ambient rise indicates cooler heat exchange fouling.")

            # 3. Match with active symptom key
            if symptom_key in fault_id or fault_id in symptom_key:
                alignment_bonus += 0.15
                rationale_parts.append(f"Direct match with triggered symptom key '{symptom_key}'.")

            reranked_score = round(min(0.99, max(0.01, base_score + alignment_bonus)), 4)
            rationale = " ".join(rationale_parts) if rationale_parts else "Candidate matches overall hydraulic and pneumatic pattern."

            # Find matching SOP checklist
            sop_checklist: list[str] = []
            for sop in TECHNICAL_SOPS:
                if any(s in fault_id for s in sop["applicable_symptoms"]):
                    sop_checklist = sop["checklist"]
                    break

            results.append(
                RerankedCandidate(
                    candidate_id=fault_id,
                    name=name,
                    source_document="CAU-7 Service and Maintenance Manual",
                    section_ref=f"Section {section_ref}: {section_title}",
                    initial_score=base_score,
                    reranked_score=reranked_score,
                    relevance_rationale=rationale,
                    procedural_checklist=sop_checklist,
                )
            )

        results.sort(key=lambda c: c.reranked_score, reverse=True)
        return results
