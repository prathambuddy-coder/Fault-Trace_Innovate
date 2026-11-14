"""Explainable 10-Point Structured Diagnostic Reporter for Industrial Machinery.

Synthesizes telemetry observations, ML anomaly scores, XGBoost classifications,
manual citations, historical incidents, and step-by-step diagnostic checklists into
an auditable industrial diagnostic report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Sequence

from .anomaly_detector import AnomalyResult
from .fault_classifier import ClassificationResult
from .historical_incidents import HistoricalIncident
from .rag_enhancements import RerankedCandidate


@dataclass
class DiagnosticReport:
    """The 10-Point Industrial Intelligence Diagnostic Report."""

    # 1. Detected abnormal behaviour
    detected_abnormal_behaviour: str

    # 2. Observed sensor evidence
    observed_sensor_evidence: dict[str, Any]

    # 3. Potential fault categories (ranked)
    potential_fault_categories: list[dict[str, Any]]

    # 4. Most likely root cause
    most_likely_root_cause: str

    # 5. Supporting maintenance manual references
    supporting_manual_references: list[dict[str, str]]

    # 6. Similar historical incidents
    similar_historical_incidents: list[dict[str, Any]]

    # 7. Severity assessment
    severity_assessment: dict[str, Any]

    # 8. Confidence and uncertainty breakdown
    confidence_and_uncertainty: dict[str, Any]

    # 9. Recommended diagnostic procedures (step-by-step checklist)
    recommended_diagnostic_procedures: list[str]

    # 10. Human verification requirement
    human_verification_requirement: dict[str, Any]

    # Metadata
    report_id: str = ""
    unit_id: str = "cau-7"
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class DiagnosticReportGenerator:
    """Combines all intelligence layers into the 10-point diagnostic report."""

    def generate(
        self,
        unit_id: str,
        current_sensors: dict[str, float],
        anomaly_res: AnomalyResult,
        classifier_res: ClassificationResult,
        reranked_candidates: Sequence[RerankedCandidate],
        historical_cases: Sequence[HistoricalIncident],
        symptom_key: str = "dryer_purge_leak",
    ) -> DiagnosticReport:
        # 1. Abnormal behavior summary
        top_feature_str = ", ".join(f"{k} (z={v:.1f})" for k, v in anomaly_res.top_contributing_features)
        behaviour_text = (
            f"Machine {unit_id} exhibits anomalous operation ({anomaly_res.message}) "
            f"primarily driven by signal deviations in: {top_feature_str or 'sensor windows'}."
        )

        # 2. Sensor evidence
        sensor_evidence = {
            "current_readings": current_sensors,
            "anomalous_features": [
                {"signal": k, "z_deviation": v}
                for k, v in anomaly_res.top_contributing_features
            ],
            "machine_mode": "loaded" if (current_sensors.get("COMP") == 0 and current_sensors.get("DV_eletric") == 1) else "unloaded/idle",
        }

        # 3. Potential fault categories
        potential_categories = []
        for fault, prob in sorted(classifier_res.probabilities.items(), key=lambda x: x[1], reverse=True)[:4]:
            potential_categories.append({
                "fault_id": fault,
                "probability": prob,
                "is_benign": fault in ("normal_operation", "high_air_demand"),
            })

        # 4. Most likely root cause
        top_candidate = reranked_candidates[0] if reranked_candidates else None
        root_cause_name = top_candidate.name if top_candidate else classifier_res.display_name
        root_cause_text = (
            f"{root_cause_name}. "
            f"Physical mechanism: {top_candidate.relevance_rationale if top_candidate else 'Mechanical deviation from nominal operating baseline.'}"
        )

        # 5. Supporting manual references
        manual_refs = []
        for c in reranked_candidates[:3]:
            manual_refs.append({
                "document": c.source_document,
                "section": c.section_ref,
                "relevance_score": f"{c.reranked_score:.2f}",
                "rationale": c.relevance_rationale,
            })

        # 6. Similar historical incidents
        historical_dicts = []
        for h in historical_cases:
            historical_dicts.append({
                "incident_id": h.incident_id,
                "machine_id": h.machine_id,
                "similarity": f"{h.similarity_score * 100:.1f}%",
                "past_fault": h.identified_fault,
                "past_root_cause": h.root_cause,
                "verified_action": h.maintenance_action,
            })

        # 7. Severity assessment
        sev_level = "high" if "leak" in classifier_res.predicted_fault or "fouled" in classifier_res.predicted_fault else "medium"
        if classifier_res.is_benign:
            sev_level = "low"
        sev_score = 3 if sev_level == "high" else (2 if sev_level == "medium" else 1)
        severity = {
            "level": sev_level,
            "score": sev_score,
            "urgency": "Immediate Inspection Required" if sev_score >= 3 else "Schedule at Next Shift Maintenance",
        }

        # 8. Confidence and uncertainty breakdown
        # Strict distinction between anomaly score, classifier confidence, and RAG score
        top_prob = classifier_res.confidence
        second_prob = sorted(classifier_res.probabilities.values(), reverse=True)[1] if len(classifier_res.probabilities) > 1 else 0.0
        margin = top_prob - second_prob
        is_uncertain = margin < 0.15 or top_prob < 0.65

        confidence_uncertainty = {
            "ml_anomaly_score": round(anomaly_res.anomaly_score, 3),
            "model_classification_confidence": round(classifier_res.confidence, 3),
            "retrieval_relevance_score": round(top_candidate.reranked_score if top_candidate else 0.5, 3),
            "margin_to_second_candidate": round(margin, 3),
            "uncertainty_flag": is_uncertain,
            "uncertainty_rationale": "High confidence separation from alternatives" if not is_uncertain else "Close alternative causes detected; manual inspection required to verify.",
        }

        # 9. Recommended diagnostic checklist
        checklist = top_candidate.procedural_checklist if (top_candidate and top_candidate.procedural_checklist) else [
            "1. Isolate compressor from line pressure manifold.",
            "2. Perform ultrasonic leak scan around pneumatic fittings and purge solenoid.",
            "3. Inspect visual pressure differential indicators across filter assemblies.",
            "4. Verify control valve coil resistances and pilot actuation signals.",
        ]

        # 10. Human verification requirement
        human_verification = {
            "required": True,  # High-integrity industrial systems always require technician confirmation
            "reason": "Safety critical pneumatic system: automated changes to physical valves strictly prohibited.",
            "technician_action": "Review ticket, verify diagnostic checklist in the field, and sign off as 'verified_correct' or 'false_alarm'.",
            "confidence_gate_tier": "ticket" if top_prob >= 0.85 else ("review" if top_prob >= 0.60 else "log_only"),
        }

        return DiagnosticReport(
            report_id=f"REP-{int(datetime.now(timezone.utc).timestamp())}",
            unit_id=unit_id,
            detected_abnormal_behaviour=behaviour_text,
            observed_sensor_evidence=sensor_evidence,
            potential_fault_categories=potential_categories,
            most_likely_root_cause=root_cause_text,
            supporting_manual_references=manual_refs,
            similar_historical_incidents=historical_dicts,
            severity_assessment=severity,
            confidence_and_uncertainty=confidence_uncertainty,
            recommended_diagnostic_procedures=checklist,
            human_verification_requirement=human_verification,
        )

