"""Unit tests for the Industrial Intelligence Platform.

Tests feature extraction, anomaly detection, XGBoost fault classification,
historical incident retrieval, query expansion, procedural reranking,
and 10-point diagnostic report generation.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from tools.intelligence.anomaly_detector import (
    HybridIndustrialAnomalyDetector,
    IsolationForestDetector,
    RobustStatisticalDetector,
)
from tools.intelligence.diagnostic_reporter import (
    DiagnosticReport,
    DiagnosticReportGenerator,
)
from tools.intelligence.fault_classifier import (
    FAULT_CLASSES,
    IndustrialFaultClassifier,
)
from tools.intelligence.feature_extractor import (
    FeatureExtractor,
    SensorRow,
)
from tools.intelligence.historical_incidents import (
    HISTORICAL_INCIDENTS_DB,
    HistoricalIncidentIntelligence,
)
from tools.intelligence.rag_enhancements import (
    ProceduralReranker,
    QueryProcessor,
    TECHNICAL_SOPS,
)


def test_feature_extraction_dimensions():
    extractor = FeatureExtractor(window_size=10)
    row = SensorRow(
        timestamp="2020-02-01T00:00:00Z",
        tp2=9.5,
        tp3=8.0,
        h1=9.4,
        dv_pressure=0.01,
        reservoirs=8.0,
        oil_temperature=65.0,
        motor_current=3.8,
        comp=0.0,
        dv_eletric=1.0,
        towers=1.0,
        mpg=1.0,
        lps=0.0,
        pressure_switch=1.0,
        oil_level=1.0,
        caudal_impulses=1.0,
    )
    features = extractor.push(row)
    assert isinstance(features, np.ndarray)
    assert features.shape == (22,)
    # Verify loaded state: comp == 0 and dv_eletric == 1
    assert row.is_loaded is True
    assert features[15] == 1.0  # mode_loaded
    assert features[16] == 0.0  # mode_unloaded


def test_statistical_anomaly_detector():
    detector = RobustStatisticalDetector(threshold_z=3.0)
    # Generate 100 normal baseline samples
    normal_data = np.random.normal(loc=10.0, scale=1.0, size=(100, 22))
    detector.fit(normal_data)

    # Test normal sample
    normal_sample = np.random.normal(loc=10.0, scale=1.0, size=(22,))
    score, top_f = detector.score(normal_sample)
    assert 0.0 <= score <= 1.0
    assert score < 0.60

    # Test extreme outlier
    anom_sample = normal_sample.copy()
    anom_sample[3] = 100.0  # huge spike
    score_anom, top_f_anom = detector.score(anom_sample)
    assert score_anom > 0.80
    assert len(top_f_anom) > 0


def test_hybrid_anomaly_detector_ensemble():
    detector = HybridIndustrialAnomalyDetector(anomaly_threshold=0.60)
    normal_data = np.random.normal(loc=5.0, scale=0.5, size=(200, 22))
    detector.train_on_baseline(normal_data)

    test_sample = np.random.normal(loc=5.0, scale=0.5, size=(22,))
    res = detector.detect(test_sample)
    assert res.anomaly_score < 0.60
    assert res.is_anomaly is False

    anom_sample = test_sample.copy()
    anom_sample[0] = 50.0
    res_anom = detector.detect(anom_sample)
    assert res_anom.anomaly_score >= 0.60
    assert res_anom.is_anomaly is True


def test_historical_incident_retrieval():
    intel = HistoricalIncidentIntelligence()
    sensors = {"DV_pressure": 0.80, "TP3": 6.8, "COMP": 0.0, "DV_eletric": 1.0}
    matches = intel.search_similar(sensors, symptom_keywords=["purge", "leak"], top_k=2)
    assert len(matches) == 2
    top = matches[0]
    assert top.identified_fault == "dryer_purge_leak"
    assert top.similarity_score > 0.80
    assert "purge" in top.observed_symptoms.lower()


def test_query_expansion():
    proc = QueryProcessor()
    q = proc.process("dryer purge blowing high air leak")
    assert "dryer purge valve seating" in q.expanded_terms
    assert "leak" in q.extracted_symptoms
    assert "DV_pressure" in q.identified_tags


def test_procedural_reranker():
    reranker = ProceduralReranker()
    candidates = [
        {"fault_id": "oil_cooler_fouled", "name": "Oil cooler fouling", "probability": 0.50},
        {"fault_id": "dryer_purge_leak", "name": "Dryer purge leak", "probability": 0.40},
    ]
    # Under high purge pressure, dryer_purge_leak should be reranked to top
    reranked = reranker.rerank(candidates, observed_sensors={"DV_pressure": 0.75}, symptom_key="dryer_purge_leak")
    assert reranked[0].candidate_id == "dryer_purge_leak"
    assert reranked[0].reranked_score > reranked[1].reranked_score
    assert len(reranked[0].procedural_checklist) > 0


def test_10_point_diagnostic_report_generation():
    gen = DiagnosticReportGenerator()
    anomaly_res = HybridIndustrialAnomalyDetector().detect(np.zeros(22))
    classifier_res = IndustrialFaultClassifier().predict(np.zeros(22))
    reranker = ProceduralReranker()
    hist_intel = HistoricalIncidentIntelligence()

    reranked = reranker.rerank(
        [{"fault_id": "dryer_purge_leak", "name": "Dryer purge leak", "probability": 0.85}],
        {"DV_pressure": 0.80},
        "dryer_purge_leak",
    )
    historical = hist_intel.search_similar({"DV_pressure": 0.80}, ["purge"], top_k=1)

    report = gen.generate(
        unit_id="cau-7",
        current_sensors={"TP2": 9.5, "TP3": 6.8, "DV_pressure": 0.80},
        anomaly_res=anomaly_res,
        classifier_res=classifier_res,
        reranked_candidates=reranked,
        historical_cases=historical,
    )

    assert isinstance(report, DiagnosticReport)
    d = report.to_dict()
    # Validate presence of all 10 points
    assert "detected_abnormal_behaviour" in d
    assert "observed_sensor_evidence" in d
    assert "potential_fault_categories" in d
    assert "most_likely_root_cause" in d
    assert "supporting_manual_references" in d
    assert "similar_historical_incidents" in d
    assert "severity_assessment" in d
    assert "confidence_and_uncertainty" in d
    assert "recommended_diagnostic_procedures" in d
    assert "human_verification_requirement" in d
    assert d["human_verification_requirement"]["required"] is True

