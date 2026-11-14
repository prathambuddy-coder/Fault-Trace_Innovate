// SPDX-FileCopyrightText: 2026 Meddle S.r.l.
// SPDX-License-Identifier: Apache-2.0

/**
 * What a technician reads on a ticket.
 *
 * Enhanced with the 10-point explainable industrial diagnostic report,
 * historical fleet incident matches, and procedural SOP checklists.
 */

import type {
  CatalogEntry,
  Decision,
  DecisionBackend,
  EntryCondition,
  EvidenceItem,
  ManualReference,
  SeverityLevel,
  SuspectEvent,
} from "@fdp/contracts";

import { findSimilarHistoricalIncidents, type HistoricalIncident } from "../retrieval/historical.ts";
import { SOP_PROCEDURES } from "../retrieval/reranker.ts";

/**
 * The part of a ticket a decision produces.
 *
 * It is everything the `ticket` contract carries except the identifiers, the
 * lifecycle and the counters, which belong to the ticket itself and not to the
 * decision that last changed it.
 */
export interface RenderedTicket {
  readonly fault_id: string;
  readonly condition_id: string;
  /** `"<cause name> — <condition title>"`, the ticket header. */
  readonly title: string;
  readonly cause: string;
  readonly remedy: string;
  readonly checks: readonly string[];
  readonly manual_ref: ManualReference;
  readonly evidence: readonly EvidenceItem[];
  readonly confidence: number;
  readonly probabilities: Readonly<Record<string, number>>;
  readonly severity: SeverityLevel;
  readonly backend: DecisionBackend;
  readonly model: string;
  /** The language-model backend's short justification, when there is one. */
  readonly rationale: string | null;
  /** The 10-point structured industrial intelligence diagnostic report. */
  readonly diagnostic_report?: Readonly<Record<string, unknown>>;
  /** Similar historical fleet incidents. */
  readonly similar_incidents?: readonly HistoricalIncident[];
  /** Procedural step-by-step diagnostic checklist. */
  readonly sop_checklist?: readonly string[];
}

/** The em dash that joins the cause and the condition in a ticket title. */
const TITLE_SEPARATOR = " — ";

/**
 * The condition the ticket is about.
 */
function conditionFor(candidate: CatalogEntry, symptomKeys: readonly string[]): EntryCondition {
  for (const symptomKey of symptomKeys) {
    const listed = candidate.conditions.find((condition) => condition.condition_id === symptomKey);
    if (listed !== undefined) return listed;
  }
  return candidate.conditions[0];
}

/**
 * Render one ticket body from the chosen candidate, the decision and the event.
 *
 * @throws Error when `candidate` is not the cause the decision chose.
 */
export function renderTicket(
  candidate: CatalogEntry,
  decision: Decision,
  event: SuspectEvent,
  ticketSymptomKey: string = event.symptom_key,
): RenderedTicket {
  if (candidate.fault_id !== decision.choice) {
    throw new Error(
      `renderTicket: decision ${decision.decision_id} chose ${decision.choice}, ` +
        `not ${candidate.fault_id}`,
    );
  }

  const condition = conditionFor(candidate, [ticketSymptomKey, event.symptom_key]);

  // Find similar historical incidents
  const similarIncidents = findSimilarHistoricalIncidents(candidate.fault_id, 2);

  // Retrieve procedural SOP checklist
  const sopChecklist = SOP_PROCEDURES[candidate.fault_id] ?? [...candidate.checks];

  // Synthesize 10-Point Industrial Diagnostic Report
  const diagnosticReport: Record<string, unknown> = {
    // 1. Detected abnormal behaviour
    detected_abnormal_behaviour: `Machine ${event.unit_id} triggered symptom '${event.symptom_key}' during mode '${event.machine_mode}'.`,
    // 2. Observed sensor evidence
    observed_sensor_evidence: {
      observations: event.observations,
      evidence_items: event.evidence,
      machine_mode: event.machine_mode,
    },
    // 3. Potential fault categories
    potential_fault_categories: Object.entries(decision.probabilities)
      .sort(([, a], [, b]) => b - a)
      .map(([fault, prob]) => ({ fault_id: fault, probability: prob })),
    // 4. Most likely root cause
    most_likely_root_cause: `${candidate.name}: ${candidate.summary}`,
    // 5. Supporting maintenance manual references
    supporting_manual_references: [
      {
        document: "CAU-7 Service and Maintenance Manual",
        section: `Section ${candidate.manual_ref.section}: ${candidate.manual_ref.title}`,
        remedy: candidate.remedy,
      },
    ],
    // 6. Similar historical incidents
    similar_historical_incidents: similarIncidents,
    // 7. Severity assessment
    severity_assessment: {
      level: decision.severity.level,
      score: decision.severity.score,
      urgency: decision.severity.score >= 3 ? "Immediate Dispatch" : "Next Shift Maintenance",
    },
    // 8. Confidence and uncertainty breakdown
    confidence_and_uncertainty: {
      decision_confidence: decision.confidence,
      model_source: decision.backend,
      uncertainty_flag: decision.confidence < 0.70,
      uncertainty_rationale:
        decision.confidence >= 0.70
          ? "High model separation and manual congruence"
          : "Alternative causes close in probability mass; technician field verification required",
    },
    // 9. Recommended diagnostic procedures
    recommended_diagnostic_procedures: sopChecklist,
    // 10. Human verification requirement
    human_verification_requirement: {
      required: true,
      protocol: "Mandatory human technician sign-off before closing maintenance ticket.",
      gate_outcome: decision.gate.outcome,
    },
  };

  return {
    fault_id: candidate.fault_id,
    condition_id: condition.condition_id,
    title: `${candidate.name}${TITLE_SEPARATOR}${condition.title}`,
    cause: candidate.summary,
    remedy: candidate.remedy,
    checks: sopChecklist,
    manual_ref: candidate.manual_ref,
    evidence: [...event.evidence],
    confidence: decision.confidence,
    probabilities: { ...decision.probabilities },
    severity: decision.severity.level,
    backend: decision.backend,
    model: decision.model,
    rationale: decision.rationale ?? null,
    diagnostic_report: diagnosticReport,
    similar_incidents: similarIncidents,
    sop_checklist: sopChecklist,
  };
}
