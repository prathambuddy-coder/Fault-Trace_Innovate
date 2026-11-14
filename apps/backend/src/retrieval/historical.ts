// SPDX-FileCopyrightText: 2026 Meddle S.r.l.
// SPDX-License-Identifier: Apache-2.0

/**
 * Historical Maintenance Incident Intelligence.
 *
 * Stores verified historical incidents across fleet units (CAU-1 to CAU-6)
 * for case-based reasoning and similarity retrieval during fault triage.
 */

export interface HistoricalIncident {
  readonly incidentId: string;
  readonly machineId: string;
  readonly timestamp: string;
  readonly observedSymptoms: string;
  readonly identifiedFault: string;
  readonly rootCause: string;
  readonly maintenanceAction: string;
  readonly resolutionStatus: "verified_resolved" | "benign_verified";
  readonly severity: "low" | "medium" | "high" | "critical";
  readonly similarityScore: number;
}

export const HISTORICAL_INCIDENTS: readonly Omit<HistoricalIncident, "similarityScore">[] = [
  {
    incidentId: "INC-2019-084",
    machineId: "CAU-3",
    timestamp: "2019-08-14T04:22:10Z",
    observedSymptoms:
      "Compressor stayed loaded continuously (>25 min) without reaching cut-out pressure. Dryer purge pressure elevated at 0.78 bar during loaded phase.",
    identifiedFault: "dryer_purge_leak",
    rootCause:
      "Desiccant dryer purge exhaust solenoid valve seat contaminated with particulate debris, preventing tight closure during compression cycle.",
    maintenanceAction:
      "Disassembled purge solenoid valve, cleaned brass seat, replaced internal Viton seal, and verified pressure drop below 0.05 bar.",
    resolutionStatus: "verified_resolved",
    severity: "high",
  },
  {
    incidentId: "INC-2019-112",
    machineId: "CAU-2",
    timestamp: "2019-10-03T11:45:00Z",
    observedSymptoms:
      "Line pressure decay rate exceeded 0.15 bar/min while unloaded. Cut-in cycle triggered every 90 seconds under low plant demand.",
    identifiedFault: "downstream_air_leak",
    rootCause:
      "Pneumatic quick-connect fitting on auxiliary air receiver manifold cracked due to structural vibration fatigue.",
    maintenanceAction:
      "Replaced manifold coupling fitting with vibration-damped SS union and performed ultrasonic leak inspection.",
    resolutionStatus: "verified_resolved",
    severity: "medium",
  },
  {
    incidentId: "INC-2020-019",
    machineId: "CAU-5",
    timestamp: "2020-01-22T14:10:30Z",
    observedSymptoms:
      "Oil temperature climbed steadily past 88°C under normal ambient conditions (16°C). Temperature trend was +4.5°C/hr over a 4-hour window.",
    identifiedFault: "oil_cooler_fouled",
    rootCause:
      "External radiator fin bank of air-blast oil cooler clogged with industrial dust and lint, reducing thermal dissipation efficiency by ~40%.",
    maintenanceAction:
      "Cleaned heat exchanger core with compressed nitrogen lance and biodegradable fin wash; verified operating oil temperature dropped to 68°C.",
    resolutionStatus: "verified_resolved",
    severity: "high",
  },
  {
    incidentId: "INC-2020-031",
    machineId: "CAU-1",
    timestamp: "2020-02-18T08:30:00Z",
    observedSymptoms:
      "Differential pressure between separator and main line exceeded 1.8 bar during peak loaded run. Motor current elevated by 12% above nominal.",
    identifiedFault: "oil_filter_blocked",
    rootCause:
      "Oil separator cartridge saturated with oxidized varnish and particulate deposits beyond 2,000-hour service interval.",
    maintenanceAction:
      "Replaced air/oil separator element and spin-on oil filter; refilled with fresh synthetic ISO VG 46 compressor lubricant.",
    resolutionStatus: "verified_resolved",
    severity: "medium",
  },
  {
    incidentId: "INC-2020-044",
    machineId: "CAU-4",
    timestamp: "2020-03-05T16:00:00Z",
    observedSymptoms:
      "Rapid pressure drop and continuous loaded run observed during factory shift changeover. Both line and reservoir pressures stabilized at 7.2 bar.",
    identifiedFault: "high_air_demand",
    rootCause:
      "Three pneumatic blast cabinets operated simultaneously on Section 4 assembly line, temporarily exceeding compressor volumetric capacity.",
    maintenanceAction:
      "No machine repair needed. Staggered blast cabinet schedules and adjusted intermediate receiver regulator setpoint.",
    resolutionStatus: "benign_verified",
    severity: "low",
  },
];

/**
 * Searches historical incidents for cases matching the current fault ID and symptom context.
 */
export function findSimilarHistoricalIncidents(
  targetFaultId: string,
  limit: number = 2,
): HistoricalIncident[] {
  const matches: HistoricalIncident[] = [];

  for (const inc of HISTORICAL_INCIDENTS) {
    let score = 0.3; // baseline similarity
    if (inc.identifiedFault === targetFaultId) {
      score = 0.95;
    } else if (
      (targetFaultId.includes("leak") && inc.identifiedFault.includes("leak")) ||
      (targetFaultId.includes("oil") && inc.identifiedFault.includes("oil"))
    ) {
      score = 0.72;
    }

    matches.push({
      ...inc,
      similarityScore: score,
    });
  }

  matches.sort((a, b) => b.similarityScore - a.similarityScore);
  return matches.slice(0, limit);
}

