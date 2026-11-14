// SPDX-FileCopyrightText: 2026 Meddle S.r.l.
// SPDX-License-Identifier: Apache-2.0

import type { Candidate } from "./types.ts";

/**
 * Technical SOP checklists for standard maintenance procedures.
 */
export const SOP_PROCEDURES: Readonly<Record<string, readonly string[]>> = {
  dryer_purge_leak: [
    "1. Isolate compressor unit and lock out electrical power (LOTO protocol).",
    "2. Check dryer purge silencer for audible airflow during loaded duty cycle.",
    "3. Measure backpressure at test port M1 (must read < 0.05 bar when closed).",
    "4. Disassemble purge exhaust solenoid; inspect Viton valve seat for particulate debris.",
    "5. Replace internal valve seal kit and torque bonnet bolts to 18 Nm.",
  ],
  downstream_air_leak: [
    "1. Perform ultrasonic acoustic leak scan along auxiliary distribution manifold.",
    "2. Inspect isolation valve packings and threaded receiver joints with soapy bubble solution.",
    "3. Replace vibrating quick-connect couplings with flexible braided stainless connectors.",
    "4. Re-pressurize system and verify line decay rate is below 0.05 bar/10 min while unloaded.",
  ],
  oil_cooler_fouled: [
    "1. Check ambient air cooling inlet temperature (must be < 40°C).",
    "2. Inspect external aluminum radiator fin matrix for dust/oil encapsulation.",
    "3. Reverse-flush cooling matrix with dry compressed air lance or biodegradable fin cleaner.",
    "4. Verify thermostatic bypass valve begins opening at 71°C and fully opens at 82°C.",
  ],
  oil_filter_blocked: [
    "1. Check separator differential pressure gauge (H1 vs TP3); replace if delta > 1.2 bar.",
    "2. Drain oil reservoir into clean container below sight glass level.",
    "3. Replace spin-on oil filter and internal air/oil separator element.",
    "4. Verify electrical bonding grounding staple on replacement element before torquing.",
  ],
  high_air_demand: [
    "1. Audit active factory floor pneumatic consumption across all production lines.",
    "2. Verify plant tooling usage does not exceed compressor volumetric displacement rating.",
    "3. Stagger high-draw equipment cycles and check intermediate receiver storage.",
  ],
};

/**
 * Re-scores retrieved candidate faults based on physical symptom agreement.
 */
export function rerankCandidatesWithPhysics(
  candidates: readonly Candidate[],
  symptomKey: string,
): Candidate[] {
  return [...candidates].sort((a, b) => {
    let scoreA = a.score.fused;
    let scoreB = b.score.fused;

    // Prioritize direct symptom matches
    if (a.fault_id === symptomKey) scoreA += 0.25;
    if (b.fault_id === symptomKey) scoreB += 0.25;

    return scoreB - scoreA;
  });
}

