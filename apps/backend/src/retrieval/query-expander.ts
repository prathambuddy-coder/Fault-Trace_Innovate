// SPDX-FileCopyrightText: 2026 Meddle S.r.l.
// SPDX-License-Identifier: Apache-2.0

/**
 * Technical Query Expansion and Terminology Normalization for RAG.
 */

export interface ExpandedQuery {
  readonly original: string;
  readonly normalized: string;
  readonly expandedTerms: readonly string[];
  readonly candidateFaultHints: readonly string[];
}

const TECHNICAL_SYNONYMS: Readonly<Record<string, readonly string[]>> = {
  leak: ["dryer_purge_leak", "downstream_air_leak", "line_pressure_decay", "solenoid seat"],
  purge: ["dryer_purge_leak", "dv_pressure", "purge exhaust", "purge valve"],
  loaded: ["stays loaded", "does not reach cut-out", "continuous load", "unloader"],
  overheating: ["oil_cooler_fouled", "oil temperature", "thermal dissipation", "radiator"],
  filter: ["oil_filter_blocked", "differential pressure", "separator cartridge"],
  demand: ["high_air_demand", "heavy plant load", "consumption exceeds rating"],
};

export function expandTechnicalQuery(rawText: string): ExpandedQuery {
  const normalized = rawText.trim().toLowerCase();
  const expanded: string[] = [];
  const faultHints: string[] = [];

  for (const [key, synonyms] of Object.entries(TECHNICAL_SYNONYMS)) {
    if (normalized.includes(key)) {
      expanded.push(...synonyms);
      for (const s of synonyms) {
        if (s.includes("_")) faultHints.push(s);
      }
    }
  }

  return {
    original: rawText,
    normalized,
    expandedTerms: Array.from(new Set(expanded)),
    candidateFaultHints: Array.from(new Set(faultHints)),
  };
}

