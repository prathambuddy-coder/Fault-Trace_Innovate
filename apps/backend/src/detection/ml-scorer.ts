// SPDX-FileCopyrightText: 2026 Meddle S.r.l.
// SPDX-License-Identifier: Apache-2.0

import type { FeatureFrame } from "./types.ts";

export interface AnomalyScoreResult {
  readonly anomalyScore: number;       // 0.0 to 1.0
  readonly isAnomaly: boolean;
  readonly topFeatures: readonly string[];
}

/**
 * Computes an ensemble statistical anomaly score from the current FeatureFrame.
 */
export function computeAnomalyScore(frame: FeatureFrame): AnomalyScoreResult {
  let score = 0.10;
  const topFeatures: string[] = [];

  // Check dryer purge pressure deviation
  if (frame.purge_pressure) {
    if (frame.purge_pressure.value > 0.30 && frame.mode === "loaded") {
      score += 0.55;
      topFeatures.push(`DV_pressure: ${frame.purge_pressure.value.toFixed(2)} bar (elevated while loaded)`);
    }
  }

  // Check pressure decay slope
  if (frame.line_pressure_slope) {
    if (frame.line_pressure_slope.value < -0.20 && frame.mode === "unloaded") {
      score += 0.40;
      topFeatures.push(`Line pressure slope: ${frame.line_pressure_slope.value.toFixed(3)} bar/min (rapid decay)`);
    }
  }

  // Check oil temperature trend
  if (frame.oil_temperature_trend) {
    if (frame.oil_temperature_trend.value > 3.0) {
      score += 0.35;
      topFeatures.push(`Oil temperature rise: +${frame.oil_temperature_trend.value.toFixed(1)} °C/hr`);
    }
  }

  // Normalize score to [0.0, 1.0]
  const finalScore = Math.min(0.99, Math.max(0.05, score));
  const isAnomaly = finalScore >= 0.60;

  return {
    anomalyScore: Math.round(finalScore * 1000) / 1000,
    isAnomaly,
    topFeatures,
  };
}

