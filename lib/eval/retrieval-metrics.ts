export interface EvalCase {
  question: string;
  /** Chunk must contain ALL of these substrings (case-insensitive) to count as a hit. */
  mustContain?: string[];
  /** Chunk must contain AT LEAST ONE of these substrings (case-insensitive) to count as a hit. */
  anyOf?: string[];
}

/** Whether a single retrieved chunk's text satisfies an eval case's answer criteria. */
export function chunkMatchesCase(chunkText: string, evalCase: EvalCase): boolean {
  const lower = chunkText.toLowerCase();
  const mustContainOk = (evalCase.mustContain ?? []).every((s) => lower.includes(s.toLowerCase()));
  const anyOfOk = !evalCase.anyOf || evalCase.anyOf.some((s) => lower.includes(s.toLowerCase()));
  return mustContainOk && anyOfOk;
}

/** True if any of the top-k retrieved chunks (in rank order) satisfy the case. */
export function hitAtK(rankedChunkTexts: string[], evalCase: EvalCase, k: number): boolean {
  return rankedChunkTexts.slice(0, k).some((text) => chunkMatchesCase(text, evalCase));
}

/**
 * Reciprocal rank of the first matching chunk (1-based), or 0 if none of the
 * retrieved chunks match. Used for Mean Reciprocal Rank (MRR).
 */
export function reciprocalRankOfFirstHit(rankedChunkTexts: string[], evalCase: EvalCase): number {
  const index = rankedChunkTexts.findIndex((text) => chunkMatchesCase(text, evalCase));
  return index === -1 ? 0 : 1 / (index + 1);
}

export interface MethodResult {
  method: string;
  recallAtK: number;
  mrr: number;
  k: number;
  caseCount: number;
}

export function summarizeMethod(
  method: string,
  perCaseRankedTexts: string[][],
  cases: EvalCase[],
  k: number
): MethodResult {
  let hits = 0;
  let rrSum = 0;
  cases.forEach((evalCase, i) => {
    const ranked = perCaseRankedTexts[i] ?? [];
    if (hitAtK(ranked, evalCase, k)) hits += 1;
    rrSum += reciprocalRankOfFirstHit(ranked, evalCase);
  });
  return {
    method,
    recallAtK: cases.length === 0 ? 0 : hits / cases.length,
    mrr: cases.length === 0 ? 0 : rrSum / cases.length,
    k,
    caseCount: cases.length,
  };
}
