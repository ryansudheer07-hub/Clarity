export interface Chunk {
  text: string;
  position: number;
}

const CHUNK_SIZE_CHARS = 2000; // ~500 tokens
const OVERLAP_CHARS = 200; // ~50 tokens

/** Splits raw text into overlapping chunks for embedding. */
export function chunkText(text: string): Chunk[] {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return [];

  const chunks: Chunk[] = [];
  let start = 0;
  let position = 0;

  while (start < clean.length) {
    const end = Math.min(start + CHUNK_SIZE_CHARS, clean.length);
    chunks.push({ text: clean.slice(start, end), position: position++ });
    if (end === clean.length) break;
    start = end - OVERLAP_CHARS;
  }

  return chunks;
}

export interface RetrievedChunk {
  chunkText: string;
  similarity: number;
}

/** Formats retrieved chunks for injection into a prompt's context section. */
export function formatContext(chunks: RetrievedChunk[]): string {
  if (chunks.length === 0) return "(no relevant context found)";
  return chunks.map((c, i) => `[${i + 1}] ${c.chunkText}`).join("\n\n");
}

// ---------------------------------------------------------------------------
// Retrieval-fusion helpers used by the eval harness (scripts/eval-retrieval.ts)
// to compare retrieval methods in-memory. These will also back a future
// server-side hybrid (vector + keyword) search — see the RAG improvement
// plan for the next step.
// ---------------------------------------------------------------------------

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error("vector length mismatch");
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * Crude keyword relevance score: fraction of unique query terms that
 * appear in the candidate text, weighted by term frequency. Not meant to
 * replace Postgres full-text search (which uses proper stemming/ranking in
 * production) — this is a lightweight stand-in for local evaluation.
 */
export function keywordScore(query: string, text: string): number {
  const queryTerms = tokenize(query);
  if (queryTerms.length === 0) return 0;
  const textTokens = tokenize(text);
  const textCounts = new Map<string, number>();
  for (const t of textTokens) textCounts.set(t, (textCounts.get(t) ?? 0) + 1);

  let score = 0;
  for (const term of new Set(queryTerms)) {
    const count = textCounts.get(term) ?? 0;
    if (count > 0) score += 1 + Math.log(count);
  }
  return score / queryTerms.length;
}

function tokenize(text: string): string[] {
  return (
    text
      .toLowerCase()
      .match(/[a-z0-9]+/g)
      ?.filter((t) => t.length > 1) ?? []
  );
}

export interface RankedItem<T> {
  item: T;
  rank: number; // 1-based rank within this ranking
}

/**
 * Reciprocal Rank Fusion: combines multiple ranked lists of the same items
 * into a single ranking using only rank position (not raw scores), which
 * avoids the need to normalize incomparable scores (cosine similarity vs.
 * a keyword-match score) onto the same scale. k=60 is the standard constant
 * from the original RRF paper (Cormack et al.) and is not sensitive to
 * tuning in practice.
 */
export function reciprocalRankFusion<T>(
  rankings: RankedItem<T>[][],
  k = 60
): { item: T; score: number }[] {
  const scores = new Map<T, number>();
  for (const ranking of rankings) {
    for (const { item, rank } of ranking) {
      scores.set(item, (scores.get(item) ?? 0) + 1 / (k + rank));
    }
  }
  return [...scores.entries()]
    .map(([item, score]) => ({ item, score }))
    .sort((a, b) => b.score - a.score);
}
