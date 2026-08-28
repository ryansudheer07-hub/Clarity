/**
 * Retrieval quality eval harness.
 *
 * Runs entirely in-memory — no Supabase connection required — using the
 * exact same chunking and fusion logic as production (lib/rag.ts). This
 * lets you measure retrieval quality (recall@k, MRR) before and after
 * changes to chunking, and compare vector-only vs. hybrid retrieval, in
 * seconds instead of needing a deployed DB with real data.
 *
 * Usage:
 *   npx tsx scripts/eval-retrieval.ts
 *   npx tsx scripts/eval-retrieval.ts --k 3
 *   npx tsx scripts/eval-retrieval.ts --file data/eval/my-course-eval.json
 *
 * Without VOYAGE_API_KEY set, this runs against the deterministic mock
 * embeddings (see lib/embeddings.ts) — useful for verifying the harness
 * itself works end-to-end, but recall numbers will be meaningless since
 * mock embeddings carry no semantic signal. Set a real VOYAGE_API_KEY in
 * your environment to get numbers that reflect actual retrieval quality.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  chunkText,
  cosineSimilarity,
  keywordScore,
  reciprocalRankFusion,
  type RankedItem,
} from "../lib/rag";
import { embed, isEmbeddingsMocked } from "../lib/embeddings";
import {
  summarizeMethod,
  chunkMatchesCase,
  type EvalCase,
  type MethodResult,
} from "../lib/eval/retrieval-metrics";

interface EvalFile {
  sourceText: string;
  cases: EvalCase[];
}

function parseArgs() {
  const args = process.argv.slice(2);
  const kIndex = args.indexOf("--k");
  const fileIndex = args.indexOf("--file");
  return {
    k: kIndex !== -1 ? Number(args[kIndex + 1]) : 5,
    file: fileIndex !== -1 ? args[fileIndex + 1] : "data/eval/retrieval-eval-set.json",
  };
}

function toRanking(scores: number[]): RankedItem<number>[] {
  // item = index into the chunk array; rank = 1-based position after
  // sorting by score descending.
  return scores
    .map((score, index) => ({ index, score }))
    .sort((a, b) => b.score - a.score)
    .map(({ index }, rank) => ({ item: index, rank: rank + 1 }));
}

async function main() {
  const { k, file } = parseArgs();

  const evalFile: EvalFile = JSON.parse(readFileSync(resolve(process.cwd(), file), "utf-8"));
  const chunks = chunkText(evalFile.sourceText);

  console.log(`\nRetrieval eval: ${file}`);
  console.log(`Source chunked into ${chunks.length} chunks. ${evalFile.cases.length} eval cases. k=${k}.`);
  if (isEmbeddingsMocked) {
    console.log(
      "⚠ VOYAGE_API_KEY is not set — using mock embeddings. Recall numbers below are NOT meaningful; " +
        "this only verifies the harness runs. Set VOYAGE_API_KEY for a real measurement.\n"
    );
  } else {
    console.log("");
  }

  const chunkEmbeddings = await embed(chunks.map((c) => c.text));

  const vectorRankedPerCase: string[][] = [];
  const keywordRankedPerCase: string[][] = [];
  const hybridRankedPerCase: string[][] = [];

  for (const evalCase of evalFile.cases) {
    const [questionEmbedding] = await embed([evalCase.question]);

    const vectorScores = chunkEmbeddings.map((e) => cosineSimilarity(questionEmbedding, e));
    const keywordScores = chunks.map((c) => keywordScore(evalCase.question, c.text));

    const vectorRanking = toRanking(vectorScores);
    const keywordRanking = toRanking(keywordScores);
    const fused = reciprocalRankFusion([vectorRanking, keywordRanking]);

    vectorRankedPerCase.push(vectorRanking.sort((a, b) => a.rank - b.rank).map((r) => chunks[r.item].text));
    keywordRankedPerCase.push(
      keywordRanking.sort((a, b) => a.rank - b.rank).map((r) => chunks[r.item].text)
    );
    hybridRankedPerCase.push(fused.map((f) => chunks[f.item].text));
  }

  const results: MethodResult[] = [
    summarizeMethod("vector-only", vectorRankedPerCase, evalFile.cases, k),
    summarizeMethod("keyword-only", keywordRankedPerCase, evalFile.cases, k),
    summarizeMethod("hybrid (RRF)", hybridRankedPerCase, evalFile.cases, k),
  ];

  const rows = results.map((r) => ({
    method: r.method,
    [`recall@${k}`]: `${(r.recallAtK * 100).toFixed(0)}%`,
    mrr: r.mrr.toFixed(3),
  }));
  console.table(rows);

  // Flag individual misses so you can see *which* questions are failing,
  // not just the aggregate score.
  const misses = evalFile.cases.filter(
    (evalCase, i) =>
      !hybridRankedPerCase[i].slice(0, k).some((text) => chunkMatchesCase(text, evalCase))
  );

  if (misses.length > 0) {
    console.log(`\nMisses under hybrid retrieval (${misses.length}/${evalFile.cases.length}):`);
    for (const m of misses) {
      console.log(`  - "${m.question}"`);
    }
  } else {
    console.log("\nNo misses under hybrid retrieval at this k. 🎉");
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
