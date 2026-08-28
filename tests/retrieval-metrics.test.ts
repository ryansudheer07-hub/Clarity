import { describe, expect, it } from "vitest";
import {
  chunkMatchesCase,
  hitAtK,
  reciprocalRankOfFirstHit,
  summarizeMethod,
} from "@/lib/eval/retrieval-metrics";

describe("chunkMatchesCase", () => {
  it("requires all mustContain terms, case-insensitively", () => {
    const evalCase = { question: "q", mustContain: ["Mitochondria", "ATP"] };
    expect(chunkMatchesCase("mitochondria produce atp", evalCase)).toBe(true);
    expect(chunkMatchesCase("mitochondria produce energy", evalCase)).toBe(false);
  });

  it("requires at least one anyOf term", () => {
    const evalCase = { question: "q", anyOf: ["ribosome", "translate"] };
    expect(chunkMatchesCase("ribosomes make proteins", evalCase)).toBe(true);
    expect(chunkMatchesCase("nothing relevant here", evalCase)).toBe(false);
  });

  it("combines mustContain and anyOf when both present", () => {
    const evalCase = { question: "q", mustContain: ["cell"], anyOf: ["nucleus", "membrane"] };
    expect(chunkMatchesCase("the cell has a nucleus", evalCase)).toBe(true);
    expect(chunkMatchesCase("the organ has a nucleus", evalCase)).toBe(false);
  });
});

describe("hitAtK", () => {
  it("finds a hit within the top-k window", () => {
    const evalCase = { question: "q", mustContain: ["mitochondria"] };
    const ranked = ["photosynthesis text", "mitochondria text", "ribosome text"];
    expect(hitAtK(ranked, evalCase, 2)).toBe(true);
  });

  it("misses when the match is outside the top-k window", () => {
    const evalCase = { question: "q", mustContain: ["ribosome"] };
    const ranked = ["photosynthesis text", "mitochondria text", "ribosome text"];
    expect(hitAtK(ranked, evalCase, 2)).toBe(false);
  });
});

describe("reciprocalRankOfFirstHit", () => {
  it("returns 1 when the first result matches", () => {
    const evalCase = { question: "q", mustContain: ["atp"] };
    expect(reciprocalRankOfFirstHit(["contains atp"], evalCase)).toBe(1);
  });

  it("returns 0 when nothing matches", () => {
    const evalCase = { question: "q", mustContain: ["atp"] };
    expect(reciprocalRankOfFirstHit(["irrelevant"], evalCase)).toBe(0);
  });
});

describe("summarizeMethod", () => {
  it("computes recall@k and MRR across cases", () => {
    const cases = [
      { question: "a", mustContain: ["alpha"] },
      { question: "b", mustContain: ["beta"] },
    ];
    const rankedPerCase = [
      ["alpha text", "other"], // hit at rank 1
      ["irrelevant", "other"], // no hit
    ];
    const result = summarizeMethod("vector", rankedPerCase, cases, 5);
    expect(result.recallAtK).toBe(0.5);
    expect(result.mrr).toBe(0.5); // (1 + 0) / 2
  });
});
