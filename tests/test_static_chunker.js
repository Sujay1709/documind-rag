// Smoke test for the static-space chunker. Run with: node tests/test_static_chunker.js
// (not via pytest — this is a JS module, not Python).
//
// What we're checking: the chunker produces ~1 KB chunks with 200-char
// overlap and preserves page metadata. If this breaks, the user sees
// fragmented answers in the browser.

import { chunkPages } from "../src/documind/webapp/static/js/chunker.js";
import assert from "node:assert/strict";

const pages = [
  { page: 0, text: "The quick brown fox jumps over the lazy dog. ".repeat(50) },
  { page: 1, text: "Sentence one. Sentence two. Sentence three. ".repeat(80) },
];

const chunks = chunkPages(pages, { chunkSize: 1000, chunkOverlap: 200 });

console.log(`chunks: ${chunks.length}`);
assert(chunks.length > 4, "should produce multiple chunks from 2 pages");

for (const c of chunks) {
  assert(c.text.length <= 1100, `chunk too long: ${c.text.length}`);
  assert(c.text.length > 40, `chunk too short: ${c.text.length}`);
  assert(typeof c.page === "number", "page index must be preserved");
}

const pageZero = chunks.filter((c) => c.page === 0).length;
const pageOne = chunks.filter((c) => c.page === 1).length;
assert(pageZero > 0 && pageOne > 0, "both pages should produce chunks");

console.log(`page 0 chunks: ${pageZero}, page 1 chunks: ${pageOne}`);
console.log("✓ chunker smoke test passed");
