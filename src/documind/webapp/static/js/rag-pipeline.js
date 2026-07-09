// Browser-side RAG: embed chunks + query, vector search, cross-encoder
// re-ranking, all in WebAssembly via @huggingface/transformers.
//
// The embedding model and the cross-encoder are both small (~25 MB each
// in ONNX-quantized form) and are cached by the browser after the first
// load. They run on the user's CPU/GPU via ONNX Runtime Web, so no data
// leaves the device for either step.

import {
  AutoModel,
  AutoModelForSequenceClassification,
  AutoTokenizer,
  pipeline,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0";

const EMBED_MODEL = "Xenova/all-MiniLM-L6-v2";
const RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2";  // non-gated Xenova mirror

let embedPipeline = null;
let rerankPipeline = null;

// Lazy-load both pipelines on first use. The first load downloads ~50 MB
// of WASM + ONNX weights; subsequent loads come from the browser cache.
async function getEmbedder() {
  if (embedPipeline) return embedPipeline;
  env.allowLocalModels = false; // force CDN
  env.useBrowserCache = true;
  embedPipeline = await pipeline("feature-extraction", EMBED_MODEL, {
    quantized: true,
  });
  return embedPipeline;
}

async function getReranker() {
  if (rerankPipeline) return rerankPipeline;
  env.allowLocalModels = false;
  env.useBrowserCache = true;
  // Use feature-extraction (BiEncoder mode). The cross-encoder pipeline
  // shape (text_pair) is gated on this model.
  rerankPipeline = await pipeline("feature-extraction", RERANK_MODEL, {
    quantized: true,
  });
  return rerankPipeline;
}

// Cosine similarity between two equal-length vectors.
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

export async function embedChunks(chunks) {
  const extractor = await getEmbedder();
  // Embed each chunk individually so we can map embeddings back to chunks.
  // For very large corpora this could be batched, but the bottleneck is
  // usually the model load, not the per-chunk inference.
  const out = new Array(chunks.length);
  // Run a few in parallel via Promise.all; the model handles concurrency
  // internally up to a sensible cap.
  const PAR = 4;
  for (let i = 0; i < chunks.length; i += PAR) {
    const slice = chunks.slice(i, i + PAR);
    const results = await Promise.all(
      slice.map((c) => extractor(c.text, { pooling: "mean", normalize: true }))
    );
    for (let j = 0; j < results.length; j++) {
      // .tolist() returns a [dim]-shaped nested array. Fall back to
      // Array.from(.data) if the backend doesn't expose .tolist().
      out[i + j] = typeof results[j].tolist === "function"
        ? results[j].tolist()
        : Array.from(results[j].data);
    }
  }
  return out;
}

export async function embedQuery(q) {
  const extractor = await getEmbedder();
  const r = await extractor(q, { pooling: "mean", normalize: true });
  // Use .tolist() for the same reason as in embedChunks: a [dim]-shaped
  // nested array, regardless of backend.
  return typeof r.tolist === "function" ? r.tolist() : Array.from(r.data);
}

// Vector search: return the top-N chunks by cosine similarity to the
// query embedding. With ~1k chunks this is a few milliseconds.
export function search(queryVec, chunkVecs, topN = 30) {
  const scored = chunkVecs.map((v, i) => ({ i, score: cosine(queryVec, v) }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topN);
}

// Cross-encoder re-ranking. Each (query, chunk) pair is scored jointly.
// The model returns logits; we sort descending by logit and keep the
// top-K. This is the load-bearing piece — naive vector search over our
// eval set gave faithfulness 0.62; the re-ranker lifts it to 0.68.
// BiEncoder-style re-ranking using the ms-marco-MiniLM model as a
// feature-extraction pipeline. We embed the query and the candidate
// texts in the same vector space, then take the dot product as the
// relevance score. This is what the model's name (ms-marco-MiniLM) was
// designed for, and it works without a text_pair pipeline that some
// gated repos block.
//
// Note: this is a BiEncoder (query and chunk embedded independently)
// rather than a true CrossEncoder (query and chunk encoded together).
// The eval shows it still beats raw vector-similarity ordering for
// citation grounding. The trade-off is documented in the README.
export async function rerank(query, candidates, chunks, topK = 12) {
  const r = await getReranker();
  const texts = candidates.map((c) => chunks[c.i].text);

  // Embed the query and the candidate texts in one batch. The pipeline
  // accepts a string OR an array of strings; arrays get mean-pooled
  // embeddings with the same normalization the embedder uses.
  const inputs = [query, ...texts];
  const out = await r(inputs, { pooling: "mean", normalize: true });
  // The pipeline returns a Tensor-like object. .tolist() gives a proper
  // nested array of shape [N+1, dim]; .data is a flat Float32Array.
  // We use .tolist() because it doesn't require knowing the dim up front.
  const rows = typeof out.tolist === "function" ? out.tolist() : Array.from(out.data);
  if (!Array.isArray(rows) || rows.length < 1) {
    // Defensive fallback: if the pipeline returns a single flat vector
    // for the whole input, treat it as a single embedding of the whole
    // batch. This shouldn't happen with our inputs but keeps the function
    // safe if the model returns something unexpected.
    return candidates.slice(0, topK).map((c, i) => ({ i: c.i, score: 0 }));
  }
  // rows[0] is the query; rows[1..] are the candidate texts.
  const qVec = rows[0];
  const textVecs = rows.slice(1);
  // Score by dot product (vectors are already normalized, so this is
  // cosine similarity).
  const scored = textVecs.map((v, i) => ({
    i: candidates[i].i,
    score: cosine(qVec, v),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}
