// Sentence-aware recursive chunker.
//
// Mirrors the server's _build_splitter in src/documind/ingestion.py:
// prefer paragraph breaks, then lines, then sentence ends, then words.
// We don't have to match the server's defaults exactly; we just need
// 1k-char chunks with 200-char overlap and sentence boundaries.

const SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""];

function splitText(text, chunkSize, chunkOverlap) {
  const out = [];
  // Recursive: try the first separator; if a piece is too long, recurse
  // on it with the next separator; if no separator works, hard-cut.
  function recurse(t, sepIdx) {
    if (t.length <= chunkSize) {
      if (t.trim()) out.push(t);
      return;
    }
    const sep = SEPARATORS[sepIdx];
    if (!sep) {
      // Hard cut at chunkSize. Don't split mid-word if we can help it.
      let cut = chunkSize;
      const lastSpace = t.lastIndexOf(" ", cut);
      if (lastSpace > chunkSize * 0.7) cut = lastSpace;
      out.push(t.slice(0, cut));
      // Continue with the rest, recursing at the most-coarse separator.
      recurse(t.slice(cut), 0);
      return;
    }
    const parts = t.split(sep);
    let buf = "";
    for (let i = 0; i < parts.length; i++) {
      const piece = parts[i] + (i < parts.length - 1 ? sep : "");
      if ((buf + piece).length > chunkSize && buf) {
        out.push(buf);
        // Overlap: keep the last chunkOverlap chars of buf as the start
        // of the next chunk so the re-ranker has continuity.
        const tail = buf.slice(Math.max(0, buf.length - chunkOverlap));
        buf = tail + piece;
      } else {
        buf += piece;
      }
    }
    if (buf.trim()) {
      // If the leftover is still too long, recurse on it with the next
      // separator. Otherwise push it.
      if (buf.length > chunkSize) {
        recurse(buf, sepIdx + 1);
      } else {
        out.push(buf);
      }
    }
  }
  recurse(text, 0);
  return out;
}

// Split a list of {page, text} pages into chunked records, preserving
// the page index on every chunk so citations can show "p. 12".
export function chunkPages(pages, { chunkSize = 1000, chunkOverlap = 200 } = {}) {
  const chunks = [];
  for (const p of pages) {
    const pieces = splitText(p.text, chunkSize, chunkOverlap);
    for (const text of pieces) {
      if (text.trim().length < 40) continue; // drop boilerplate
      chunks.push({ page: p.page, text: text.trim() });
    }
  }
  return chunks;
}
