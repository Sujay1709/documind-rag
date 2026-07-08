// Browser-side PDF parsing using PDF.js (loaded from CDN).
//
// Each uploaded PDF becomes:
//   [{ page: 0, text: "..." }, { page: 1, text: "..." }, ...]
// This mirrors the per-page document objects that the server-side
// ingestion produces (one Document per page), so the chunker can
// preserve page metadata for citations.

let pdfjsPromise = null;

function loadPdfjs() {
  if (pdfjsPromise) return pdfjsPromise;
  pdfjsPromise = import(
    /* @vite-ignore */ "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.0.379/build/pdf.min.mjs"
  ).then((mod) => {
    // Disable the worker (we don't ship one in a static space); PDF.js
    // can run on the main thread for small/medium PDFs without blocking
    // visibly because the rest of the pipeline is async.
    mod.GlobalWorkerOptions.workerSrc =
      "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.0.379/build/pdf.worker.min.mjs";
    return mod;
  });
  return pdfjsPromise;
}

export async function parsePdf(file) {
  const pdfjs = await loadPdfjs();
  const buf = await file.arrayBuffer();
  const doc = await pdfjs.getDocument({ data: buf }).promise;
  const pages = [];
  for (let i = 1; i <= doc.numPages; i++) {
    const page = await doc.getPage(i);
    const content = await page.getTextContent();
    // PDF.js returns a stream of text items; their coordinates aren't
    // useful for our purposes, so we just concatenate the strings and
    // collapse runs of whitespace. Page numbers are 1-indexed for
    // human-facing citations; we shift to 0-indexed when displaying.
    const text = content.items
      .map((it) => ("str" in it ? it.str : ""))
      .join(" ");
    const cleaned = text.replace(/\s+/g, " ").trim();
    if (cleaned) {
      pages.push({ page: i - 1, text: cleaned });
    }
  }
  return pages;
}
