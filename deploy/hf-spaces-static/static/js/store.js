// Tiny IndexedDB-backed store for chunks and their embeddings.
//
// Why IndexedDB: a visitor can come back tomorrow, reopen the same docs,
// and ask a new question without re-indexing. The whole thing is
// client-side: chunks, embeddings, and source metadata are stored in
// the browser's storage. We don't store the original PDF — only the
// extracted text and its embedding.

const DB_NAME = "documind-static";
const DB_VERSION = 1;
const STORE_DOCS = "docs"; // {id, name, chunks: [{page, text}], vectors: number[][]}
const STORE_META = "meta"; // {key, value}

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_DOCS)) {
        db.createObjectStore(STORE_DOCS, { keyPath: "id" });
      }
      if (!db.objectStoreNames.contains(STORE_META)) {
        db.createObjectStore(STORE_META, { keyPath: "key" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx(db, store, mode = "readonly") {
  return db.transaction(store, mode).objectStore(store);
}

function awaitReq(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function saveDoc(doc) {
  const db = await openDb();
  await awaitReq(tx(db, STORE_DOCS, "readwrite").put(doc));
  db.close();
}

export async function deleteDoc(id) {
  const db = await openDb();
  await awaitReq(tx(db, STORE_DOCS, "readwrite").delete(id));
  db.close();
}

export async function listDocs() {
  const db = await openDb();
  const all = await awaitReq(tx(db, STORE_DOCS).getAll());
  db.close();
  return all;
}

export async function setMeta(key, value) {
  const db = await openDb();
  await awaitReq(tx(db, STORE_META, "readwrite").put({ key, value }));
  db.close();
}

export async function getMeta(key) {
  const db = await openDb();
  const v = await awaitReq(tx(db, STORE_META).get(key));
  db.close();
  return v?.value;
}
