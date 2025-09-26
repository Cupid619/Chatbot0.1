import os, io, json, time, hashlib, math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import numpy as np

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import tiktoken
except Exception:
    tiktoken = None

# --- OpenAI SDK v1 ---
from openai import OpenAI
from openai import APIError, RateLimitError

# --- optional: S3 Backend ---
USE_S3_DEFAULT = False
try:
    import boto3
except Exception:
    boto3 = None


# =========================
# Utility: Hash/Token/Chunk
# =========================
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def estimate_tokens(s: str) -> int:
    if tiktoken is None:
        return max(1, len(s) // 4)  # grob
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(s))
    except Exception:
        return max(1, len(s) // 4)

def chunk_text(text: str, chunk_tok: int = 800, overlap_tok: int = 120, hard_max_chunks: int = 2000) -> List[str]:
    if not text.strip():
        return []
    if tiktoken is None:
        approx = chunk_tok * 4
        step = max(1, approx - overlap_tok * 4)
        max_len = min(len(text), approx * hard_max_chunks)
        return [text[i:i+approx].strip() for i in range(0, max_len, step) if text[i:i+approx].strip()]
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    out = []
    i = 0
    limit = min(len(toks), chunk_tok * hard_max_chunks)
    while i < limit:
        window = toks[i:i+chunk_tok]
        out.append(enc.decode(window).strip())
        i += (chunk_tok - overlap_tok)
    return [c for c in out if c]


# =========================
# Storage Backends (KB)
# =========================
class LocalFSStorage:
    def __init__(self, root: str = "kb_store"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "docs").mkdir(exist_ok=True)

    def put_file(self, key: str, data: bytes):
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get_file(self, key: str) -> Optional[bytes]:
        p = self.root / key
        return p.read_bytes() if p.exists() else None

    def list_prefix(self, prefix: str) -> List[str]:
        base = self.root / prefix
        if not base.exists():
            return []
        return [str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file()]

class S3Storage:
    def __init__(self, bucket: str, region: str, access_key: str, secret_key: str):
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def put_file(self, key: str, data: bytes):
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get_file(self, key: str) -> Optional[bytes]:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
            return obj["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None
        except Exception:
            return None

    def list_prefix(self, prefix: str) -> List[str]:
        keys = []
        continuation = None
        while True:
            kw = dict(Bucket=self.bucket, Prefix=prefix)
            if continuation:
                kw["ContinuationToken"] = continuation
            resp = self.client.list_objects_v2(**kw)
            for it in resp.get("Contents", []):
                keys.append(it["Key"])
            if not resp.get("IsTruncated"):
                break
            continuation = resp.get("NextContinuationToken")
        return keys


# =========================
# Embeddings / Retrieval
# =========================
@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((APIError, RateLimitError))
)
def embed_texts(client: OpenAI, model: str, texts: List[str]) -> np.ndarray:
    vecs: List[List[float]] = []
    batch = 96
    for i in range(0, len(texts), batch):
        part = texts[i:i+batch]
        resp = client.embeddings.create(model=model, input=part)  # OpenAI Embeddings :contentReference[oaicite:5]{index=5}
        vecs.extend([d.embedding for d in resp.data])
    return np.array(vecs, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b) + 1e-9)
    return a_norm @ b_norm

def build_context(query: str, kb: Dict[str, Any], client: OpenAI, top_k: int, max_ctx_tokens: int) -> Tuple[str, List[Dict[str, Any]]]:
    if not kb or "vectors" not in kb or kb["vectors"] is None or len(kb["chunks"]) == 0:
        return "", []
    qv = embed_texts(client, kb["embedding_model"], [query])[0]
    sims = cosine_sim(kb["vectors"], qv)
    idx = np.argsort(-sims)[:top_k]

    selected, total_tokens = [], 0
    for i in idx:
        ch = kb["chunks"][i]
        src = kb["sources"][i]
        toks = estimate_tokens(ch)
        if total_tokens + toks > max_ctx_tokens:
            continue
        total_tokens += toks
        selected.append({"text": ch, "source": src, "score": float(sims[i])})

    block = "\n\n".join([f"[{i+1}] ({s['source']})\n{s['text']}" for i, s in enumerate(selected)])
    return block, selected


# =========================
# Persistente Wissensbasis
# =========================
class KBStore:
    """
    Speichert Dokument-Chunks + Embeddings persistent.
    Backend: LocalFS (kb_store/) oder S3 (Bucket).
    Struktur:
      - index.json  -> Liste bekannter Dokumente (doc_id, name, ts)
      - docs/{doc_id}/meta.json
      - docs/{doc_id}/chunks.json
      - docs/{doc_id}/vectors.npy
    """
    def __init__(self, backend):
        self.backend = backend
        self.index_key = "index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        raw = self.backend.get_file(self.index_key)
        if not raw:
            return {"docs": []}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {"docs": []}

    def _save_index(self):
        self.backend.put_file(self.index_key, json.dumps(self.index, ensure_ascii=False).encode("utf-8"))

    def add_document(self, name: str, chunks: List[str], sources: List[str], vectors: np.ndarray, embedding_model: str) -> str:
        doc_bytes = (name + "".join(chunks)).encode("utf-8", "ignore")
        doc_id = sha256_bytes(doc_bytes)[:16]

        meta = {"name": name, "embedding_model": embedding_model, "ts": time.time(), "n_chunks": len(chunks)}
        self.backend.put_file(f"docs/{doc_id}/meta.json", json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        self.backend.put_file(f"docs/{doc_id}/chunks.json", json.dumps({"chunks": chunks, "sources": sources}, ensure_ascii=False).encode("utf-8"))
        # Save vectors
        bio = io.BytesIO()
        np.save(bio, vectors)
        self.backend.put_file(f"docs/{doc_id}/vectors.npy", bio.getvalue())

        # update index
        if not any(d["doc_id"] == doc_id for d in self.index["docs"]):
            self.index["docs"].append({"doc_id": doc_id, "name": name, "ts": meta["ts"]})
            self._save_index()
        return doc_id

    def load_all(self) -> Dict[str, Any]:
        # Aggregiere alle Docs in ein gemeinsames KB-Objekt
        chunks_all, sources_all, vecs_list = [], [], []
        embed_model = None
        for d in self.index["docs"]:
            doc_id = d["doc_id"]
            meta_raw = self.backend.get_file(f"docs/{doc_id}/meta.json")
            ch_raw = self.backend.get_file(f"docs/{doc_id}/chunks.json")
            v_raw = self.backend.get_file(f"docs/{doc_id}/vectors.npy")
            if not (meta_raw and ch_raw and v_raw):
                continue
            meta = json.loads(meta_raw.decode("utf-8"))
            ch = json.loads(ch_raw.decode("utf-8"))
            bio = io.BytesIO(v_raw)
            vecs = np.load(bio)
            chunks_all.extend(ch["chunks"])
            sources_all.extend(ch["sources"])
            vecs_list.append(vecs)
            embed_model = meta.get("embedding_model")
        vectors = np.vstack(vecs_list) if vecs_list else np.zeros((0, 1536), dtype=np.float32)
        return {
            "embedding_model": embed_model or "text-embedding-3-large",
            "chunks": chunks_all,
            "sources": sources_all,
            "vectors": vectors
        }


# =========================
# OpenAI Chat Wrapper
# =========================
@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((APIError, RateLimitError))
)
def chat_complete(client: OpenAI, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    # Chat Completions API – offizielles Muster :contentReference[oaicite:6]{index=6}
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="PostBot – Schweizerische Post", page_icon="📮", layout="wide")
st.title("📮 PostBot – Schweizerische Post (GPT-5)")

# --- Sidebar: Einstellungen & Persistenz ---
with st.sidebar:
    st.header("⚙️ Einstellungen")

    # API Key: Sidebar > Secrets > Env
    api_in = st.text_input("OPENAI_API_KEY", type="password", value=st.session_state.get("OPENAI_API_KEY", ""))
    if api_in:
        st.session_state["OPENAI_API_KEY"] = api_in

    def get_api_key() -> str:
        return (st.session_state.get("OPENAI_API_KEY")
                or st.secrets.get("OPENAI_API_KEY", "")
                or os.environ.get("OPENAI_API_KEY", "")).strip()

    model = st.selectbox("Chat-Modell", ["gpt-5", "gpt-4o"], index=0)
    temperature = st.slider("Temperatur", 0.0, 1.2, 0.35, 0.05)
    max_answer_tokens = st.slider("Max Tokens/Antwort", 256, 4096, 1024, 64)

    st.markdown("---")
    st.subheader("📚 Wissensbasis (RAG) – Persistenz")

    use_s3 = st.toggle("S3-Persistenz aktivieren (für Cloud empfohlen)", value=USE_S3_DEFAULT and (boto3 is not None))
    storage = None
    if use_s3 and boto3 is not None and all(k in st.secrets for k in ["AWS_S3_BUCKET", "AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]):
        storage = S3Storage(
            bucket=st.secrets["AWS_S3_BUCKET"],
            region=st.secrets["AWS_REGION"],
            access_key=st.secrets["AWS_ACCESS_KEY_ID"],
            secret_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        )
        st.success("S3-Persistenz aktiv")
    else:
        storage = LocalFSStorage("kb_store")
        if use_s3 and boto3 is None:
            st.warning("boto3 nicht installiert – fallback auf lokalen Speicher.")
        elif use_s3:
            st.warning("S3-Secrets fehlen – fallback auf lokalen Speicher.")

    embedding_model = st.selectbox("Embedding-Modell", ["text-embedding-3-large"], help="Hochwertige Embeddings für Suche/RAG. :contentReference[oaicite:7]{index=7}")
    top_k = st.slider("Top-k Chunks", 1, 12, 5)
    max_ctx_tokens = st.slider("Max Kontext-Tokens (RAG)", 256, 8000, 3000, 64)

    # Upload
    files = st.file_uploader("Dateien laden (PDF, TXT, MD)", type=["pdf", "txt", "md"], accept_multiple_files=True)
    build_btn = st.button("🔧 Chunks & Embeddings persistent speichern", use_container_width=True)

    # Rolle speichern
    st.markdown("---")
    st.subheader("🧠 Rolle")
    default_role = ("Ich bin ein Verkaufs- und Beratungstalent der Schweizerischen Post. "
                    "Meine Antworten sind klar und kompakt. Meine Antworten haben, anhand der beigefügten Dokumente, "
                    "eine sehr hohe Fachkompetenz. Nur wenn der Kunde etwas abschliessen möchte, leite ich ihn an leitung.bahnhof@post.ch weiter. "
                    "Wenn Informationen fehlen, sage ich es offen und frage nach.")
    role_key = "role.text"
    role_text = st.text_area("System-Rolle", value=st.session_state.get(role_key, default_role), height=140)
    if st.button("Rolle übernehmen", use_container_width=True):
        st.session_state[role_key] = role_text
        st.success("Rolle gesetzt.")

    st.markdown("---")
    st.subheader("🧹 Verlauf/Export")
    if st.button("Konversation leeren", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()
    if st.session_state.get("messages"):
        st.download_button("⬇️ Chatverlauf (JSON)", data=json.dumps(st.session_state["messages"], ensure_ascii=False).encode("utf-8"),
                           file_name="chat_history.json", use_container_width=True)

# --- Client erstellen ---
def ensure_client() -> Optional[OpenAI]:
    key = get_api_key()
    if not key:
        st.warning("Bitte OPENAI_API_KEY in der Sidebar/Secrets/ENV setzen.")
        return None
    return OpenAI(api_key=key)

# --- Wissensbasis laden (bestehende persistente Daten aggregieren) ---
kb_store = KBStore(storage)
kb_cached = kb_store.load_all()

# --- Upload & Persistenz-Aufbau ---
if build_btn and files:
    client = ensure_client()
    if client:
        texts, sources = [], []
        for f in files:
            name = f.name
            if name.lower().endswith(".pdf"):
                if PdfReader is None:
                    st.error("pypdf fehlt – PDF wird übersprungen.")
                    continue
                try:
                    pdf = PdfReader(f)
                    pages = []
                    for p in pdf.pages:
                        try:
                            pages.append(p.extract_text() or "")
                        except Exception:
                            pages.append("")
                    txt = "\n".join(pages)
                except Exception as e:
                    st.error(f"PDF-Fehler ({name}): {e}")
                    continue
            else:
                txt = f.read().decode("utf-8", errors="ignore")

            if not txt.strip():
                st.warning(f"Keine extrahierbaren Inhalte in {name}.")
                continue

            chunks = chunk_text(txt, chunk_tok=800, overlap_tok=120)
            texts.extend(chunks)
            sources.extend([name] * len(chunks))

        if not texts:
            st.warning("Keine Chunks erzeugt.")
        else:
            with st.spinner(f"Erzeuge Embeddings für {len(texts)} Chunks …"):
                try:
                    vecs = embed_texts(client, embedding_model, texts)
                    doc_id = kb_store.add_document(name="upload_batch", chunks=texts, sources=sources, vectors=vecs, embedding_model=embedding_model)
                    st.success(f"Wissensbasis aktualisiert (doc_id={doc_id}).")
                    kb_cached = kb_store.load_all()  # neu laden
                except Exception as e:
                    st.error(f"Embedding-Fehler: {e}")

# --- State init ---
st.session_state.setdefault("messages", [])

# --- Vorhandene Messages rendern ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat ---
prompt = st.chat_input("Ihre Frage …")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = ensure_client()
    if client is None:
        st.stop()

    # RAG-Kontext bauen
    context_block, selected_docs = "", []
    if kb_cached and len(kb_cached.get("chunks", [])) > 0:
        with st.spinner("🔎 Suche relevante Passagen …"):
            context_block, selected_docs = build_context(
                query=prompt,
                kb=kb_cached,
                client=client,
                top_k=top_k,
                max_ctx_tokens=max_ctx_tokens
            )

    # Nachrichten für LLM
    role_msg = st.session_state.get(role_key, default_role)
    llm_messages = [{"role": "system", "content": role_msg}]

    if context_block:
        llm_messages.append({
            "role": "system",
            "content": (
                "Nutze ausschließlich die folgenden geprüften Auszüge aus der Wissensbasis, "
                "wenn sie relevant sind. Zitiere prägnant und erfinde keine Quellen.\n\n"
                f"{context_block}"
            )
        })

    # bisherige Unterhaltung (ohne System/KB)
    for m in st.session_state["messages"]:
        if m["role"] in ("user", "assistant"):
            llm_messages.append(m)

    with st.chat_message("assistant"):
        try:
            try_model = model
            answer = chat_complete(client, model=try_model, messages=llm_messages, temperature=temperature, max_tokens=max_answer_tokens)
        except Exception:
            # Fallback auf gpt-4o, falls gpt-5 nicht verfügbar ist
            answer = chat_complete(client, model="gpt-4o", messages=llm_messages, temperature=temperature, max_tokens=max_answer_tokens)

        st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})

        # Quellen anzeigen
        if selected_docs:
            with st.expander("🔎 verwendete KB-Quellen"):
                for i, s in enumerate(selected_docs, 1):
                    st.markdown(f"**[{i}]** {s['source']}  —  Score: {s['score']:.3f}")

        # Feedback-Learning
        st.markdown("---")
        st.write("**War die Antwort korrekt?**")
        col1, col2 = st.columns(2)
        with col1:
            good = st.button("👍 Ja")
        with col2:
            bad = st.button("👎 Nein")
        correction = st.text_area("Korrektur / richtige Antwort inkl. Quelle(n):", height=120, key=f"corr_{len(st.session_state['messages'])}")

        if good or bad:
            fb = {
                "ts": time.time(),
                "label": "good" if good else "bad",
                "prompt": prompt,
                "answer": answer,
                "correction": correction.strip() or None
            }
            storage.put_file(f"feedback/{int(fb['ts'])}.json", json.dumps(fb, ensure_ascii=False).encode("utf-8"))
            st.success("Feedback gespeichert.")
            # Hotfix: Negative Korrektur als Wissensnotiz in die KB einspielen
            if bad and fb["correction"]:
                # neue Notiz als Mini-Dokument persistieren
                note_chunks = chunk_text(fb["correction"], chunk_tok=400, overlap_tok=40)
                if note_chunks:
                    vecs = embed_texts(client, kb_cached["embedding_model"], note_chunks)
                    kb_store.add_document(name="user_corrections", chunks=note_chunks,
                                          sources=["user_corrections"] * len(note_chunks),
                                          vectors=vecs, embedding_model=kb_cached["embedding_model"])
                    kb_cached = kb_store.load_all()
                    st.info("Korrektur in Wissensbasis aufgenommen.")
