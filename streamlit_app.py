import os
import json
import math
from typing import List, Tuple, Dict, Any

import streamlit as st
import numpy as np

try:
    # pypdf ist leichtgewichtig und robust für Text-PDFs
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import tiktoken  # optional für bessere Chunkgrößen
except Exception:
    tiktoken = None

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
from openai import APIError, RateLimitError


# ---------- Seite konfigurieren ----------
st.set_page_config(
    page_title="GPT-5 Chatbot (Streamlit)",
    page_icon="🤖",
    layout="wide"
)

# ---------- Hilfsfunktionen ----------
def get_api_key() -> str:
    # Reihenfolge: Sidebar-Eingabe > Secrets > Env
    key = st.session_state.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    return key.strip()

def ensure_client() -> OpenAI:
    api_key = get_api_key()
    if not api_key:
        st.warning("Bitte OPENAI_API_KEY in der Sidebar setzen (oder als Secret/ENV).")
    return OpenAI(api_key=api_key) if api_key else None

def read_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")

def read_pdf(file) -> str:
    if PdfReader is None:
        st.error("pypdf ist nicht installiert – PDF-Import nicht möglich.")
        return ""
    try:
        reader = PdfReader(file)
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n".join(texts)
    except Exception as e:
        st.error(f"PDF konnte nicht gelesen werden: {e}")
        return ""

def estimate_tokens(s: str, model: str = "gpt-5") -> int:
    # sehr grobe Schätzung, falls tiktoken fehlt
    if tiktoken is None:
        return max(1, len(s) // 4)
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(s))
    except Exception:
        return max(1, len(s) // 4)

def chunk_text(
    text: str,
    chunk_tok: int = 800,
    overlap_tok: int = 120,
    hard_max_chunks: int = 800
) -> List[str]:
    """Teilt Text in überlappende Token-Chunks (oder approx. chars)."""
    if not text.strip():
        return []

    if tiktoken is None:
        # Fallback: char-basierte Stückelung (4 chars ~ 1 Token grob)
        approx = chunk_tok * 4
        step = max(1, approx - overlap_tok * 4)
        chunks = [text[i:i+approx] for i in range(0, min(len(text), approx * hard_max_chunks), step)]
        return [c.strip() for c in chunks if c.strip()]

    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    out = []
    i = 0
    limit = min(len(toks), chunk_tok * hard_max_chunks)
    while i < limit:
        window = toks[i:i+chunk_tok]
        out.append(enc.decode(window))
        i += (chunk_tok - overlap_tok)
    return [c.strip() for c in out if c.strip()]

@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((APIError, RateLimitError))
)
def embed_texts(client: OpenAI, model: str, texts: List[str]) -> np.ndarray:
    """Erzeugt Embeddings für eine Liste von Texten, batching für große Mengen."""
    vectors: List[List[float]] = []
    batch_size = 96
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend([item.embedding for item in resp.data])
    return np.array(vectors, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (n, d), b: (d,)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b) + 1e-9)
    return np.dot(a_norm, b_norm)

def build_context(
    query: str,
    kb: Dict[str, Any],
    top_k: int,
    max_context_tokens: int
) -> Tuple[str, List[Dict[str, Any]]]:
    """Wählt Top-k Chunks via Cosine-Similarity und baut einen Kontextblock."""
    if not kb or not kb.get("vectors") is not None:
        return "", []

    client = ensure_client()
    if client is None:
        return "", []

    query_vec = embed_texts(client, kb["embedding_model"], [query])[0]
    sims = cosine_sim(kb["vectors"], query_vec)
    idx = np.argsort(-sims)[:top_k]

    selected = []
    total_tokens = 0
    for i in idx:
        ch = kb["chunks"][i]
        src = kb["sources"][i]
        toks = estimate_tokens(ch)
        if total_tokens + toks > max_context_tokens:
            continue
        total_tokens += toks
        selected.append({"text": ch, "source": src, "score": float(sims[i])})

    # Klarer, komprimierter Kontext
    context_block = "\n\n".join(
        [f"[{i+1}] ({s['source']})\n{s['text']}" for i, s in enumerate(selected)]
    )
    return context_block, selected

@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((APIError, RateLimitError))
)
def chat_complete(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.5,
    max_tokens: int = 1024
) -> str:
    """Einfacher Wrapper um Chat Completions."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

def ensure_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("kb", None)

ensure_state()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Einstellungen")

    api_key_input = st.text_input("OPENAI_API_KEY", type="password", value=st.session_state.get("OPENAI_API_KEY", ""))
    if api_key_input:
        st.session_state["OPENAI_API_KEY"] = api_key_input

    model = st.selectbox(
        "Chat-Modell",
        options=["gpt-5", "gpt-4o", "gpt-4.1-mini"],
        index=0
    )
    temperature = st.slider("Temperatur", 0.0, 1.2, 0.4, 0.05)
    system_role = st.text_area(
        "System-Rolle",
        value="Du bist ein hilfreicher, präziser Assistent. Antworte kurz, klar und nenne Annahmen explizit.",
        height=100
    )
    max_answer_tokens = st.slider("Max Tokens pro Antwort", 256, 4096, 1024, 64)

    st.markdown("---")
    st.subheader("📚 Wissensbasis (RAG)")
    embedding_model = st.selectbox(
        "Embedding-Modell",
        options=["text-embedding-3-large"],
        index=0,
        help="Hochwertiges Embedding-Modell (OpenAI Docs)."
    )
    top_k = st.slider("Top-k Chunks", 1, 12, 5)
    max_ctx_tokens = st.slider("Max Kontext-Tokens", 256, 8000, 2400, 64)

    files = st.file_uploader(
        "Dateien hinzufügen (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    col_kb1, col_kb2 = st.columns(2)
    with col_kb1:
        build_btn = st.button("🔧 Wissensbasis aufbauen/aktualisieren", use_container_width=True)
    with col_kb2:
        reset_kb = st.button("♻️ Wissensbasis leeren", use_container_width=True)

    if reset_kb:
        st.session_state["kb"] = None
        st.success("Wissensbasis zurückgesetzt.")

    if build_btn and files:
        client = ensure_client()
        if client:
            texts, sources = [], []
            for f in files:
                name = f.name
                if name.lower().endswith(".pdf"):
                    txt = read_pdf(f)
                else:
                    txt = read_txt(f)
                if not txt.strip():
                    continue
                chunks = chunk_text(txt, chunk_tok=800, overlap_tok=120)
                texts.extend(chunks)
                sources.extend([name] * len(chunks))

            if not texts:
                st.warning("Keine extrahierbaren Inhalte gefunden.")
            else:
                with st.spinner(f"Erzeuge Embeddings für {len(texts)} Chunks …"):
                    try:
                        vectors = embed_texts(client, embedding_model, texts)
                        st.session_state["kb"] = {
                            "embedding_model": embedding_model,
                            "chunks": texts,
                            "sources": sources,
                            "vectors": vectors
                        }
                        st.success(f"Wissensbasis bereit: {len(texts)} Chunks.")
                    except Exception as e:
                        st.error(f"Fehler beim Embedding: {e}")

    if st.session_state.get("kb"):
        kb = st.session_state["kb"]
        st.info(f"KB aktiv – {len(kb['chunks'])} Chunks aus {len(set(kb['sources']))} Datei(en).")
        # Download der KB als JSON (nur Metadaten + Chunks; Vektoren ausgelassen, um Datei klein zu halten)
        kb_dl = {
            "embedding_model": kb["embedding_model"],
            "chunks": kb["chunks"],
            "sources": kb["sources"]
        }
        st.download_button(
            "⬇️ Wissensbasis (ohne Vektoren) exportieren",
            data=json.dumps(kb_dl, ensure_ascii=False).encode("utf-8"),
            file_name="knowledge_base.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("🧹 Verlauf")
    if st.button("Konversation leeren", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    if st.session_state["messages"]:
        chat_json = json.dumps(st.session_state["messages"], ensure_ascii=False, indent=2)
        st.download_button(
            "⬇️ Chatverlauf exportieren (JSON)",
            data=chat_json.encode("utf-8"),
            file_name="chat_history.json",
            mime="application/json",
            use_container_width=True
        )


# ---------- Hauptbereich ----------
st.title("🤖 GPT-5 Chatbot (Streamlit)")

# Anzeige bestehender Nachrichten
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat-Eingabe
prompt = st.chat_input("Frage stellen oder Nachricht eingeben …")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Kontext aus KB aufbereiten (falls vorhanden)
    kb = st.session_state.get("kb")
    context_block = ""
    selected_docs = []

    if kb:
        with st.spinner("🔎 Suche relevante Passagen …"):
            context_block, selected_docs = build_context(
                query=prompt,
                kb=kb,
                top_k=top_k,
                max_context_tokens=max_ctx_tokens
            )

    # Nachrichten für das LLM zusammenbauen
    llm_messages = [{"role": "system", "content": system_role}]
    # optionaler Kontext – als eigene Systeminstruktion
    if context_block:
        llm_messages.append({
            "role": "system",
            "content": (
                "Nutze die folgenden Auszüge aus meiner Wissensbasis, "
                "wenn sie relevant sind. Zitiere prägnant und vermeide Widersprüche.\n\n"
                f"{context_block}"
            )
        })
    # bisherige Unterhaltung
    for m in st.session_state["messages"]:
        if m["role"] in ("user", "assistant"):
            llm_messages.append(m)

    # Antwort generieren
    with st.chat_message("assistant"):
        client = ensure_client()
        if client is None:
            st.stop()
        try:
            try_model = model
            try:
                answer = chat_complete(
                    client=client,
                    model=try_model,
                    messages=llm_messages,
                    temperature=temperature,
                    max_tokens=max_answer_tokens
                )
            except Exception as primary_err:
                # Fallback, falls z. B. gpt-5 noch nicht verfügbar ist
                if try_model != "gpt-4o":
                    answer = chat_complete(
                        client=client,
                        model="gpt-4o",
                        messages=llm_messages,
                        temperature=temperature,
                        max_tokens=max_answer_tokens
                    )
                else:
                    raise primary_err
            st.markdown(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})

            # Quellenhinweise aus RAG anzeigen
            if selected_docs:
                with st.expander("🔎 verwendete KB-Quellen"):
                    for i, s in enumerate(selected_docs, 1):
                        st.markdown(f"**[{i}]** {s['source']}  —  Score: {s['score']:.3f}")

        except Exception as e:
            st.error(f"Fehler vom Modell: {e}")
