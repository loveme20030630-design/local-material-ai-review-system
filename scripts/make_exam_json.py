from __future__ import annotations

"""
用法：
  python scripts\make_exam_json.py <SUBJECT> <ppt|pdf> <CHAPTER> --n 10
"""

import argparse
import json
import os
import random
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# -------- paths --------
THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parent
DATA_DIR = BASE_DIR / "data"
ART_DIR = BASE_DIR / "artifacts" / "exams"

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("data_type", choices=["ppt", "pdf"])
    ap.add_argument("chapter")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max_ctx_chars", type=int, default=6000)
    ap.add_argument("--ctx_per_doc_chars", type=int, default=1200)
    ap.add_argument("--lang", choices=["cn", "en", "auto"], default="auto")
    ap.add_argument("--strict_quote", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ollama_model", default=os.environ.get("OLLAMA_MODEL", "").strip())
    ap.add_argument("--ollama_base_url", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").strip() or "http://localhost:11434")
    return ap.parse_args()

def resolve_exam_corpus(subject: str, data_type: str) -> Path:
    return DATA_DIR / subject / ("pdf_exam_corpus" if data_type == "pdf" else "ppt_exam_corpus")


def fetch_ollama_tags(base_url: str) -> list[str]:
    """讀取本機 Ollama 已安裝模型清單。"""
    url = base_url.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"}, method="GET")
    try:
        timeout_seconds = int(os.environ.get("LLM_API_TIMEOUT", "300").strip())
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"failed to query Ollama tags from {url}: {e}")

    models = data.get("models", [])
    if not isinstance(models, list):
        return []

    out: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or item.get("model", "")).strip()
        if name:
            out.append(name)
    return out

def resolve_ollama_model(base_url: str, preferred: str, allow_auto_scan: bool = True) -> str:
    """
    模型選擇策略：
      1) 有明確指定 preferred：若存在則使用；不存在直接報錯
      2) 未指定：掃描 /api/tags，按白名單順序挑選
    """
    preferred = (preferred or "").strip()
    tags = fetch_ollama_tags(base_url)

    if preferred:
        if preferred in tags:
            print(f"[LLM] using Ollama model: {preferred}")
            return preferred
        avail = ", ".join(tags) if tags else "<none>"
        raise RuntimeError(
            f"requested Ollama model not found: {preferred}. available models: {avail}"
        )

    if not allow_auto_scan:
        raise RuntimeError("OLLAMA_MODEL is empty and auto scan is disabled")

    preferred_order = [
        "qwen2:7b",
        "llama3.1:8b",
        "qwen2",
        "llama3.1",
    ]
    for name in preferred_order:
        if name in tags:
            print(f"[LLM] auto-selected Ollama model: {name}")
            return name

    if tags:
        print(f"[LLM] auto-selected fallback Ollama model: {tags[0]}")
        return tags[0]

    raise RuntimeError("no Ollama models found via /api/tags")

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

class QueryEncoder:
    def __init__(self, model_name: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)
            last = out.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu().numpy().astype(np.float32)

def load_index(index_dir: Path) -> tuple[faiss.Index, list[dict[str, Any]], list[str], dict[str, Any]]:

    if not index_dir.is_dir():
        raise FileNotFoundError(f"index dir not found: {index_dir}")
    index = faiss.read_index(str(index_dir / "vectors.faiss"))
    pages = json.loads((index_dir / "pages.json").read_text(encoding="utf-8"))
    texts = json.loads((index_dir / "texts.json").read_text(encoding="utf-8"))
    info = json.loads((index_dir / "index.info.json").read_text(encoding="utf-8"))
    n = index.ntotal
    if len(pages) != n or len(texts) != n:
        raise RuntimeError(f"index size mismatch: ntotal={n}, pages={len(pages)}, texts={len(texts)}")
    if "model" not in info:
        raise RuntimeError("index.info.json missing key: 'model'")
    return index, pages, texts, info

def search_topk(
    index: faiss.Index,
    pages: list[dict[str, Any]],
    texts: list[str],
    encoder: QueryEncoder,
    query: str,
    k: int,
) -> list[tuple[float, dict[str, Any], str]]:
    """向量檢索 topK（k 不是關鍵字數）。"""
    qv = encoder.encode(query)
    scores, ids = index.search(qv, k)

    out: list[tuple[float, dict[str, Any], str]] = []
    for idx, score in zip(ids[0], scores[0]):
        if int(idx) < 0:
            continue
        out.append((float(score), pages[int(idx)], texts[int(idx)]))
    return out

def build_topic_pool_from_texts(texts: list[str], limit: int = 200) -> list[str]:
    """
    從 texts.json 抽出適合拿來檢索的 topic：
    - 優先抓章節/小節標題
    - 再抓內容較完整的首句
    - 避免「這頁是沒有txt的」與純章號
    """
    topics: list[str] = []
    seen = set()

    for t in texts:
        s = (t or "").strip()
        if not s:
            continue
        if s == "這頁是沒有txt的":
            continue

        first_line = s.splitlines()[0].strip()
        first_line = re.sub(r"\s+", " ", first_line)

        # 跳過太短、太像純章號/純目錄的內容
        if len(first_line) < 4:
            continue
        if re.fullmatch(r"CH\d+\s*[　 ].*", first_line):
            continue
        if re.fullmatch(r"\d+[-－]\d+.*", first_line) or re.fullmatch(r"\d+-\d+-\d+.*", first_line):
            topic = first_line
        else:
            # 若第一行不是標題，就取前 40~80 字當 topic
            topic = s[:60].strip()


        topic = re.sub(r"\s+", " ", topic)

        # 太短、太像殘句、太像純目錄就跳過
        if len(topic) < 12:
            continue
        if topic.count(" ") < 1 and len(topic) < 20:
            continue
        if re.fullmatch(r"[0-9A-Za-z\-\.\(\) ]+", topic):
            continue

        if topic not in seen:
            seen.add(topic)
            topics.append(topic)

        if len(topics) >= limit:
            break

    return topics
 
def _normalize_source_images(page: dict[str, Any]) -> list[str]:
    """
    將 page 裡可能出現的 image / images 統一成 list[str]。
    目的：讓 exam JSON 的 sources_map 穩定提供圖片路徑給前端。
    """
    out: list[str] = []

    def add(v: Any) -> None:
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
            return

        if isinstance(v, list):
            for item in v:
                add(item)
            return

        if isinstance(v, dict):
            for key in ("rel_path", "path", "file_name", "name", "image", "url", "image_url"):
                add(v.get(key))
            add(v.get("images"))

    add(page.get("image"))
    add(page.get("images"))

    seen: set[str] = set()
    clean: list[str] = []
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        clean.append(item)

    return clean


def build_context(
    results: list[tuple[float, dict[str, Any], str]],
    max_ctx_chars: int,
    per_doc_chars: int,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """
    回傳：
      ctx: 給 LLM 的 CONTEXT（含 [1]...[k]）
      sources_map: {"[1]": {"doc_id":..., "page":..., "snippet":...}, ...}
    """
    chunks: list[str] = []
    sources_map: dict[str, dict[str, Any]] = {}
    used = 0

    for rank, (score, page, text) in enumerate(results, 1):
        raw_doc = page.get("doc_id", "unknown")
        doc_id = "" if raw_doc is None else str(raw_doc).strip()
        if not doc_id or doc_id.lower() == "none":
            doc_id = "unknown"
        pno = page.get("page", None)
        ocr = page.get("ocr", "") or ""
        images = _normalize_source_images(page)

        blob = (ocr + "\n" + (text or "")).strip()
        blob = re.sub(r"\s+", " ", blob)  # 重要：避免換行污染
        snippet = blob[: max(0, per_doc_chars)]
        tag = f"[{rank}]"

        sources_map[tag] = {
            "doc_id": doc_id,
            "page": pno,
            "image": images[0] if images else "",
            "images": images,
            "snippet": snippet,
            "score": float(score),
        }

        piece = f"{tag} {doc_id}/p{pno} score={score:.4f}\n{snippet}\n"
        if used +  len(piece) > max_ctx_chars:
            break

        chunks.append(piece)
        used += len(piece)
    return "\n".join(chunks).strip(), sources_map

def llm_post_json(url: str, payload: dict[str, Any], api_key: str = "") -> dict[str, Any]:
    """HTTP POST JSON → dict（Ollama / OpenAI 共用）。"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        timeout_seconds = int(os.environ.get("LLM_API_TIMEOUT", "300").strip())
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        info = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(f"LLM API request failed: {e.code} {e.reason}, details: {info}")

    return json.loads(raw)

def call_ollama_chat(model: str, base_url: str, system: str, user: str) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 750},
    }
    data = llm_post_json(url, payload, "")
    return (data.get("message", {}).get("content", "") or "").strip()

def parse_first_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    blob = m.group(0).strip()
    try:
        return json.loads(blob)
    except Exception:
        return None
    
MCQ_SCHEMA_HINT = (
  '{'
  '"question":"...",'
  '"options":{"A":"(plain text)","B":"(plain text)","C":"(plain text)","D":"(plain text)"},'
  '"answer":"A",'
  '"explanation":[{"text":"...", "sources":["[1]"]}],'
  '"sources_used":["[1]","[3]"]'
  '}'
)

def _bad_mcq(obj: dict[str, Any]) -> bool:
    q = str(obj.get("question",""))
    opts = obj.get("options", {}) or {}
    ans = str(obj.get("answer","")).strip()

    # 題幹不得包含選項字樣
    if re.search(r"\n\s*[ABCD]\)", q) or re.search(r"\n\s*[ABCD]\.", q):
        return True

    # 必須四個選項都有
    for k in ["A","B","C","D"]:
        v = str(opts.get(k,"")).strip()
        if not v:
            return True
        # 禁止 NOT_FOUND / 來源tag 當選項
        if v == "NOT_FOUND" or re.fullmatch(r"\[\d+\]", v):
            return True

    # answer 必須合法
    if ans not in ["A","B","C","D"]:
        return True
    
    if re.search(r"\b[ABCD][\)\.]\s", q):
        return True

    # explanation 必須有 sources
    exp = obj.get("explanation", [])
    if not isinstance(exp, list) or len(exp) == 0:
        return True
    for it in exp:
        ss = it.get("sources", [])
        if not isinstance(ss, list) or len(ss) == 0:
            return True
    return False

def _has_verbatim_quote_in_ctx(obj: dict[str, Any], ctx: str) -> bool:
    exp = obj.get("explanation", [])
    if not isinstance(exp, list) or not exp:
        return False
    for it in exp:
        text = str(it.get("text", ""))
        #抓...(引文)
        quotes = re.findall(r"\"([^\"]{3,120})\"", text)
        for q in quotes:
            if q in ctx:
                return True
    return False

def generate_mcq_from_context(
    topic: str,
    ctx: str,
    sources_map: dict[str, dict[str, Any]],
    ollama_model: str,
    ollama_base_url: str,
    lang: str,
    strict_quote: bool,
    max_retry: int = 2,
) -> dict[str, Any] | None:
    is_cn = (lang == "cn") or (lang == "auto" and bool(re.search(r"[\u4e00-\u9fff]", topic + " " + ctx[:500])))

    if is_cn:
        system = (
            "你是出題助手。你只能根據 CONTEXT 出題，不得外推或補充常識。"
            "請產生 1 題四選一選擇題（A/B/C/D），只能有(A/B/C/D）四個選項且只能有 1 個正確答案。"
            "解析必須是條列，每一點都要附上 sources（只能用 [1]...[k] 這些標記）。"
            "如果 CONTEXT 不足以出題，輸出：NOT_FOUND。"
            "解析每一點都必須包含至少一段「直接引文」，用雙引號包起來（例如 \"...\"），且該引文必須逐字出現在 CONTEXT。做不到就輸出 NOT_FOUND。"
            "只輸出 JSON，格式必須符合："
            + MCQ_SCHEMA_HINT
        )
        user = f"主題：{topic}\n\nCONTEXT:\n{ctx}"
    else:
        system = (
            "You are an exam question writer. Use ONLY the provided CONTEXT. "
            "Do NOT add any background knowledge. "
            "Create exactly ONE 4-option multiple-choice question (A/B/C/D) with exactly ONE correct option. "
            "Explanation MUST be bullet-like list in JSON (array). Each explanation item MUST include sources tags "
            "from CONTEXT (only [1]...[k]). "
            "If CONTEXT is insufficient, output: NOT_FOUND. "
            "Each explanation item MUST include at least one direct quote from CONTEXT wrapped in double quotes, "
            "and the quoted text must appear verbatim in CONTEXT. If you cannot quote, output NOT_FOUND. "
            "Return STRICT JSON only, schema: " + MCQ_SCHEMA_HINT
        )
        user = f"TOPIC: {topic}\n\nCONTEXT:\n{ctx}"
    for _ in range(max_retry + 1):
        raw = call_ollama_chat(ollama_model, ollama_base_url, system, user)
        if raw.strip() == "NOT_FOUND":
            return None
        obj = parse_first_json(raw)
        if obj and "question" in obj and "options" in obj and "answer" in obj and "explanation" in obj:
            if _bad_mcq(obj):
                print("[RETRY] bad mcq")
                continue  # retry

            if strict_quote and (not _has_verbatim_quote_in_ctx(obj, ctx)):
                print("[RETRY] no verbatim quote")
                continue
            used = []
            for it in obj.get("explanation", []):
                for s in it.get("sources", []):
                    if isinstance(s, str) and re.fullmatch(r"\[\d+\]", s) and s not in used:
                        used.append(s)

            obj["sources_used"] = used
            # 將 sources_map 補回題目物件（UI 用）
            obj["sources_map"] = {k: sources_map[k] for k in obj.get("sources_used", []) if k in sources_map}

            if not obj["sources_map"] or not obj["sources_used"]:
                print("[RETRY] empty sources")
                continue

            return obj
    return None

def main() -> None:
    args = parse_args()

    random.seed(args.seed)

    args.ollama_model = resolve_ollama_model(args.ollama_base_url, args.ollama_model)

    corpus_dir = resolve_exam_corpus(args.subject, args.data_type)
    index_dir = corpus_dir / "merge_jsonl" / "index" / args.chapter
    index, pages, texts, info = load_index(index_dir)
    encoder = QueryEncoder(info["model"])
    strict_quote = args.strict_quote
    
    topics = build_topic_pool_from_texts(texts, limit=300)

    if not topics:
        topics = [args.chapter]

    lang = args.lang

    questions: list[dict[str ,Any]] = []
    attempts = 0
    # 允許題目生成錯誤(NOT FOUND/JOSN異常)，所以多嘗試
    max_attempts = args.n*3
    seen_q = set()
    while len(questions) < args.n and attempts < max_attempts:
        attempts += 1
        topic = random.choice(topics)
        if re.fullmatch(r"CH\d+(?:-CH\d+)?", topic.strip(), re.IGNORECASE):
            continue

        results = search_topk(index, pages, texts, encoder, topic, args.k)
        ctx, sources_map = build_context(results, args.max_ctx_chars, args.ctx_per_doc_chars)
        if not ctx:
            continue
        if len(ctx) < 500:
            continue
        if len(re.findall(r"\bCH\d+\b", ctx)) >= 3 and len(ctx) < 1000:
            continue

        q = generate_mcq_from_context(
            topic=topic,
            ctx=ctx,
            sources_map=sources_map,
            ollama_model=args.ollama_model,
            ollama_base_url=args.ollama_base_url,
            lang=lang,
            strict_quote=strict_quote,
        )

        if q is None:
            continue
        qid = f"Q{len(questions)+1:04d}"
        q["id"] = qid
        q["type"] = "mcq"
        q["topic"] = topic

        norm = re.sub(r"\s+", " ", q.get("question","")).strip().lower()
        if norm in seen_q:
            continue
        seen_q.add(norm)

        questions.append(q)

    if not questions:
        raise RuntimeError(
            f"[ERROR] 題目生成失敗：0 題。chapter={args.chapter}, attempts={attempts}, requested={args.n}"
        )

    if len(questions) < args.n:
        print(
            f"[WARN] 題目生成數量不足：requested={args.n}, generated={len(questions)}, attempts={attempts}"
        )

    out_dir = ART_DIR / args.subject / args.data_type / args.chapter
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"exam_{now_ts()}.json"

    payload = {
        "meta": {
            "subject": args.subject,
            "data_type": args.data_type,
            "chapter": args.chapter,
            "lang": lang,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "generator": {
                "retrieval": {"k": args.k, "max_ctx_chars": args.max_ctx_chars, "ctx_per_doc_chars": args.ctx_per_doc_chars},
                "llm": {"provider": "ollama", "model": args.ollama_model, "temperature": 0.0},
            },
        },
        "questions": questions,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote {out_path} (questions={len(questions)}, attempts={attempts})")

if __name__ == "__main__":
    main()
