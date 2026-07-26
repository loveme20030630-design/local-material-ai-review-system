from __future__ import annotations

"""
用法：
  # 1) Search：只做向量檢索（不呼叫 LLM）
  python search_index_qa.py <SUBJECT> <ppt|pdf> <CHAPTER> "<QUERY>" [k]

  # 2) QA（RAG）：先檢索 topK，再把結果當 context 丟給本地 qwen 回答（預設離線）
  set LLM_PROVIDER=ollama
  set OLLAMA_MODEL=llama3.1:8b
  set OLLAMA_BASE_URL=http://localhost:11434
  python search_index_qa.py <SUBJECT> <ppt|pdf> <CHAPTER> "<QUESTION>" [k] --qa

註：
  - k = 檢索回來的 topK 結果數（不是關鍵字數）
  - --qa 模式會把 topK 結果組成 context 再問 LLM（RAG）
  - 預設走 Ollama localhost：離線，不會上網抓資料
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import unicodedata

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parent
DATA_DIR = BASE_DIR / "data"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

class QueryEncoder:
    """
    Query → embedding（mean pooling + L2 normalize）

    注意：
      - 這個 model 必須跟 build_index 時用的 encoder 一致
        不一致會讓向量空間不同，檢索會變得很爛
    """

    def __init__(self, model_name: str) -> None:
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        """回傳 shape=(1, dim) np.float32"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            last = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu().numpy().astype(np.float32)

def parse_args() -> argparse.Namespace:
    """CLI 參數：對齊 search_index.py 的位置參數 + 多一個 --qa。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", help="學科資料夾名（data/<subject>/...）")
    ap.add_argument("data_type", help="ppt or pdf")
    ap.add_argument("chapter", help="章節，例如 CH02-CH04 或 II-V")
    ap.add_argument("query", nargs="?", default=None, help="查詢文字；若無輸入則進入互動模式")
    ap.add_argument("k", nargs="?", type=int, default=15, help="topK，預設=15")

    # QA / RAG
    ap.add_argument("--qa", action="store_true", help="啟用 RAG 問答（先檢索，再讓 LLM 回答）")
    ap.add_argument("--max_ctx_chars", type=int, default=9000, help="送給 LLM 的 context 總字元上限")
    ap.add_argument("--ctx_per_doc_chars", type=int, default=700, help="每筆檢索結果最多取多少字元")

    # LLM：預設 Ollama local（離線）
    ap.add_argument(
        "--llm_provider",
        choices=["ollama", "openai"],
        default=os.environ.get("LLM_PROVIDER", "ollama").strip().lower(),
    )
    ap.add_argument("--ollama_model", default=(os.environ.get("OLLAMA_MODEL", "").strip()))
    ap.add_argument(
        "--ollama_base_url",
        default=(os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").strip() or "http://localhost:11434"),
    )
    ap.add_argument("--openai_model", default=os.environ.get("OPENAI_MODEL", "").strip())
    return ap.parse_args()

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

def load_index(index_dir: Path) -> tuple[faiss.Index, list[dict[str, Any]], list[str], dict[str, Any]]:
    """載入 vectors.faiss / pages.json / texts.json / index.info.json（含 model）。"""
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

def format_one_result(rank: int, score: float, page: dict[str, Any], text: str) -> str:
    """終端顯示用（score 是相似度，不是機率）。"""
    doc_id = page.get("doc_id")
    pno = page.get("page")
    images = page.get("images", [])
    ocr = page.get("ocr", "")

    out: list[str] = []
    out.append(f"#{rank} score = {float(score):.4f}")
    out.append(f"doc_id = {doc_id} page = {pno}")
    out.append(f"  images = {images}")
    if isinstance(ocr, str) and ocr.strip():
        out.append("  OCR:")
        out.append(ocr)
    out.append("  TEXT:")
    out.append(text)
    out.append("-" * 60)
    return "\n".join(out)

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

def _result_key(item: tuple[float, dict[str, Any], str]) -> tuple[str, Any]:
    _score, page, _text = item
    return (str(page.get("doc_id", "")), page.get("page", None))

def hybrid_search(
    index: faiss.Index,
    pages: list[dict[str, Any]],
    texts: list[str],
    encoder: QueryEncoder,
    query: str,
    k: int,
) -> list[tuple[float, dict[str, Any], str]]:
    """
    先做 exact search，再用向量搜尋補滿。
    適合 QA / 整理資料，避免專有名詞被 FAISS 漏掉。
    """
    exact_hits = exact_search(pages, texts, query, k)
    vector_hits = search_topk(index, pages, texts, encoder, query, k)

    merged: list[tuple[float, dict[str, Any], str]] = []
    seen: set[tuple[str, Any]] = set()

    for item in exact_hits + vector_hits:
        key = _result_key(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= k:
            break

    return merged

def run_search(
    index: faiss.Index,
    pages: list[dict[str, Any]],
    texts: list[str],
    encoder: QueryEncoder,
    query: str,
    k: int,
) -> None:
    """Search 模式：先做文字精確命中，再做向量可能相關。"""
    exact_hits = exact_search(pages, texts, query, k)

    print(f"\n[QUERY {query}]\n")

    if exact_hits:
        print("[EXACT MATCH] 找到直接命中結果\n")
        for rank, (score, page, text) in enumerate(exact_hits, 1):
            print(format_one_result(rank, score, page, text))
        return

    print("[NO EXACT MATCH] 未找到直接文字命中，以下為向量檢索的可能相關內容\n")
    results = hybrid_search(index, pages, texts, encoder, query, k)
    for rank, (score, page, text) in enumerate(results, 1):
        print(format_one_result(rank, score, page, text))

def _llm_post_json(url: str, payload: dict[str, Any], api_key: str = "") -> dict[str, Any]:
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

def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))

def _norm_search_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    return s

def _page_blob(page: dict[str, Any], text: str) -> str:
    parts = [
        str(page.get("doc_id", "") or ""),
        str(page.get("title", "") or ""),
        str(page.get("section", "") or ""),
        str(page.get("ocr", "") or ""),
        text or "",
    ]
    return "\n".join(parts)

def exact_search(
    pages: list[dict[str, Any]],
    texts: list[str],
    query: str,
    k: int,
) -> list[tuple[float, dict[str, Any], str]]:
    variants = _exact_query_variants(query)
    hits: list[tuple[float, dict[str, Any], str]] = []
    seen: set[tuple[str, Any]] = set()

    for q0 in variants:
        q = _norm_search_text(q0)
        if not q:
            continue

        for page, text in zip(pages, texts):
            blob = _page_blob(page, text)
            blob_norm = _norm_search_text(blob)

            if q in blob_norm:
                key = (str(page.get("doc_id", "")), page.get("page", None))
                if key in seen:
                    continue
                seen.add(key)
                hits.append((1.0, page, text))

                if len(hits) >= k:
                    return hits

    return hits

def _detect_task_type(question: str) -> str:
    q = question or ""

    if any(w in q for w in ["流程", "步驟"]):
        return "process"

    if any(w in q for w in ["比較", "差異", "不同"]):
        return "compare"

    if any(w in q for w in ["整理", "重點", "考前", "列出"]):
        return "summary"

    return "qa"


def _build_core_rules(question: str) -> str:
    if _contains_cjk(question):
        return (
            "請使用繁體中文回答。\n"
            "只能根據 CONTEXT 回答，不要使用模型既有知識補充。\n"
            "如果 CONTEXT 不足以回答，請說明目前檢索內容主要在講什麼。\n"
            "答案中的主要概念、技術或結論要附來源頁碼，例如 CH05/p39。\n"
        )

    return (
        "Answer in English.\n"
        "Use only the provided CONTEXT. Do not add outside knowledge.\n"
        "If the CONTEXT is insufficient, state what the retrieved content is mainly about.\n"
        "Main concepts, techniques, or conclusions should include source pages, such as CH05/p39.\n"
    )


def _build_task_rules(question: str) -> str:
    task = _detect_task_type(question)

    if _contains_cjk(question):
        if task == "process":
            return (
                "這是流程/步驟題。\n"
                "若 CONTEXT 沒有明確定義標準流程，請先說明：以下是依檢索內容歸納出的複習流程，不代表教材明確定義的唯一流程。\n"
                "請用表格回答，欄位固定為：步驟、目的、相關技術、根據 CONTEXT 的說明、來源頁碼。\n"
                "每列都必須有來源頁碼；沒有來源的內容不要列入。\n"
                "不要使用『完整流程』『標準流程』『一定包含』等絕對語氣。\n"
            )

        if task == "compare":
            return (
                "這是比較題。\n"
                "請用表格回答，欄位固定為：項目、用途/定義、主要差異、來源頁碼。\n"
                "使用者指定但 CONTEXT 找不到的項目，請標示「未找到」。\n"
            )

        if task == "summary":
            return (
                "這是整理題。\n"
                "請條列整理重點，每點附來源頁碼。\n"
                "不要列出 CONTEXT 沒有支撐的內容。\n"
            )

        return (
            "這是一般問答題。\n"
            "若 CONTEXT 有部分內容可回答，請先整理可被 CONTEXT 支撐的答案，再說明哪些細節不足。\n"
            "不要一開始就說無法回答，除非 CONTEXT 完全沒有相關內容。\n"
            "請簡潔回答，並附來源頁碼。\n"
        )

    if task == "process":
        return (
            "This is a process/step question.\n"
            "If the CONTEXT does not define a standard process, state that the process is inferred from the retrieved content.\n"
            "Use a table with columns: Step, Purpose, Related Techniques, CONTEXT-based Explanation, Source Page.\n"
            "Every row must include a source page; omit unsupported content.\n"
        )

    if task == "compare":
        return (
            "This is a comparison question.\n"
            "Use a table with columns: Item, Definition/Use, Main Difference, Source Page.\n"
            "If a requested item is not found in the CONTEXT, mark it as not found.\n"
        )

    if task == "summary":
        return (
            "This is a summary question.\n"
            "Use bullet points. Each point should include source pages.\n"
            "Do not include unsupported content.\n"
        )

    return (
    "This is a general Q&A question.\n"
    "If the CONTEXT partially supports an answer, provide the supported answer first, then state what details are missing.\n"
    "Do not say the question cannot be answered at the beginning unless the CONTEXT is completely unrelated.\n"
    "Answer concisely with source pages.\n"
    )


def _build_prompt_policy(question: str) -> str:
    return _build_core_rules(question) + "\n" + _build_task_rules(question)

def _build_user_task(question: str, ctx: str) -> str:
    task = _detect_task_type(question)

    if _contains_cjk(question):
        if task == "process":
            task_text = (
                "請根據 CONTEXT 整理此主題可能涉及的流程或步驟。"
                "不要追求教材外的完整性，只整理 CONTEXT 有支撐的內容。"
                "請使用指定表格格式回答。"
            )
        elif task == "compare":
            task_text = (
                "請根據 CONTEXT 比較使用者提到的項目。"
                "找不到的項目請標示未找到，不要自行補充。"
            )
        elif task == "summary":
            task_text = (
                "請根據 CONTEXT 整理重點。"
                "只列出 CONTEXT 有支撐的內容。"
            )
        else:
            task_text = (
                "請根據 CONTEXT 回答問題。"
                "若無法回答，請說明目前檢索內容主要在講什麼。"
            )
    else:
        if task == "process":
            task_text = (
                "Summarize the possible process or steps based on the CONTEXT. "
                "Do not aim for completeness beyond the textbook passages. "
                "Use the required table format."
            )
        elif task == "compare":
            task_text = (
                "Compare the requested items based on the CONTEXT. "
                "Mark missing items as not found."
            )
        elif task == "summary":
            task_text = (
                "Summarize the key points based only on the CONTEXT."
            )
        else:
            task_text = (
                "Answer the question based on the CONTEXT. "
                "If unsupported, state what the retrieved content is mainly about."
            )

    return (
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"TASK:\n{task_text}"
    )

def _exact_query_variants(query: str) -> list[str]:
    q = query or ""
    variants = [q]

    cleaned = q
    remove_words = [
        "請", "幫我", "整理", "介紹", "說明", "解釋", "列出", "比較",
        "並附頁碼", "附上頁碼", "附頁碼", "頁碼",
        "的相關概念", "相關概念", "相關內容", "內容",
        "考前重點", "找不到請標示未找到", "找不到", "請用表格",
        "並", "和", "與"
    ]

    for w in remove_words:
        cleaned = cleaned.replace(w, " ")

    parts = re.split(r"[，,。；;：:\n\r\t]+", cleaned)
    variants.extend(parts)

    out = []
    seen = set()
    for v in variants:
        v = v.strip()
        if not v:
            continue
        nv = _norm_search_text(v)
        if not nv or nv in seen:
            continue
        seen.add(nv)
        out.append(v)

    return out

def _build_rag_context(
    results: list[tuple[float, dict[str, Any], str]],
    max_ctx_chars: int,
    ctx_per_doc_chars: int,
) -> tuple[str, list[str]]:
    """
    組 RAG context：
      - OCR + TEXT 合併
      - 壓縮空白（\\s+ → single space）避免換行污染
      - 每筆最多 ctx_per_doc_chars
      - 全部不超過 max_ctx_chars
    """
    chunks: list[str] = []
    cites: list[str] = []
    used = 0

    for score, page, text in results:
        doc_id = str(page.get("doc_id", "unknown"))
        pno = page.get("page", None)
        ocr = page.get("ocr", "") or ""

        blob = (ocr + "\n" + (text or "")).strip()
        blob = re.sub(r"\s+", " ", blob).strip()
        if not blob:
            continue

        take = blob[: max(0, ctx_per_doc_chars)]
        cite = f"{doc_id}/p{pno}" if pno is not None else doc_id
        next_idx = len(cites) + 1
        piece = f"[{next_idx}] {cite} score={score:.4f}\n{take}\n"
        if used + len(piece) > max_ctx_chars:
            break

        chunks.append(piece)
        cites.append(cite)
        used += len(piece)

    return "\n".join(chunks).strip(), cites


def _qa_answer_is_usable(ans: str) -> bool:
    text = (ans or "").strip()
    if not text:
        return False

    low = text.lower().strip()

    if low.startswith("sources:"):
        return False

    if len(text) < 10:
        return False

    return True

def _build_compact_fallback_answer(
    results: list[tuple[float, dict[str, Any], str]],
    question: str,
    max_items: int = 3,
    max_chars_per_item: int = 220,
) -> str:
    """
    當 QA + 摘要 fallback 都失敗時，用檢索結果做最後的 template fallback。
    """
    is_zh = _contains_cjk(question)
    lines: list[str] = []

    lines.append(f"[TEMPLATE FALLBACK] {question}")
    if is_zh:
        lines.append("目前無法穩定生成回答，以下是最相關段落重點：")
    else:
        lines.append("The QA step was not usable. Most relevant retrieved snippets:")
    lines.append("")

    used = 0
    for rank, (score, page, text) in enumerate(results[:max_items], 1):
        doc_id = str(page.get("doc_id", "unknown"))
        pno = page.get("page", None)
        ocr = page.get("ocr", "") or ""
        blob = (ocr + "\n" + (text or "")).strip()
        blob = re.sub(r"\s+", " ", blob).strip()
        if not blob:
            continue

        short = blob[:max_chars_per_item].rstrip()
        if len(blob) > max_chars_per_item:
            short += "..."

        cite = f"{doc_id} / p{pno}" if pno is not None else doc_id
        lines.append(f"[{rank}] {cite} (score={score:.4f})")
        lines.append(short)
        lines.append("")
        used += 1

    if used == 0:
        lines.append("[no retrievable context found]")
        lines.append("")
    else:
        lines.append("SOURCES: " + ", ".join(f"[{i}]" for i in range(1, used + 1)))

    return "\n".join(lines).strip()


def _call_llm_chat(
    args: argparse.Namespace,
    system: str,
    user: str,
    num_predict: int,
) -> str:
    if args.llm_provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        if not args.openai_model:
            raise RuntimeError("--openai_model is empty (or OPENAI_MODEL not set)")

        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": args.openai_model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = _llm_post_json(url, payload, api_key)
        return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()

    base = args.ollama_base_url.rstrip("/")
    url = base + "/api/chat"
    payload = {
        "model": args.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": num_predict},
    }
    data = _llm_post_json(url, payload, "")
    return (data.get("message", {}).get("content", "") or "").strip()


def _llm_summarize_retrieved_context(
    args: argparse.Namespace,
    question: str,
    ctx: str,
) -> str:
    policy = _build_prompt_policy(question)
    system = (
        "You summarize retrieved textbook passages.\n"
        "Only use the provided context.\n"
        "Keep the result short and grounded.\n"
        + policy
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED CONTEXT:\n{ctx}\n\n"
        "Task:\n"
        "1. Summarize what the retrieved passages actually say.\n"
        "2. If they do not directly answer the question, state the main topic of the retrieved passages.\n"
        "3. Mention only information grounded in the retrieved text.\n"
    )
    return _call_llm_chat(args, system, user, num_predict=256)

def _num_predict_for_question(question: str) -> int:
    task = _detect_task_type(question)
    if task == "process":
        return 768
    if task in {"summary", "compare"}:
        return 640
    return 384

def _extract_main_topic(question: str) -> str:
    q = question or ""

    remove_words = [
        "請", "幫我",
        "詳細介紹", "介紹", "說明", "解釋", "整理", "列出",
        "流程步驟", "流程", "步驟",
        "越詳細越好", "越詳細", "最好", "加上", "步驟說明",
        "的", "一下", "一下子",
        "，", ",", "。", "？", "?", "！", "!",
    ]

    cleaned = q
    for w in remove_words:
        cleaned = cleaned.replace(w, " ")

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or question


def _build_retrieval_query(question: str) -> str:
    task = _detect_task_type(question)

    if task == "process":
        topic = _extract_main_topic(question)

        if _contains_cjk(question):
            return f"{topic} 相關概念 技術 主題 章節"
        return f"{topic} related concepts techniques topics chapters"

    return question

def run_qa(
    args: argparse.Namespace,
    index: faiss.Index,
    pages: list[dict[str, Any]],
    texts: list[str],
    encoder: QueryEncoder,
    question: str,
    k: int,
) -> None:
    """
    RAG QA：
      1) search topK
      2) build context
      3) ask LLM（預設 Ollama local model）
      4) QA 失敗時改走 summary fallback
      5) summary fallback 也失敗才走 template fallback
    """
    task = _detect_task_type(question)

    if task == "process":
        k = min(k, 30)
    elif task in {"summary", "compare"}:
        k = min(k, 25)
    else:
        k = min(k, 15)

    retrieval_query = _build_retrieval_query(question)
    results = hybrid_search(index, pages, texts, encoder, retrieval_query, k)
    ctx, cites = _build_rag_context(results, args.max_ctx_chars, args.ctx_per_doc_chars)

    print(f"\n[QA {question}]\n")

    if not ctx:
        print(_build_compact_fallback_answer(results, question))
        return

    policy = _build_prompt_policy(question)
    system = (
        "You are a Q&A assistant for an offline textbook corpus.\n"
        "Follow the rules strictly.\n"
        + policy
    )
    policy = _build_prompt_policy(question)
    system = (
        "You are a Q&A assistant for an offline textbook corpus.\n"
        "Follow the rules strictly.\n"
        + policy
    )

    user = _build_user_task(question, ctx)

    ans = ""
    try:
        ans = _call_llm_chat(
        args,
        system,
        user,
        num_predict=_num_predict_for_question(question),
        )
    except Exception as e:
        print(f"[QA] primary LLM failed: {e}")

    if _qa_answer_is_usable(ans):
        final_answer = ans
    else:
        summary_ans = ""
        try:
            summary_ans = _llm_summarize_retrieved_context(args, question, ctx)
        except Exception as e:
            print(f"[QA] summary fallback failed: {e}")

        if _qa_answer_is_usable(summary_ans):
            final_answer = summary_ans
        else:
            final_answer = _build_compact_fallback_answer(results, question)

    print(final_answer)
    if cites:
        print("\n[RETRIEVED]")
        for i, c in enumerate(cites, 1):
            print(f"[{i}] {c}")

def _resolve_exam_corpus(subject: str, data_type: str) -> Path:
    """根據 subject + data_type 推導 exam_corpus 目錄。"""
    subject_base = DATA_DIR / subject
    if data_type == "pdf":
        return subject_base / "pdf_exam_corpus"/"merge_jsonl"
    return subject_base / "ppt_exam_corpus"/"merge_jsonl"

def main() -> None:
    args = parse_args()

    if args.llm_provider == "ollama" and args.qa:
        args.ollama_model = resolve_ollama_model(args.ollama_base_url, args.ollama_model)

    corpus_dir = _resolve_exam_corpus(args.subject, args.data_type)

    # 允許 chapter 傳 "II-V" 或 "D:\\...\\II-V.jsonl"
    ch = args.chapter
    if isinstance(ch, str) and ch.lower().endswith(".jsonl"):
        ch = Path(ch).stem

    index_dir = corpus_dir / "index" / ch

    index, pages, texts, info = load_index(index_dir)
    encoder = QueryEncoder(info["model"])

    if args.query is not None:
        q = args.query.strip()
        if not q:
            raise SystemExit("query is empty")
        if args.qa:
            run_qa(args, index, pages, texts, encoder, q, args.k)
        else:
            run_search(index, pages, texts, encoder, q, args.k)
        return

    # interactive mode
    try:
        while True:
            q = input("QUERY> ").strip()
            if not q:
                break
            if args.qa:
                run_qa(args, index, pages, texts, encoder, q, args.k)
            else:
                run_search(index, pages, texts, encoder, q, args.k)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()