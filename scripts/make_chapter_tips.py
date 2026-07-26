from __future__ import annotations

import argparse  # CLI 參數解析
import json  
import math  # log
import re  
import os 
import urllib.parse  # 處理 URL
import urllib.error  # 處理 URL 錯誤
import urllib.request  # 發送 HTTP 請求

from collections import Counter, defaultdict  # 計數器
from pathlib import Path  
from typing import Any

THIS_DIR: Path = Path(__file__).resolve().parent
BASE_DIR: Path = THIS_DIR.parent
DATA_DIR: Path = BASE_DIR / "data"
# 常見程式碼/碎片詞黑名單：避免被當成章節重點
STOP_CN: set[str] = {
    "例如", "可以", "定義", "表示", "如下", "其中", "因此", "我們", "你們", "以及", "進行", "使用",
    "工作", "目標", "系統", "資料", "結果", "問題", "相關", "圖形", "程式", "大致", "分成下列幾種",
    "應用", "好的", "一下", "個英文字母", "假設事件", "顧名思義", "比較", "進入", "開始", "假設", "設定",
    "如圖", "這頁是沒有", "首先", "下列", "形成下列", "包含下列", "被應用在", "可應用在", "可用於", "用於", 
    "用在", "總共", "都是", "直接"
}
STOP_EN: set[str] = {
    "the", "and", "with", "from", "that", "this", "into", "used", "using",
    "int", "void", "main", "char", "double", "float", "bool",
    "return", "include", "std", "cout", "cin", "printf", "println",
    "true", "false", "null", "none", "class", "public", "private",
    "input", "output", "width", "height", "size", "type", "value",
    "range", "rate", "bits", "pixel", "point", "data", "test",
    "modern", "until", "rest", "stay", "side", "given", "large", "often", "areas", "count",
}
STOP_TERMS: set[str] = {
    "img", "inter", "main", "array", "filename",
    "opencv", "fontface", "roi",
}
FORMAT_NOISE: set[str] = {
    "www", "http", "https", "uri", "url",
    "com", "net", "org", "edu", "gov", "io",
    "pdf", "ppt", "pptx", "doc", "docx", "xls", "xlsx", "csv", "json", "jsonl", "md", "txt",
    "jpeg", "jpg", "png", "gif", "bmp", "tiff", "svg",
}
_ROMAN_TOKEN_RE = re.compile(r"^(?=[ivxlcdm]+$)(?:m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))$", re.IGNORECASE)
def is_roman_numeral_token(t: str) -> bool:
    s = (t or "").strip()
    if not s:
        return False
    # 常見頁碼/章節：i, ii, iii, iv, v...（也包含較長 roman）
    return bool(_ROMAN_TOKEN_RE.match(s))
def is_junk_fragment(term: str) -> bool:
    """
    通用：判斷 term 是否高度可疑（OCR斷裂/流程字串/格式碎片）
    不依賴科目詞庫；只看形態特徵。
    """
    t = (term or "").strip()
    if not t:
        return True

    # 1) 明顯流程/模板字串/格式碎片（不限定語言）

    boiler = (
        "圖片裡沒文字", "這頁是沒有txt的", "這頁是沒有", "沒有抓到", "跳過", "SKIP", "DONE", "ERROR", "WARN"
    )
    for b in boiler:
        if b in t:
            return True
        
    # 2) 太短且資訊量低（跨語料）：2~3字很常是斷裂或泛詞
    if len(t) <= 2:
        return True

    # 3) 字元種類混雜/符號過多（路徑、網址、格式殘片）
    if "://" in t or "/" in t or "\\" in t:
        return True

    # 4) 重複度過高（例如 OCR 亂碼/重複字串）
    s = re.sub(r"\s+", "", t)
    if len(s) >= 8:
        uniq_ratio = len(set(s)) / len(s)
        if uniq_ratio < 0.35:
            return True

    # 5) 全是非文字/數字/符號
    if re.fullmatch(r"[\d\W_]+", t):
        return True

    # 6) 很像被截斷的尾巴/頭巴：短詞 + 常見功能性結尾
    if len(t) <= 4:
        if re.search(r"(ing|ed|tion|ness|ment)$", t.lower()):
            return True
        if re.search(r"[態能值例序構法]$", t):
            return True
        
    # 7)流程/敘述句片段：通常不是術語（跨科通用）
    if re.match(r"^(?:即|若|則|因此|所以|可以|用來|用於|進入|開始|總共|首先|接著|然後)", t):
        return True
    
    # 8) OCR 常見斷裂：短詞 + 常見名詞後綴（多字）
    if len(t) <= 6 and re.search(r"(型態|能力|序列|元素|範例)$", t):
        return True
    
    # 9) 殘句/片語模式
    if re.search(r"(分為下列|包含下列|如下|如圖|舉例|例如|可以|用於|用來|透過|通過|因此|所以|若|則)", t):
        # 這類若又不夠長，幾乎一定不是術語
        if len(t) <= 8:
            return True

    # 常見「約/大約/落在/介於」這種敘述片段
    if re.search(r"(約落在|大約|落在|介於)", t):
        return True

    # 10) 通用語法碎片（跨科）：以功能字開頭/結尾的短片語
    if len(t) <= 6:
        # 開頭功能字（不是名詞術語的起手）
        if re.match(r"^(?:其|此|該|本|各|每|某|若|則|較|更|最|可|會|將|已|未|在|由|對|於|以|而|及|與|或|之|的)", t):
            return True
        # 結尾功能字（像「包含/步驟/容易」這種常出現在敘述句）
        if re.search(r"(?:包含|步驟|容易|可能|因此|所以|等等|如下)$", t):
            return True

    # 11) 典型掉字殘片：以「出/個/其/此」開頭 + 後面接名詞（多半是被截斷）
    if re.match(r"^(?:出|個|其|此)[\u4e00-\u9fff]{2,}$", t) and len(t) <= 8:
        return True
    return False
def in_stop_term(term: str) -> bool:
    t = term.strip()
    if not t:
        return True

    tl = t.lower()

    if tl in STOP_TERMS:
        return True

    if tl in FORMAT_NOISE:
        return True

    if "." in tl and tl.rsplit(".", 1)[-1] in FORMAT_NOISE:
        return True

    if "://" in t or "/" in t or "\\" in t:
        return True
    
    if re.fullmatch(r"[\d\W_]+", t):
        return True

    if re.fullmatch(r"[a-z]{1,2}", tl):
        return True

    return False

def is_cn(term: str) ->bool:
    """
    判斷 term 是否包含中文，避免常見英文被過多選取
    """
    return any("\u4e00" <= c <= "\u9fff" for c in term)

def parse_args() -> argparse.Namespace:
    """
    用法：
      python make_chapter_tips.py <SUBJECT> <pdf|ppt> <cn|en> <JSONL_PATH> [topn] --use_llm --refresh_llm
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", help="選擇科目")
    ap.add_argument("data_type", choices = ["ppt", "pdf"], help="ppt or pdf")
    ap.add_argument("lang", choices=["cn", "en"], default="cn", help="選擇中文(cn)或是英文(en)")
    ap.add_argument("jsonl_path", help="選擇jsonl的路徑")
    ap.add_argument("topn",nargs="?", type=int, default=8, help="輸出重點數量，預設值為8")
    ap.add_argument("--use_llm", action="store_true", help="使用LLM來過濾重點")
    ap.add_argument("--refresh_llm", action="store_true", help="忽略cache，強制呼叫LLM")
    ap.add_argument("--ollama_model", default=os.environ.get("OLLAMA_MODEL", "").strip(), help="可選：指定 Ollama 模型名；不指定時自動掃描")
    ap.add_argument("--ollama_base_url", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").strip() or "http://localhost:11434", help="Ollama base URL")
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

def read_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    """讀取 jsonl；輸入不存在或沒有有效資料列時直接失敗。"""
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"[ERROR] 找不到 tips 輸入 JSONL: {jsonl_path}")

    page: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                page.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"[ERROR] JSONL 格式錯誤: {jsonl_path} line {line_no}: {e}")

    if not page:
        raise RuntimeError(f"[ERROR] tips 輸入 JSONL 為空: {jsonl_path}")

    return page

def extract_term(text: str, lang: str = "cn") -> list[str]:
    """
    不用切詞的抽詞
    中文：抓連續 CJK 片段，長度 2~8
    英文：抓 a-zA-Z 單字，長度 >=3
    """
    terms: list[str] =[]

    if lang == "cn":
        for m in re.finditer(r"[\u4e00-\u9fff]{2,8}", text):
            t = m.group(0)
            if t in STOP_CN:
                continue
            terms.append(t)
        return terms
    
    elif lang == "en":
        # 1) Acronym / ALL CAPS tokens
        for m in re.finditer(r"\b[A-Z]{2,8}\b", text):
            t = m.group(0)
            tl = t.lower()
            if is_roman_numeral_token(tl):
                continue
            if tl in STOP_EN:
                continue
            if in_stop_term(tl):
                continue
            terms.append(tl)

        # 2) Title-like phrases (e.g., "Religious Wars")
        for m in re.finditer(r"\b(?:[A-Z][a-z]{2,}\s){1,4}[A-Z][a-z]{2,}\b", text):
            t = re.sub(r"\s+", " ", m.group(0)).strip().lower()
            if is_roman_numeral_token(t):
                continue
            if t in STOP_EN:
                continue
            if in_stop_term(t):
                continue
            terms.append(t)

        # 3) General lowercase words
        for m in re.finditer(r"\b[a-z]{3,}\b", text.lower()):
            t = m.group(0)
            if is_roman_numeral_token(t):
                continue
            if t in STOP_EN:
                continue
            if in_stop_term(t):
                continue
            terms.append(t)

    return terms

def pick_page_number(p: dict[str, Any]) -> int|None:
    """
    嘗試從 page dict 抽出頁碼
    """
    for k in ("page", "page_no", "page_num"):
        v = p.get(k, None)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.strip().isdigit():
            s = v.strip()
            if s.isdigit():
                return int(s)
    return None

def pick_image_path(p: dict[str, Any]) -> str | None:
    """
    抽圖片路徑欄位
    常見欄位：image / images
    """ 
    img = p.get("image", None)
    if isinstance(img, str) and img.strip():
        return img.strip()
    
    imgs = p.get("images", None)
    if isinstance(imgs, list) and imgs:
        first = imgs[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    
    return None

def _percentile_score(sorted_vals: list[float], q: float) -> float:
    """
    計算 sorted_vals 的 q 分位數
    """
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    w = pos - lo
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w

def _pick_doc_id(p: dict[str, Any]) -> str:
    return str(p.get("doc_id") or p.get("doc") or p.get("ch") or "unknown").strip() 

def _llm_cache_path(subject: str, data_type: str, jsonl_path: Path, lang: str) -> Path:
    corpus_dir = DATA_DIR / subject / ("pdf_exam_corpus" if data_type == "pdf" else "ppt_exam_corpus")
    stem = jsonl_path.stem
    out_dir = corpus_dir / "chapter_tips"/ f"{stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{jsonl_path.stem}.{lang}.llm_allow_terms.json"

def _chunk_text(text: str, chunk_char: int = 4500, max_chunks: int = 10) -> list[str]:
    """
    將長文本切成多段，每段 chunk_char 字元，最多 max_chunks 段
    """
    text = text or ""
    if chunk_char <= 0:
        return [text]

    chunks: list[str] = []
    cur = 0
    n = len(text)

    while cur < n and len(chunks) < max_chunks:
        chunks.append(text[cur:cur + chunk_char])
        cur += chunk_char

    return chunks

def _llm_post_json(url: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    """
    發送 POST 請求到 LLM API，並返回 JSON 解析結果
    """
    headers={
        "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), 
        headers=headers, method="POST")
    
    try:
        timeout_seconds = int(os.environ.get("LLM_API_TIMEOUT", "300").strip())
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            resp_data = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        error_info = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        print(f"[ERROR] LLM API request failed: {e.code} {e.reason}, details: {error_info}")
        raise RuntimeError(f"LLM API request failed: {e.code} {e.reason}")
    return json.loads(resp_data)

def _build_evidence_snippet(pages: list[dict[str, Any]], page_idx: int | None, term: str) -> str:
    """
    給 LLM 的證據片段：從代表頁抓 text+ocr 前 N 字
    """
    if page_idx is None:
        return ""
    if page_idx < 0 or page_idx >= len(pages):
        return ""
    p = pages[page_idx]
    ocr_blob = p.get("ocr", "") or p.get("ocrs", "")
    blob = f"{p.get('text', '')}\n{ocr_blob}".strip()
    if not blob:
        return ""
    blob = re.sub(r"\s+", " ", blob)  # 壓縮空白，避免 prompt 爆長
    # 盡量讓 term 附近出現（若找得到）
    tl = term.strip()
    pos = blob.find(tl) if tl else -1
    if pos >= 0:
        s = max(0, pos - 60)
        e = min(len(blob), pos + 60)
        return blob[s:e]
    return blob[:120]

def _llm_select_ids_from_candidates(
    chapter_ID: str,
    pages: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    lang: str,
    k: int,
) -> list[int]:
    """
    讓 LLM 從候選清單中挑選 idx（避免 JSON / 避免字串對齊）
    candidates: items（已算好 raw_score 等）
    回傳：保留的候選 index（0-based，對應 candidates 的位置）
    """
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower() or "ollama"
    # 控制 prompt 長度：只送前 M 個候選，避免 7B 卡死
    max_cands = int(os.environ.get("LLM_MAX_CANDS", "120").strip() or "120")
    cand = candidates[:max_cands]

    # 建立帶 idx 的候選列表
    lines: list[str] = []
    for i, it in enumerate(cand):
        term = str(it.get("term", "")).strip()
        if not term:
            continue

        # 把 page_df = 0 的候選刪除，避免 LLM  
        feats = it.get("features", {}) or {}
        page_df = feats.get("page_df", 0)
        if isinstance(page_df, int) and page_df <= 0:
            continue

        rep = it.get("representative", None) or {}
        page_idx = rep.get("page_idx", None)

        ev = _build_evidence_snippet(pages, page_idx if isinstance(page_idx, int) else None, term)
        # idx \t term \t evidence
        lines.append(f"{i}\t{term}\t{ev}")

    if not lines:
        return []

    if lang == "cn":
        system_prompt = (
            "你是課本章節的關鍵詞篩選器。"
            f"我給你一份候選詞清單（每行：idx<TAB>term<TAB>evidence）。"
            f"請你只挑選最重要的 {k} 個，必須是本章核心術語、方法、名詞。"
            "務必刪掉：泛詞（如 基本概念/資料/結果）、流程字串、OCR 斷裂碎片、無意義短詞。"
            "輸出格式嚴格：只輸出一行 `KEEP: i,j,k,...`（用逗號分隔 idx），不要任何其他字。"
            "硬性規則：不要選以「即/若/則/因此/所以/可以/用來/用於/進入/開始/總共/首先/接著/然後」開頭的 term。"
            "硬性規則：不要選描述動作/流程的片語；只選名詞概念或方法名。"
            "硬性規則：包含「分為下列/包含下列/如下/如圖/舉例/例如/可以/用於/用來/透過/通過/因此/所以/若/則/約落在/介於/其中/包括/其中包括/較容易/較不容易/較為容易」的 term 一律不要選。"
            "如果開頭為「給定/若/則/因此/所以/可以/用來/用於/進入/開始/總共/首先/接著/然後/的」，也一律不要選。"
            "如果結尾為「即/若/則/因此/所以/可以/用來/用於/進入/開始/總共/首先/接著/然後」，也一律不要選。"
        )
    else:
        system_prompt = (
            "You are a key-term filter for textbook chapters."
            f"I will give candidate terms per line: idx<TAB>term<TAB>evidence."
            f"Select the {k} most important core technical terms."
            "Drop generic words, boilerplate strings, OCR fragments."
            "Strict output: one line `KEEP: i,j,k,...` only."
            "Hard rule: Do not select terms starting with 'given/if/then/therefore/so/can/be used for/used in/enter/start/total/firstly/next/then'."
            "Hard rule: Do not select action/process-describing phrases; only select noun concepts or method names."
            "Hard rule: Do not select terms containing 'divided into the following/including the following/as follows/as shown/for example/such as/can be used for/used to/through/by/therefore/so/given/if/then/about around/fall between'."
            "Hard rule: If starting with 'given/if/then/therefore/so/can/be used for/used in/enter/start/total/firstly/next/then/of', do not select."        
            "Hard rule: If ending with 'is/are/if/then/therefore/so/can/be used for/used in/enter/start/total/firstly/next/then/of', do not select."        
        )

    user_obj = {
        "chapter_ID": chapter_ID,
        "K": k,
        "candidates": "\n".join(lines),
    }

    # 送給 LLM，讓它從 idx 中挑選出要保留的（避免直接輸出 term，造成對齊問題）
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")
        model = os.environ.get("OPENAI_MODEL", "").strip()
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
            ],
        }
        data = _llm_post_json(url, payload, api_key)
        content = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
    else:
        api_key = ""
        model = os.environ.get("OLLAMA_MODEL", "").strip() or "llama3.1:8b"
        base_url = os.environ.get("OLLAMA_BASE_URL", "").strip() or "http://localhost:11434"
        url = base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 128},
        }
        data = _llm_post_json(url, payload, api_key)
        content = (data.get("message", {}).get("content", "") or "").strip()

    # 解析 `KEEP: 1,2,3`
    m = re.search(r"KEEP\s*:\s*([0-9,\s]+)", content, re.IGNORECASE)
    if not m:
        return []
    nums = re.findall(r"\d+", m.group(1))
    if not nums:
        return []

    keep: list[int] = []
    seen = set()
    for s in nums:
        try:
            idx = int(s)
        except:
            continue
        if idx < 0 or idx >= len(cand):
            continue
        if idx in seen:
            continue
        seen.add(idx)
        keep.append(idx)
        if len(keep) >= k:
            break
    return keep

def _llm_extract_terms_for_chapter(
    chapter_ID: str,
    chapter_text: str,
    lang: str,
    k: int = 30,
) -> list[str]:
    """
    使用 LLM 來抽取章節重點詞，作為額外的過濾條件
    """
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower() or "ollama"
    
    chunks = _chunk_text(chapter_text, chunk_char=3000, max_chunks=6)

    if lang == "cn":
        system_prompt = f"你是章節重點詞抽取專家，請從以下章節內容中抽取 {k} 個最重要的詞彙。請只回傳詞彙本身，每行一個詞，不要任何解釋、不要編號、不要多餘符號，所有詞都必須從文本裡選取。"
        user_obj = {
            "chapter_ID": chapter_ID,   
            "K": k,
            "chunks": chunks, 
        }
    else:
        system_prompt = f"You are an expert in extracting key terms from chapter content. Please extract the {k} most important terms from the following chapter content. Only return the terms themselves, one term per line, without any explanations, numbering, or extra symbols. All terms must be selected from the text."
        user_obj = {
            "chapter_ID": chapter_ID,   
            "K": k,
            "chunks": chunks, 
        }

    if provider == "openai":
        # provider == "openai"
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")
        model = os.environ.get("OPENAI_MODEL", "").strip()
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model,
            "temperature": 0.0, # 固定輸出，避免隨機性
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"}
        }
        data = _llm_post_json(url, payload, api_key)
        content = data["choices"][0]["message"]["content"]
        

    else:
        # provider == "ollama" (default)
        api_key = "" # Ollama 本地部署通常不需要 API Key
        model = os.environ.get("OLLAMA_MODEL", "").strip() or "llama3.1:8b"
        base_url = os.environ.get("OLLAMA_BASE_URL", "").strip() or "http://localhost:11434"
        url = base_url.rstrip("/") + "/api/chat"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
            ], 
            "stream": False,
            #"format": "json",
            "options": {"temperature": 0.0, "num_predict": 256}
        }
    
        data = _llm_post_json(url, payload, api_key)
        
    content = (data.get("message", {}).get("content", "")).strip()
    line = [x.strip() for x in content.splitlines() if x.strip()]
    if line:
        terms = line
    else:
        terms = []

    # 嘗試解析 LLM 回傳的 JSON，並抽取 terms
    if not terms:
        try:
            out = json.loads(content)
            terms = out.get("terms", [])
        except Exception as e:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                try:
                    out = json.loads(m.group(0))
                    terms = out.get("terms", [])
                except Exception as e2:
                    print(f"[WARN] 無法解析 LLM 回傳的 JSON，將嘗試抽取內容文字當 terms: {e2}")
                    terms = []
            else:
                print(f"[WARN] 無法解析 LLM 回傳的 JSON，將嘗試抽取內容文字當 terms: {e}")
                terms = []

    # 如果 JSON 格式錯誤，直接抽取內容文字當 terms
    if not terms:
        terms = []
        cand = re.findall(r"\"([^\"]+)\"", content)
        if not cand:
            cand = re.findall(r"\'([^\']+)\'", content)

        bad = {"terms", "term", "keywords", "keyword", "key_terms", "key_term"}
        terms = [t.strip() for t in cand if t.strip() not in bad]

    # 後處理：過濾掉明顯不合理的詞彙
    cleaned_terms: list[str] = []
    for t in terms:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t:
            continue
        if in_stop_term(t):
            continue
        cleaned_terms.append(t)
    return cleaned_terms[:k]  # 前面可能是章節標題或重複的詞，跳過

def build_llm_allow_terms_by_chapter(
        subject: str,
        data_type: str,
        jsonl_path: Path,
        pages: list[dict[str, Any]],
        lang: str,
        refresh: bool = False
) -> dict[str, set[str]]:
    """
    為每章建立一個 LLM 允許詞列表，作為後續過濾章節重點詞的條件

    1. 嘗試從快取讀取，避免重複呼叫 LLM
    2. 如果沒有快取或 refresh=True，則對每章呼叫 LLM
    3. LLM 的輸入是章節文本，輸出是該章的重點詞列表
    4. 寫入快取，供後續使用
    """
    cache_path = _llm_cache_path(subject, data_type, jsonl_path, lang)
    # 嘗試從快取讀取
    if cache_path.exists() and not refresh:
        try:
            cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
            out: dict[str, set[str]] = {}
            if isinstance(cache_data, dict):
                for doc_id, terms in cache_data.items():
                    if isinstance(terms, list):
                        out[str(doc_id)] = set(str(x).strip() for x in terms if str(x).strip())
                return out
        except Exception as e:
            print(f"[WARN] 無法讀取 LLM 允許詞快取，將重新生成: {e}")
            pass

    # 沒有快取，或快取無效，對每章呼叫 LLM
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    
    for p in pages:
        doc_id = _pick_doc_id(p)

        ocr_blob = p.get("ocr", "")
        if not ocr_blob:
            ocr_blob = p.get("ocrs", "")

        blob = f"{p.get('text', '')}\n{ocr_blob}"
        if blob.strip():
            by_doc[doc_id].append(blob)
    
    out: dict[str, list[str]] = {}
    out_dump: dict[str, Any] = {}

    # 對每章呼叫 LLM
    for doc_id, blobs in by_doc.items():
        chapter_text = "\n".join(blobs)

        llm_terms = _llm_extract_terms_for_chapter(
            chapter_ID=doc_id,
            chapter_text=chapter_text,
            lang=lang,
            k=30,
        )
        
        allow_terms: set[str] = set()

    # 後處理 LLM 輸出，建立允許詞列表
        for t in llm_terms:
            if not isinstance(t, str):
                continue
            t = t.strip()
            if not t:
                continue
            if in_stop_term(t):
                continue
            allow_terms.add(t)

        out[doc_id] = allow_terms
        out_dump[doc_id] = sorted(allow_terms)

    # 寫入快取
    cache_path.write_text(json.dumps(out_dump, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

def build_chapter_tips_representatives(
        pages: list[dict[str, Any]],
        topn: int,
        lang: str,
        llm_allow_terms_by_chapter: dict[str, set[str]] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """
    每章 topN 詞
    score：簡單 TF-IDF
    score = tf * log((C+1)/(df+1))
    代表頁：在該章內出現次數最多的 page，用 text+ocr 抽詞計數
    """
    # 章 -> term TF
    doc_tf: dict[str, Counter[str]] =  defaultdict(Counter)

    doc_term_page_hit: dict[str, dict[str, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))

    doc_ids: set[str] = set()

    pages_in_doc_by_docid: Counter[str] = Counter()

    
    for page_idx, p in enumerate(pages):
        doc_id = str(p.get("doc_id") or p.get("doc") or p.get("ch") or "unknown").strip()
        doc_ids.add(doc_id)
        pages_in_doc_by_docid[doc_id] += 1
        ocr_blob = p.get("ocr", "")
        if not ocr_blob:
            ocr_blob = p.get("ocrs", "")
        blob = f"{p.get('text', '')}\n{ocr_blob}"
        terms = extract_term(blob, lang=lang)

        terms = [t for t in terms if (not in_stop_term(t)) and (not is_junk_fragment(t))]

        # 這一頁的 term 次數
        page_counter = Counter(terms)

        # 累積到章節 TF
        doc_tf[doc_id ].update(terms)
    
        # 累積 term->page hits
        for term, tf in page_counter.items():
            if in_stop_term(term):
                continue
            if is_junk_fragment(term):
                continue
            doc_term_page_hit[doc_id][term][page_idx] += tf
    doc_id_list = sorted(doc_ids)
    C = len(doc_id_list)

    # df：某 term 出現過的章節數，(Document Frequency)
    df: Counter[str] = Counter()
    for doc_id  in doc_id_list:
        for term in doc_tf[doc_id].keys():
            df[term] +=1

    out: dict[str, Any] = {}
    metrics_by_doc: dict[str, Any] = {}


    for doc_id in doc_id_list:
        items: list[dict[str, Any]] =[]
        for term, tf in doc_tf[doc_id].items():
            if in_stop_term(term):
                continue
            # 這個詞在「整本書」出現得普不普遍  (Document Frequency Integer)
            dfi =int(df[term]) 
            # 章內跨頁覆蓋：出現於多少頁
            page_hit_counter = doc_term_page_hit[doc_id].get(term, None)
            page_df = len(page_hit_counter) if page_hit_counter else 0
            pages_in_doc = max(1, int(pages_in_doc_by_docid.get(doc_id, 0)))
            coverage = page_df / pages_in_doc  
            # 透過 log 避免過度壓縮
            heading_hits = 0
            for _pi in (page_hit_counter.keys() if page_hit_counter else []):
                t80 = (pages[_pi].get("text","") or "")[:80].lower()
                if term.lower() in t80:
                    heading_hits += 1
            heading = min(1.0, heading_hits * 0.5)

            idf = math.log((C - dfi + 0.5) / (dfi + 0.5) + 1.0)
            tf_sat = tf / (tf + 3.0)  

            common_ratio = (dfi / C) if C > 0 else 0.0
            common_penalty = max(1e-6, (1.0 - common_ratio)) ** 1.5
            raw = (0.55 * idf + 0.35 * coverage + 0.10 * heading) * tf_sat * common_penalty

            raw_score = raw

            if len(term) >= 7:
                raw_score *= 0.5
            if is_cn(term):
                if term.startswith(("的", "是", "為", "於", "而", "中", "以", "等", "上", "下", "若","當" ,"年", "已")):
                    continue
                # 句尾虛詞/語助
                if term.endswith(("的", "是", "為", "於", "而", "中", "以", "等", "上", "下", "個", "由")):
                    continue
                if re.search(r"(稱為|如下|例如|簡述|舉例|可以|用來|主要|進行|結果|根據|表示)", term):
                    continue
                if re.search(r"(包含下列|形成下列|則形成|下列幾|被應用在|可應用在|可用於|用於|用在)", term):
                    
                    continue
                if len(term) <= 3 and re.search(r"(的|之|於)", term):
                    continue
                if len(term) <= 2 and tf <= 2:
                    continue
            effective_lang = lang
            if effective_lang == "cn":
                if not is_cn(term):
                    continue
            elif lang == "en":
                if is_cn(term):
                    continue

            if raw_score <= 0:
                continue
            if is_cn(term):
                # 出現明顯動詞/功能結尾，高機率是敘述殘片
                if re.search(r"(成|為|得|進行|使用|轉換|形成|開始|進入)$", term):
                    continue

                # 含有介系詞/連接語，但不是長詞 → 殘句
                if re.search(r"(而|則|於|以)", term) and len(term) <= 6:
                    continue
            rep_page_idx: int | None = None
            rep_hits: int = 0
            page_hit_counter = doc_term_page_hit[doc_id].get(term, None)
            if page_hit_counter:
                rep_page_idx, rep_hits = page_hit_counter.most_common(1)[0]

            rep: dict[str, Any] | None = None
            if rep_page_idx is not None:
                pp = pages[rep_page_idx]
                rep ={
                    "page_idx": int(rep_page_idx),
                    "page": pick_page_number(pp),
                    "image": pick_image_path(pp),
                    "hits": int(rep_hits)
                }
            items.append(
                {
                    "term": term,
                    "raw_score": float(raw_score),
                    "score_norm": 0.0,
                    "score": 0.0,  # 保持原欄位存在，最後填 score_norm
                    "tf": int(tf),
                    "df": int(dfi),
                    "representative": rep,
                    "features": {
                        "idf": float(idf),
                        "tf_sat": float(tf_sat),
                        "common_ratio": float(common_ratio),
                        "common_penalty": float(common_penalty),
                        "page_df": int(page_df),
                        "pages_in_doc": int(pages_in_doc),
                        "coverage": float(coverage),
                        "heading": float(heading),
                        "word_count": int(len(str(term).split())),
                        "is_phrase": bool(" " in term),
                    },
                }
            )
        items.sort(key=lambda x:x["raw_score"], reverse=True)
        if lang == "en" and items:
            #  英文章節常有同一個詞既有單字又有片語（如 "force" 和 "force of gravity"）同時出現，嘗試用前面幾十個候選詞來判斷是否有更長的片語比單字更重要，如果有就替換掉單字
            cand2 = items[:min(200, len(items))]
            best_phrase_for: dict[str, dict[str, Any]] = {}

            for it in cand2:
                t = str(it.get("term", "")).strip()
                if " " not in t:
                    continue
                for w in t.split():
                    w = w.strip().lower()
                    if len(w) <= 2:
                        continue
                    cur = best_phrase_for.get(w)
                    if cur is None or float(it.get("raw_score", 0.0) or 0.0) > float(cur.get("raw_score", 0.0) or 0.0):
                        best_phrase_for[w] = it

            replaced: list[dict[str, Any]] = []
            for it in items:
                t = str(it.get("term", "")).strip()
                if not t:
                    continue
                if " " in t:
                    replaced.append(it)
                    continue

                w = t.lower()
                phrase = best_phrase_for.get(w)

                # 沒有對應片語 -> 保留原單字
                if phrase is None:
                    replaced.append(it)
                    continue

                fe_w = it.get("features", {}) or {}
                fe_p = phrase.get("features", {}) or {}

                page_df_w = int(fe_w.get("page_df", 0) or 0)
                page_df_p = int(fe_p.get("page_df", 0) or 0)
                heading_w = float(fe_w.get("heading", 0.0) or 0.0)
                heading_p = float(fe_p.get("heading", 0.0) or 0.0)

                wc_p = int(fe_p.get("word_count", 1) or 1)

                # 只替換成真正片語（>=2 words）
                if wc_p >= 2:
                    # 覆蓋度不差、或更像標題（heading） -> 用片語替換
                    if (page_df_p >= page_df_w) or (heading_p > heading_w) or (heading_p >= 0.5):
                        replaced.append(phrase)
                        continue

                # 不替換 -> 保留原單字
                replaced.append(it)

            # 去重，避免同一個詞既有單字又有片語（如 "force" 和 "force of gravity"）同時出現
            seen_terms: set[str] = set()
            new_items: list[dict[str, Any]] = []
            for it in replaced:
                tt = str(it.get("term", "")).strip()
                if not tt or tt in seen_terms:
                    continue
                seen_terms.add(tt)
                new_items.append(it)

            items = new_items
        # llm_allow_terms_by_chapter 拿來當作「是否啟用 LLM」的開關，如果有提供這個 dict 就啟用 LLM 過濾，沒有提供就完全不過濾（保留原本分數排序 topn）
        if llm_allow_terms_by_chapter is not None and items:
            try:
                # 讓 LLM 從前 M 個候選中挑 topn（你也可以改成 topn*2）
                keep_ids = _llm_select_ids_from_candidates(
                    chapter_ID=doc_id,
                    pages=pages,
                    candidates=items,
                    lang=lang,
                    k=topn,
                )
                if keep_ids:
                    picked = [items[i] for i in keep_ids]

                    # 將 page_df = 0 的候選刪除，避免 LLM 選到沒出現過的詞
                    picked2: list[dict[str, Any]] = []
                    for it in picked:
                        feats = it.get("features", {}) or {}
                        page_df = int(feats.get("page_df", 0) or 0)
                        heading = float(feats.get("heading", 0.0) or 0.0)
                        
                        if int(feats.get("page_df", 0) or 0) <= 0:
                            continue
                        picked2.append(it)  
                    if len(picked2) < topn:
                        seen = set(str(it.get("term", "")) for it in picked2)
                        for it in items:
                            term = str(it.get("term", ""))
                            if term in seen:
                                continue
                            feats = it.get("features", {}) or {}
                            if int(feats.get("page_df", 0) or 0) <= 0:
                                continue
                            picked2.append(it)
                            seen.add(term)
                            if len(picked2) >= topn:
                                break
                    if picked2:
                        picked2.sort(key=lambda x: x["raw_score"], reverse=True)
                        items = picked2
                    else: 
                        print(f"[WARN] LLM 選出來的詞全部 page_df=0，將 fallback 回原本分數 topn")
                        pass
                    # 依原本的 raw_score 再排一次，避免 LLM 選出來的詞順序亂掉
            except Exception as e:
                print(f"[WARN] LLM select 失敗，fallback 回原本分數 topn: {e}")
                # fallback：不動 items

        if items:
            max_score = max(it["raw_score"] for it in items) or 1.0
            for it in items:
                score_norm = float(100.0 * it["raw_score"] / max_score)
                it["score_norm"] = score_norm
                it["score"] = score_norm
            # 計算分位數，嘗試過濾掉過於平坦的章節
            score_sorted = sorted((it["raw_score"] for it in items))  # 升序給 percentile
            p10 = _percentile_score(score_sorted, 0.10)
            p50 = _percentile_score(score_sorted, 0.50)
            p90 = _percentile_score(score_sorted, 0.90)

            mean_score = (sum(score_sorted) / len(score_sorted)) if score_sorted else 0.0

            topk = items[:min(100, len(items))]
            mean_top = sum(it["raw_score"] for it in topk) / max(len(topk), 1)
            metrics_by_doc[doc_id] = {
            "p10": float(p10),
            "p50": float(p50),
            "p90": float(p90),
            "sep_p90_p50": float(p90 / p50) if p50 > 0 else 0.0,
            "head_top10_mean_over_all": float(mean_top / mean_score) if mean_score > 0 else 0.0,
            "tail_p10_over_p90": float(p10 / p90) if p90 > 0 else 0.0,
            "n_terms": int(len(items)),
            }
        else:
            metrics_by_doc[doc_id] = {
            "p10": 0.0, "p50": 0.0, "p90": 0.0,
            "sep_p90_p50": 0.0,
            "head_top10_mean_over_all": 0.0,
            "tail_p10_over_p90": 0.0,
            "n_terms": 0,
            }

        def _norm_key(x: str) -> str:
            x = x.lower().strip()
            x = re.sub(r"[\s\-\_\,\.\:\;\(\)\[\]\{\}]+", "", x)
            x = re.sub(r"[則的為而與或依所於以]", "", x)
            return x

        def _try_add_term_replace_on_overlap(
            selected: list[dict[str, Any]],
            selected_keys: list[str],
            it: dict[str, Any],
            norm_key_fn,
        ) -> bool:
            """
            若重疊：保留更長者；長度同則保留 raw_score 更高者
            """
            t_new = str(it.get("term", ""))
            k_new = norm_key_fn(t_new)

            for j, k_old in enumerate(selected_keys):
                if (k_new in k_old) or (k_old in k_new):
                    t_old = str(selected[j].get("term", ""))
                    if len(t_new) > len(t_old):
                        selected[j] = it
                        selected_keys[j] = k_new
                        return True
                    if len(t_new) == len(t_old):
                        if float(it.get("raw_score", 0.0) or 0.0) > float(selected[j].get("raw_score", 0.0) or 0.0):
                            selected[j] = it
                            selected_keys[j] = k_new
                            return True
                    return False

            selected.append(it)
            selected_keys.append(k_new)
            return True
        
        m = metrics_by_doc.get(doc_id, {}) if isinstance(metrics_by_doc, dict) else {}
        sep = float(m.get("sep_p90_p50", 0.0) or 0.0)
        n_terms = int(m.get("n_terms", 0) or 0)

        eff_topn = int(topn)
        if n_terms <= 6:
            eff_topn = min(eff_topn, 5)
        if sep > 0 and sep < 1.10:
            eff_topn = min(eff_topn, 3)
        elif sep > 0 and sep < 1.25:
            eff_topn = min(eff_topn, 5)

        core_n = min(4, eff_topn)

        selected: list[dict[str, Any]] = []
        selected_keys: list[str] = []

        # Pass 1：先挑 core_n 個（page_df>=2 或 tf>=2）
        for it in items:
            feats = it.get("features", {}) or {}
            page_df = int(feats.get("page_df", 0) or 0)
            tfv = int(it.get("tf", 0) or 0)
            if page_df <= 0:
                continue
            if lang == "en":
                t = str(it.get("term", "")).strip()
                feats = it.get("features", {}) or {}
                heading = float(feats.get("heading", 0.0) or 0.0)
                page_df = int(feats.get("page_df", 0) or 0)

                # 單字且太短 → 只有在很強證據下才允許進榜
                if (" " not in t) and (len(t) <= 4) and not (heading >= 0.5 or page_df >= 4):
                    continue
            if (page_df >= 2) or (tfv >= 2):
                _try_add_term_replace_on_overlap(selected, selected_keys, it, _norm_key)
                if len(selected) >= core_n:
                    break

        # Pass 2：補滿到 eff_topn（仍 page_df>0）
        if len(selected) < eff_topn:
            for it in items:
                feats = it.get("features", {}) or {}
                if page_df <= 0:
                    continue
                # 單字且太短 → 只有在很強證據下才允許進榜
                if lang == "en":
                    t = str(it.get("term", "")).strip()

                    # 單字且太短 
                    if (" " not in t) and (len(t) <= 4) and not (heading >= 0.5 or page_df >= 4):
                        continue
                    wc = int(feats.get("word_count", 1) or 1)

                    if wc == 1:
                        # 不是標題詞、又只在 <=2 頁出現：很高機率是泛詞（counts/law/flee 類）
                        if heading < 1.0 and page_df <= 2:
                            continue

                        # （可選）動詞型單字（ensured / created / using）不當作重點詞
                        if re.search(r"(ed|ing)$", t.lower()):
                            continue

                    # 更乾淨：短單字且 heading==0 且 page_df<=2 直接不收
                    if (" " not in t) and (len(t) <= 4) and (heading <= 0.0) and (page_df <= 2):
                        continue
                _try_add_term_replace_on_overlap(selected, selected_keys, it, _norm_key)
                if len(selected) >= eff_topn:
                    break

        out[doc_id] = selected
    out["__meta__"] = {"metrics": metrics_by_doc}
    return out

def save_outputs(subject: str, data_type: str, jsonl_path: Path, lang: str, tips: dict[str, Any]) -> tuple[Path, Path]:
    """
    輸出到：
    data/<subject>/<pdf_exam_corpus|ppt_exam_corpus>/chapter_tips/<stem>/<stem>.tips.json/.md
    """
    corpus_dir = DATA_DIR / subject / ("pdf_exam_corpus" if data_type == "pdf" else "ppt_exam_corpus")
    out_dir = corpus_dir / "chapter_tips"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = jsonl_path.stem
    save_dir = out_dir / f"{stem}"
    save_dir.mkdir(parents=True, exist_ok=True)

    out_json = save_dir / f"{stem}.{lang}.tips.json"
    out_md = save_dir / f"{stem}.{lang}.tips.md"

    out_json.write_text(json.dumps(tips, ensure_ascii=False, indent=2),encoding="utf-8")

    lines: list[str] = [f"# Chapter tips (with representative pages): {stem}", ""]

    meta = tips.get("__meta__", {}) if isinstance(tips, dict) else {}
    metrics = meta.get("metrics", {}) if isinstance(meta, dict) else {}

    for ch in sorted(k for k in tips.keys() if not str(k).startswith("__")):
        m = metrics.get(ch, {}) if isinstance(metrics, dict) else {}
        lines.append(
            f"## {ch}"
            + (f"  (sep={m.get('sep_p90_p50', 0.0):.2f}, head={m.get('head_top10_mean_over_all', 0.0):.2f}, tail={m.get('tail_p10_over_p90', 0.0):.2f})" if m else "")
        )
        for i, item in enumerate(tips[ch], 1):
            rep = item.get("representative", None) or {}
            rep_page = rep.get("page", None)
            rep_hits = rep.get("hits", None)
            term_show = re.sub(r"\s+", " ", str(item["term"])).strip()
            lines.append(
                f"{i}. {term_show}  (norm={item['score_norm']:.1f}, raw={item['raw_score']:.4f}, tf={item['tf']}, df={item['df']}, page_df={item.get('features', {}).get('page_df', 0)})"
                + (f"  -> rep_page={rep_page}, hits={rep_hits}" if rep_page is not None else "")
            )
        lines.append("")
    out_md.write_text("\n".join(lines),encoding='utf-8')

    return out_json, out_md

def main() -> None:
    args = parse_args()
    if args.ollama_model:
        os.environ["OLLAMA_MODEL"] = args.ollama_model
    os.environ["OLLAMA_BASE_URL"] = args.ollama_base_url
    if args.use_llm:
        chosen_model = resolve_ollama_model(args.ollama_base_url, args.ollama_model)
        os.environ["OLLAMA_MODEL"] = chosen_model
    jsonl_path = Path(args.jsonl_path)

    pages = read_jsonl(jsonl_path)
    llm_allow_terms_by_chapter: dict[str, set[str]] | None = None
    if args.use_llm:
        llm_allow_terms_by_chapter = build_llm_allow_terms_by_chapter(
            subject=args.subject,
            data_type=args.data_type,
            jsonl_path=jsonl_path,
            pages=pages,
            lang=args.lang,
            refresh=args.refresh_llm,
        )
    tips = build_chapter_tips_representatives(
    pages, topn=args.topn, 
    lang=args.lang, llm_allow_terms_by_chapter=llm_allow_terms_by_chapter)

    out_jsonl, out_md = save_outputs(args.subject, args.data_type, jsonl_path, args.lang, tips)
    print(f"[DONE] tips(jsonl) saved: {out_jsonl}")
    print(f"[DONE] tips(md) saved: {out_md}")

if __name__ == "__main__":
    main()