from __future__ import annotations

import json
import subprocess
import sys
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

SEARCH_SCRIPT = SCRIPTS_DIR / "search_index_qa.py"


class RunSearchRequest(BaseModel):
    subject: str
    data_type: str
    chapter: str
    query: str
    k: int = 5
    qa: bool = False
    ollama_model: str | None = None


def _resolve_merge_index_dir(subject: str, data_type: str, chapter: str) -> Path:
    corpus_dir = DATA_ROOT / subject / f"{data_type}_exam_corpus"
    return corpus_dir / "merge_jsonl" / "index" / chapter

def _resolve_tips_json_path(subject: str, data_type: str, chapter: str, lang: str) -> Path:
    corpus_dir = DATA_ROOT / subject / f"{data_type}_exam_corpus"
    return corpus_dir / "chapter_tips" / chapter / f"{chapter}.{lang}.tips.json"


def _resolve_llm_allow_terms_path(subject: str, data_type: str, chapter: str, lang: str) -> Path:
    corpus_dir = DATA_ROOT / subject / f"{data_type}_exam_corpus"
    return corpus_dir / "chapter_tips" / chapter / f"{chapter}.{lang}.llm_allow_terms.json"


def _load_llm_allow_terms(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}

    out: dict[str, list[str]] = {}
    for chapter_key, terms in raw.items():
        key = str(chapter_key or "").strip()
        if not key or key.startswith("__"):
            continue
        if not isinstance(terms, list):
            continue

        seen: set[str] = set()
        clean_terms: list[str] = []
        for item in terms:
            term = str(item or "").strip()
            if not term or term in seen:
                continue
            seen.add(term)
            clean_terms.append(term)

        if clean_terms:
            out[key] = clean_terms

    return out


def _pick_keywords_from_llm_allow(
    allow_terms: dict[str, list[str]],
    tip_chapter: str | None,
    limit: int,
) -> tuple[list[str], str | None, list[str]]:
    chapter_keys = list(allow_terms.keys())
    if not chapter_keys:
        return [], None, []

    if tip_chapter and tip_chapter in allow_terms:
        selected_tip_chapter = tip_chapter
    else:
        selected_tip_chapter = chapter_keys[0]

    return chapter_keys, selected_tip_chapter, allow_terms.get(selected_tip_chapter, [])[:limit]


def _safe_parse_images(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []

    out: list[str] = []
    for m in re.finditer(r"""['"]([^'"]+\.(?:png|jpg|jpeg|gif|webp|bmp|svg))['"]""", raw, re.IGNORECASE):
        p = str(m.group(1) or "").strip()
        if p:
            out.append(p)
    return out


def _parse_search_stdout(stdout: str) -> list[dict]:
    text = str(stdout or "")
    if not text.strip():
        return []

    lines = text.splitlines()
    results: list[dict] = []
    cur: dict | None = None
    mode: str | None = None
    text_lines: list[str] = []
    ocr_lines: list[str] = []

    def flush_current():
        nonlocal cur, mode, text_lines, ocr_lines, results
        if not cur:
            return
        cur["text"] = "\n".join(text_lines).strip()
        cur["ocr"] = "\n".join(ocr_lines).strip()
        cur.setdefault("images", [])
        results.append(cur)
        cur = None
        mode = None
        text_lines = []
        ocr_lines = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        m_rank = re.match(r"^#(\d+)\s+score\s*=\s*([0-9.]+)", line.strip(), re.IGNORECASE)
        if m_rank:
            flush_current()
            cur = {
                "rank": int(m_rank.group(1)),
                "score": float(m_rank.group(2)),
                "doc_id": "",
                "page": None,
                "images": [],
                "ocr": "",
                "text": "",
            }
            continue

        if cur is None:
            continue

        m_doc = re.match(r"^doc_id\s*=\s*(.*?)\s+page\s*=\s*(.+?)\s*$", line.strip(), re.IGNORECASE)
        if m_doc:
            cur["doc_id"] = str(m_doc.group(1) or "").strip()
            page_raw = str(m_doc.group(2) or "").strip()
            try:
                cur["page"] = int(page_raw)
            except Exception:
                cur["page"] = page_raw or None
            continue

        m_img = re.match(r"^images\s*=\s*(.+)$", line.strip(), re.IGNORECASE)
        if m_img:
            cur["images"] = _safe_parse_images(m_img.group(1))
            continue

        if re.match(r"^OCR\s*:\s*$", line.strip(), re.IGNORECASE):
            mode = "ocr"
            continue

        if re.match(r"^TEXT\s*:\s*$", line.strip(), re.IGNORECASE):
            mode = "text"
            continue

        if re.fullmatch(r"-{20,}", line.strip()):
            flush_current()
            continue

        if mode == "ocr":
            ocr_lines.append(line)
        elif mode == "text":
            text_lines.append(line)

    flush_current()
    return results


@router.get("/api/search_candidates")
def get_search_candidates():
    candidates: list[dict] = []

    if not DATA_ROOT.exists():
        return {"candidates": candidates}

    for subject_dir in sorted(DATA_ROOT.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name

        for data_type in ("ppt", "pdf"):
            corpus_dir = subject_dir / f"{data_type}_exam_corpus"
            merge_index_dir = corpus_dir / "merge_jsonl" / "index"

            if not merge_index_dir.is_dir():
                continue

            chapters = sorted([p.name for p in merge_index_dir.iterdir() if p.is_dir()])
            if not chapters:
                continue

            candidates.append(
                {
                    "subject": subject,
                    "data_type": data_type,
                    "chapters": chapters,
                    "merge_index_dir": str(merge_index_dir),
                }
            )

    return {"candidates": candidates}


@router.get("/api/search_keywords")
def get_search_keywords(
    subject: str,
    data_type: str,
    chapter: str,
    tip_chapter: str | None = None,
    limit: int = 8,
    lang: str = "cn",
):
    subject = (subject or "").strip()
    data_type = (data_type or "").strip().lower()
    chapter = (chapter or "").strip()
    tip_chapter = (tip_chapter or "").strip() or None
    lang = (lang or "").strip().lower()

    if not subject:
        raise HTTPException(status_code=400, detail="subject is required")
    if data_type not in {"ppt", "pdf"}:
        raise HTTPException(status_code=400, detail="data_type must be 'ppt' or 'pdf'")
    if not chapter:
        raise HTTPException(status_code=400, detail="chapter is required")
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be > 0")
    if lang not in {"cn", "en"}:
        raise HTTPException(status_code=400, detail="lang must be 'cn' or 'en'")

    tips_json_path = _resolve_tips_json_path(subject, data_type, chapter, lang)
    llm_allow_terms_path = _resolve_llm_allow_terms_path(subject, data_type, chapter, lang)

    def fallback_from_llm_allow(ok_if_empty_tips: bool) -> dict | None:
        allow_terms = _load_llm_allow_terms(llm_allow_terms_path)
        chapter_keys, selected_tip_chapter, keywords = _pick_keywords_from_llm_allow(
            allow_terms,
            tip_chapter,
            limit,
        )
        if not keywords:
            return None

        return {
            "ok": True if ok_if_empty_tips else False,
            "subject": subject,
            "data_type": data_type,
            "chapter": chapter,
            "lang": lang,
            "tips_json_path": str(tips_json_path),
            "llm_allow_terms_path": str(llm_allow_terms_path),
            "used_llm_allow_fallback": True,
            "chapter_keys": chapter_keys,
            "selected_tip_chapter": selected_tip_chapter,
            "keywords": keywords,
        }

    if not tips_json_path.is_file():
        fallback = fallback_from_llm_allow(ok_if_empty_tips=False)
        if fallback is not None:
            return fallback

        return {
            "ok": False,
            "subject": subject,
            "data_type": data_type,
            "chapter": chapter,
            "lang": lang,
            "tips_json_path": str(tips_json_path),
            "llm_allow_terms_path": str(llm_allow_terms_path),
            "used_llm_allow_fallback": False,
            "chapter_keys": [],
            "selected_tip_chapter": None,
            "keywords": [],
        }

    try:
        data = json.loads(tips_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read tips json: {e}")

    chapter_keys = [k for k in data.keys() if not str(k).startswith("__")]

    if not chapter_keys:
        fallback = fallback_from_llm_allow(ok_if_empty_tips=True)
        if fallback is not None:
            return fallback

        return {
            "ok": True,
            "subject": subject,
            "data_type": data_type,
            "chapter": chapter,
            "lang": lang,
            "tips_json_path": str(tips_json_path),
            "llm_allow_terms_path": str(llm_allow_terms_path),
            "used_llm_allow_fallback": False,
            "chapter_keys": [],
            "selected_tip_chapter": None,
            "keywords": [],
        }

    if tip_chapter and tip_chapter in chapter_keys:
        selected_tip_chapter = tip_chapter
    else:
        selected_tip_chapter = chapter_keys[0]

    items = data.get(selected_tip_chapter, [])
    if not isinstance(items, list):
        items = []

    keywords: list[str] = []
    seen: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        term = str(item.get("term", "")).strip()
        if not term:
            continue
        if term in seen:
            continue

        seen.add(term)
        keywords.append(term)

        if len(keywords) >= limit:
            break

    if not keywords:
        fallback = fallback_from_llm_allow(ok_if_empty_tips=True)
        if fallback is not None:
            return fallback

    return {
        "ok": True,
        "subject": subject,
        "data_type": data_type,
        "chapter": chapter,
        "lang": lang,
        "tips_json_path": str(tips_json_path),
        "llm_allow_terms_path": str(llm_allow_terms_path),
        "used_llm_allow_fallback": False,
        "chapter_keys": chapter_keys,
        "selected_tip_chapter": selected_tip_chapter,
        "keywords": keywords,
    }


@router.post("/api/run_search")
def run_search(req: RunSearchRequest):
    subject = (req.subject or "").strip()
    data_type = (req.data_type or "").strip().lower()
    chapter = (req.chapter or "").strip()
    query = (req.query or "").strip()
    ollama_model = (req.ollama_model or "").strip()

    if not subject:
        raise HTTPException(status_code=400, detail="subject is required")
    if data_type not in {"ppt", "pdf"}:
        raise HTTPException(status_code=400, detail="data_type must be 'ppt' or 'pdf'")
    if not chapter:
        raise HTTPException(status_code=400, detail="chapter is required")
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    if req.k <= 0:
        raise HTTPException(status_code=400, detail="k must be > 0")
    if not SEARCH_SCRIPT.exists():
        raise HTTPException(status_code=404, detail=f"search script not found: {SEARCH_SCRIPT}")

    merge_index_dir = _resolve_merge_index_dir(subject, data_type, chapter)
    if not merge_index_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"merge index dir not found: {merge_index_dir}")

    cmd = [
        sys.executable,
        str(SEARCH_SCRIPT),
        subject,
        data_type,
        chapter,
        query,
        str(req.k),
    ]

    if req.qa:
        cmd.append("--qa")

    if ollama_model:
        cmd.extend(["--ollama_model", ollama_model])

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    parsed_results = _parse_search_stdout(result.stdout) if (result.returncode == 0 and not req.qa) else []
    return {
        "ok": result.returncode == 0,
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "subject": subject,
        "data_type": data_type,
        "chapter": chapter,
        "query": query,
        "qa": req.qa,
        "parsed_results": parsed_results,
    }