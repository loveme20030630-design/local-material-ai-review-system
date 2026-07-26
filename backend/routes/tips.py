from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from urllib.parse import quote
from pydantic import BaseModel

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

MAKE_TIPS_SCRIPT = SCRIPTS_DIR / "make_chapter_tips.py"


class RunTipsRequest(BaseModel):
    subject: str
    data_type: str
    chapter: str
    topn: int = 8
    lang: str = "cn"
    use_llm: bool = True
    refresh_llm: bool = False


def _resolve_merge_jsonl_path(subject: str, data_type: str, chapter: str) -> Path:
    corpus_dir = DATA_ROOT / subject / f"{data_type}_exam_corpus"
    return corpus_dir / "merge_jsonl" / f"{chapter}.jsonl"


def _resolve_tips_paths(subject: str, data_type: str, chapter: str, lang: str) -> tuple[Path, Path]:
    corpus_dir = DATA_ROOT / subject / f"{data_type}_exam_corpus"
    tips_dir = corpus_dir / "chapter_tips" / chapter
    return tips_dir / f"{chapter}.{lang}.tips.json", tips_dir / f"{chapter}.{lang}.tips.md"


def _resolve_llm_allow_path(subject: str, data_type: str, chapter: str, lang: str) -> Path:
    corpus_dir = DATA_ROOT / subject / f"{data_type}_exam_corpus"
    return corpus_dir / "chapter_tips" / chapter / f"{chapter}.{lang}.llm_allow_terms.json"


def _tips_is_empty(tips_json: dict | None) -> bool:
    if not isinstance(tips_json, dict):
        return True

    chapter_keys = [k for k in tips_json.keys() if not str(k).startswith("__")]
    if not chapter_keys:
        return True

    for key in chapter_keys:
        items = tips_json.get(key, [])
        if isinstance(items, list) and len(items) > 0:
            return False

    return True


def _load_llm_allow_as_tips(subject: str, data_type: str, chapter: str, lang: str) -> tuple[dict | None, Path]:
    allow_path = _resolve_llm_allow_path(subject, data_type, chapter, lang)
    if not allow_path.is_file():
        return None, allow_path

    try:
        allow_data = json.loads(allow_path.read_text(encoding="utf-8"))
    except Exception:
        return None, allow_path

    if not isinstance(allow_data, dict):
        return None, allow_path

    fallback: dict = {}
    for chapter_key, terms in allow_data.items():
        if str(chapter_key).startswith("__"):
            continue
        if not isinstance(terms, list):
            continue

        clean_terms: list[str] = []
        seen: set[str] = set()
        for term in terms:
            t = str(term or "").strip()
            if not t or t in seen:
                continue
            seen.add(t)
            clean_terms.append(t)

        fallback[str(chapter_key)] = [
            {
                "term": term,
                "raw_score": 0.0,
                "score_norm": 0.0,
                "score": 0.0,
                "tf": 0,
                "df": 0,
                "representative": {
                    "page": None,
                    "image": "",
                    "hits": 0,
                },
                "features": {},
                "source": "llm_allow_fallback",
            }
            for term in clean_terms
        ]

    if _tips_is_empty(fallback):
        return None, allow_path

    fallback["__meta__"] = {
        "fallback": "llm_allow_terms",
        "reason": "tips_json_empty",
        "llm_allow_terms_path": str(allow_path),
    }
    return fallback, allow_path


def _apply_llm_allow_fallback_if_empty(
    tips_json: dict | None,
    subject: str,
    data_type: str,
    chapter: str,
    lang: str,
) -> tuple[dict | None, Path | None, bool]:
    if not _tips_is_empty(tips_json):
        return tips_json, None, False

    fallback_tips, allow_path = _load_llm_allow_as_tips(subject, data_type, chapter, lang)
    if fallback_tips is None:
        return tips_json, allow_path, False

    return fallback_tips, allow_path, True

def _resolve_tips_dir(subject: str, data_type: str) -> Path:
    corpus_dir = DATA_ROOT / subject / f"{data_type}_exam_corpus"
    return corpus_dir / "chapter_tips"


def _to_static_url(path_under_data: Path) -> str:
    rel = path_under_data.relative_to(DATA_ROOT).as_posix()
    return "/static/" + quote(rel, safe="/._-~()")

@router.get("/api/resolve_image_url")
def resolve_image_url(image: str = Query(..., min_length=1)):
    raw = (image or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="image is required")

    clean = raw.replace("\\", "/").lstrip("./")
    if clean.startswith("data/"):
        clean = clean[5:]

    # 1) 先當成 data/ 下的相對路徑直接找
    direct = (DATA_ROOT / clean).resolve()
    if DATA_ROOT in direct.parents and direct.is_file():
        return {"ok": True, "url": _to_static_url(direct)}

    # 2) 找不到就用檔名 fallback 掃描
    name = Path(clean).name
    if not name:
        raise HTTPException(status_code=404, detail="Not Found")

    matches = [p for p in DATA_ROOT.rglob(name) if p.is_file()]
    if not matches:
        raise HTTPException(status_code=404, detail="Not Found")

    # 3) 多個重名時優先 raw_img
    raw_img_matches = [p for p in matches if "raw_img" in p.as_posix()]
    chosen = raw_img_matches[0] if raw_img_matches else matches[0]

    return {
        "ok": True,
        "url": _to_static_url(chosen),
    }

@router.get("/api/tips_candidates")
def get_tips_candidates():
    candidates: list[dict] = []

    if not DATA_ROOT.exists():
        return {"candidates": candidates}

    for subject_dir in sorted(DATA_ROOT.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name

        for data_type in ("ppt", "pdf"):
            corpus_dir = subject_dir / f"{data_type}_exam_corpus"
            merge_dir = corpus_dir / "merge_jsonl"

            if not merge_dir.is_dir():
                continue

            chapters = sorted([p.stem for p in merge_dir.glob("*.jsonl") if p.is_file()])
            if not chapters:
                continue

            candidates.append(
                {
                    "subject": subject,
                    "data_type": data_type,
                    "chapters": chapters,
                    "merge_dir": str(merge_dir),
                }
            )

    return {"candidates": candidates}


@router.post("/api/run_tips")
def run_tips(req: RunTipsRequest):
    subject = (req.subject or "").strip()
    data_type = (req.data_type or "").strip().lower()
    chapter = (req.chapter or "").strip()
    lang = (req.lang or "").strip().lower()

    if not subject:
        raise HTTPException(status_code=400, detail="subject is required")
    if data_type not in {"ppt", "pdf"}:
        raise HTTPException(status_code=400, detail="data_type must be 'ppt' or 'pdf'")
    if not chapter:
        raise HTTPException(status_code=400, detail="chapter is required")
    if lang not in {"cn", "en"}:
        raise HTTPException(status_code=400, detail="lang must be 'cn' or 'en'")
    if req.topn <= 0:
        raise HTTPException(status_code=400, detail="topn must be > 0")
    if not MAKE_TIPS_SCRIPT.exists():
        raise HTTPException(status_code=404, detail=f"make_chapter_tips.py not found: {MAKE_TIPS_SCRIPT}")

    jsonl_path = _resolve_merge_jsonl_path(subject, data_type, chapter)
    if not jsonl_path.is_file():
        raise HTTPException(status_code=404, detail=f"merge jsonl not found: {jsonl_path}")

    cmd = [
        sys.executable,
        str(MAKE_TIPS_SCRIPT),
        subject,
        data_type,
        lang,
        str(jsonl_path),
        str(req.topn),
    ]

    if req.use_llm:
        cmd.append("--use_llm")
    if req.refresh_llm:
        cmd.append("--refresh_llm")

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    tips_json_path, tips_md_path = _resolve_tips_paths(subject, data_type, chapter, lang)
    tips_json = None
    tips_md = None

    if tips_json_path.is_file():
        try:
            tips_json = json.loads(tips_json_path.read_text(encoding="utf-8"))
        except Exception:
            tips_json = None

    if tips_md_path.is_file():
        try:
            tips_md = tips_md_path.read_text(encoding="utf-8")
        except Exception:
            tips_md = None

    tips_json, llm_allow_terms_path, used_llm_allow_fallback = _apply_llm_allow_fallback_if_empty(
        tips_json, subject, data_type, chapter, lang
    )

    return {
        "ok": result.returncode == 0 and tips_json is not None,
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "subject": subject,
        "data_type": data_type,
        "chapter": chapter,
        "jsonl_path": str(jsonl_path),
        "tips_json_path": str(tips_json_path) if tips_json_path.exists() else None,
        "tips_md_path": str(tips_md_path) if tips_md_path.exists() else None,
        "llm_allow_terms_path": str(llm_allow_terms_path) if llm_allow_terms_path else None,
        "used_llm_allow_fallback": used_llm_allow_fallback,
        "tips": tips_json,
        "tips_md": tips_md,
    }

@router.get("/api/generated_tips")
def get_generated_tips(subject: str, data_type: str, lang: str = "cn"):
    subject = (subject or "").strip()
    data_type = (data_type or "").strip().lower()
    lang = (lang or "").strip().lower()

    if not subject:
        raise HTTPException(status_code=400, detail="subject is required")
    if data_type not in {"ppt", "pdf"}:
        raise HTTPException(status_code=400, detail="data_type must be 'ppt' or 'pdf'")
    if lang not in {"cn", "en"}:
        raise HTTPException(status_code=400, detail="lang must be 'cn' or 'en'")

    tips_dir = _resolve_tips_dir(subject, data_type)
    items: list[dict] = []

    if not tips_dir.is_dir():
        return {
            "subject": subject,
            "data_type": data_type,
            "lang": lang,
            "items": items,
        }

    for chapter_dir in sorted([p for p in tips_dir.iterdir() if p.is_dir()]):
        chapter = chapter_dir.name
        tips_json_path = chapter_dir / f"{chapter}.{lang}.tips.json"
        tips_md_path = chapter_dir / f"{chapter}.{lang}.tips.md"

        if not tips_json_path.is_file():
            continue

        items.append(
            {
                "chapter": chapter,
                "lang": lang,
                "tips_json_path": str(tips_json_path),
                "tips_md_path": str(tips_md_path) if tips_md_path.is_file() else None,
            }
        )

    return {
        "subject": subject,
        "data_type": data_type,
        "lang": lang,
        "items": items,
    }

@router.get("/api/load_generated_tips")
def load_generated_tips(subject: str, data_type: str, chapter: str, lang: str = "cn"):
    subject = (subject or "").strip()
    data_type = (data_type or "").strip().lower()
    chapter = (chapter or "").strip()
    lang = (lang or "").strip().lower()

    if not subject:
        raise HTTPException(status_code=400, detail="subject is required")
    if data_type not in {"ppt", "pdf"}:
        raise HTTPException(status_code=400, detail="data_type must be 'ppt' or 'pdf'")
    if not chapter:
        raise HTTPException(status_code=400, detail="chapter is required")
    if lang not in {"cn", "en"}:
        raise HTTPException(status_code=400, detail="lang must be 'cn' or 'en'")

    tips_json_path, tips_md_path = _resolve_tips_paths(subject, data_type, chapter, lang)

    if not tips_json_path.is_file():
        raise HTTPException(status_code=404, detail=f"tips json not found: {tips_json_path}")

    try:
        tips_json = json.loads(tips_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read tips json: {e}")

    tips_md = None
    if tips_md_path.is_file():
        try:
            tips_md = tips_md_path.read_text(encoding="utf-8")
        except Exception:
            tips_md = None

    tips_json, llm_allow_terms_path, used_llm_allow_fallback = _apply_llm_allow_fallback_if_empty(
        tips_json, subject, data_type, chapter, lang
    )

    return {
        "ok": True,
        "subject": subject,
        "data_type": data_type,
        "chapter": chapter,
        "lang": lang,
        "tips_json_path": str(tips_json_path),
        "tips_md_path": str(tips_md_path) if tips_md_path.is_file() else None,
        "tips": tips_json,
        "tips_md": tips_md,
    }