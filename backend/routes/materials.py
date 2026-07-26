from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


@router.get("/api/materials")
def get_materials():
    materials = []

    if not DATA_ROOT.exists():
        return {"materials": materials}

    for subject_dir in DATA_ROOT.iterdir():
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name

        for corpus_dir in subject_dir.iterdir():
            if not corpus_dir.is_dir():
                continue

            corpus_name = corpus_dir.name
            if not corpus_name.endswith("_exam_corpus"):
                continue

            data_type = corpus_name.replace("_exam_corpus", "")
            chapters = []

            merge_index_dir = corpus_dir / "merge_jsonl" / "index"
            if merge_index_dir.exists() and merge_index_dir.is_dir():
                merge_index_subdirs = [p.name for p in merge_index_dir.iterdir() if p.is_dir()]
                if merge_index_subdirs:
                    chapters = sorted(merge_index_subdirs)

            if not chapters:
                index_dir = corpus_dir / "index"
                if index_dir.exists() and index_dir.is_dir():
                    index_subdirs = [p.name for p in index_dir.iterdir() if p.is_dir()]
                    if index_subdirs:
                        chapters = sorted(index_subdirs)

            if not chapters:
                merge_dir = corpus_dir / "merge_jsonl"
                if merge_dir.exists() and merge_dir.is_dir():
                    merge_jsonls = [p.stem for p in merge_dir.glob("*.jsonl")]
                    if merge_jsonls:
                        chapters = sorted(merge_jsonls)

            if not chapters:
                corpus_jsonls = [p.stem for p in corpus_dir.glob("*.jsonl")]
                chapters = sorted(corpus_jsonls)

            materials.append(
                {
                    "subject": subject,
                    "data_type": data_type,
                    "corpus_dir": str(corpus_dir),
                    "chapters": chapters,
                }
            )

    return {"materials": materials}


@router.get("/api/material_viewer_sources")
def get_material_viewer_sources():
    """
    viewer 專用教材清單：
    優先顯示前處理後的原始分章節 jsonl，不優先顯示 merge_jsonl。
    """
    materials = []

    if not DATA_ROOT.exists():
        return {"materials": materials}

    for subject_dir in sorted(DATA_ROOT.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name

        for data_type in ("ppt", "pdf"):
            corpus_dir = subject_dir / f"{data_type}_exam_corpus"
            if not corpus_dir.is_dir():
                continue

            raw_chapters = sorted(
                p.stem for p in corpus_dir.glob("*.jsonl")
                if p.is_file()
            )

            merge_dir = corpus_dir / "merge_jsonl"
            merged_chapters = []
            if merge_dir.is_dir():
                merged_chapters = sorted(
                    p.stem for p in merge_dir.glob("*.jsonl")
                    if p.is_file()
                )

            chapters = raw_chapters if raw_chapters else merged_chapters

            if not chapters:
                continue

            materials.append(
                {
                    "subject": subject,
                    "data_type": data_type,
                    "corpus_dir": str(corpus_dir),
                    "chapters": chapters,
                    "raw_chapters": raw_chapters,
                    "merged_chapters": merged_chapters,
                    "viewer_source": "raw_jsonl" if raw_chapters else "merge_jsonl",
                }
            )

    return {"materials": materials}


def _resolve_material_jsonl(subject: str, data_type: str, chapter: str) -> Path:
    corpus_dir = DATA_ROOT / subject / f"{data_type}_exam_corpus"

    # viewer 優先讀原始分章節 jsonl
    corpus_jsonl = corpus_dir / f"{chapter}.jsonl"
    if corpus_jsonl.is_file():
        return corpus_jsonl

    # 找不到才 fallback 到 merge jsonl
    merge_jsonl = corpus_dir / "merge_jsonl" / f"{chapter}.jsonl"
    if merge_jsonl.is_file():
        return merge_jsonl

    raise HTTPException(
        status_code=404,
        detail=f"material jsonl not found: {corpus_jsonl}",
    )


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read jsonl: {e}")
    return rows


def _page_to_int(value):
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return value


def _normalize_images(value) -> list[str]:
    out: list[str] = []

    def walk(v):
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s)
            return

        if isinstance(v, list):
            for item in v:
                walk(item)
            return

        if isinstance(v, dict):
            for key in ("rel_path", "path", "file_name", "name", "image", "url", "image_url"):
                walk(v.get(key))
            walk(v.get("images"))

    walk(value)

    seen: set[str] = set()
    clean: list[str] = []
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        clean.append(item)
    return clean


def _make_page_key(row: dict) -> dict:
    doc_id = str(row.get("doc_id") or row.get("doc") or row.get("source_doc") or "unknown").strip()
    page = _page_to_int(row.get("page", row.get("page_no", row.get("pageno"))))
    text = str(row.get("text") or "").strip()
    ocr = str(row.get("ocr") or row.get("ocrs") or "").strip()
    images = _normalize_images([row.get("image"), row.get("images")])

    return {
        "doc_id": doc_id,
        "page": page,
        "label": f"{doc_id} / p{page}",
        "has_text": bool(text and text != "這頁是沒有txt的"),
        "has_ocr": bool(ocr),
        "image_count": len(images),
    }


@router.get("/api/material_pages")
def get_material_pages(
    subject: str = Query(..., min_length=1),
    data_type: str = Query(..., min_length=1),
    chapter: str = Query(..., min_length=1),
):
    subject = (subject or "").strip()
    data_type = (data_type or "").strip().lower()
    chapter = (chapter or "").strip()

    if data_type not in {"ppt", "pdf"}:
        raise HTTPException(status_code=400, detail="data_type must be 'ppt' or 'pdf'")

    jsonl_path = _resolve_material_jsonl(subject, data_type, chapter)
    rows = _read_jsonl(jsonl_path)
    page_keys = [_make_page_key(row) for row in rows]

    return {
        "ok": True,
        "subject": subject,
        "data_type": data_type,
        "chapter": chapter,
        "jsonl_path": str(jsonl_path),
        "count": len(page_keys),
        "page_keys": page_keys,
        "first_page": page_keys[0] if page_keys else None,
    }


@router.get("/api/material_page")
def get_material_page(
    subject: str = Query(..., min_length=1),
    data_type: str = Query(..., min_length=1),
    chapter: str = Query(..., min_length=1),
    doc_id: str | None = None,
    page: int | None = None,
):
    subject = (subject or "").strip()
    data_type = (data_type or "").strip().lower()
    chapter = (chapter or "").strip()
    doc_id = (doc_id or "").strip() or None

    if data_type not in {"ppt", "pdf"}:
        raise HTTPException(status_code=400, detail="data_type must be 'ppt' or 'pdf'")

    jsonl_path = _resolve_material_jsonl(subject, data_type, chapter)
    rows = _read_jsonl(jsonl_path)

    if not rows:
        return {
            "ok": False,
            "subject": subject,
            "data_type": data_type,
            "chapter": chapter,
            "jsonl_path": str(jsonl_path),
            "detail": "jsonl is empty",
        }

    target_index = None

    for idx, row in enumerate(rows):
        row_doc_id = str(row.get("doc_id") or row.get("doc") or row.get("source_doc") or "unknown").strip()
        row_page = _page_to_int(row.get("page", row.get("page_no", row.get("pageno"))))

        if doc_id is not None and row_doc_id != doc_id:
            continue

        if page is not None:
            try:
                if int(row_page) != int(page):
                    continue
            except Exception:
                if str(row_page) != str(page):
                    continue

        target_index = idx
        break

    if target_index is None:
        raise HTTPException(status_code=404, detail="page not found")

    row = rows[target_index]
    page_keys = [_make_page_key(r) for r in rows]

    row_doc_id = str(row.get("doc_id") or row.get("doc") or row.get("source_doc") or "unknown").strip()
    row_page = _page_to_int(row.get("page", row.get("page_no", row.get("pageno"))))
    images = _normalize_images([row.get("image"), row.get("images")])

    return {
        "ok": True,
        "subject": subject,
        "data_type": data_type,
        "chapter": chapter,
        "jsonl_path": str(jsonl_path),
        "index": target_index,
        "total": len(rows),
        "doc_id": row_doc_id,
        "page": row_page,
        "text": row.get("text") or "",
        "ocr": row.get("ocr") or row.get("ocrs") or "",
        "images": images,
        "page_keys": page_keys,
        "prev": page_keys[target_index - 1] if target_index > 0 else None,
        "next": page_keys[target_index + 1] if target_index < len(rows) - 1 else None,
    }
