from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "exams"

MAKE_EXAM_SCRIPT = SCRIPTS_DIR / "make_exam_json.py"


class RunExamRequest(BaseModel):
    subject: str
    data_type: str
    chapter: str
    n: int = 10
    k: int = 5
    lang: str = "cn"
    strict_quote: bool = False


def _resolve_merge_index_dir(subject: str, data_type: str, chapter: str) -> Path:
    corpus_dir = DATA_ROOT / subject / f"{data_type}_exam_corpus"
    return corpus_dir / "merge_jsonl" / "index" / chapter


def _resolve_exam_output_dir(subject: str, data_type: str, chapter: str) -> Path:
    return ARTIFACTS_ROOT / subject / data_type / chapter


def _pick_latest_exam_json(out_dir: Path) -> Path | None:
    if not out_dir.is_dir():
        return None

    files = sorted(
        [p for p in out_dir.glob("exam_*.json") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None

def _resolve_exam_base_dir(subject: str, data_type: str) -> Path:
    return PROJECT_ROOT / "artifacts" / "exams" / subject / data_type

@router.get("/api/exam_candidates")
def get_exam_candidates():
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


@router.post("/api/run_exam")
def run_exam(req: RunExamRequest):
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
    if lang not in {"cn", "en", "auto"}:
        raise HTTPException(status_code=400, detail="lang must be one of: cn, en, auto")
    if req.n <= 0:
        raise HTTPException(status_code=400, detail="n must be > 0")
    if req.k <= 0:
        raise HTTPException(status_code=400, detail="k must be > 0")

    if not MAKE_EXAM_SCRIPT.exists():
        raise HTTPException(status_code=404, detail=f"make_exam_json.py not found: {MAKE_EXAM_SCRIPT}")

    merge_index_dir = _resolve_merge_index_dir(subject, data_type, chapter)
    if not merge_index_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"merge index dir not found: {merge_index_dir}"
        )

    cmd = [
        sys.executable,
        str(MAKE_EXAM_SCRIPT),
        subject,
        data_type,
        chapter,
        "--n",
        str(req.n),
        "--k",
        str(req.k),
        "--lang",
        lang,
    ]

    if req.strict_quote:
        cmd.append("--strict_quote")

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    out_dir = _resolve_exam_output_dir(subject, data_type, chapter)
    latest_exam = _pick_latest_exam_json(out_dir)

    exam_payload = None
    if latest_exam is not None and latest_exam.is_file():
        try:
            exam_payload = json.loads(latest_exam.read_text(encoding="utf-8"))
        except Exception:
            exam_payload = None

    return {
        "ok": result.returncode == 0 and exam_payload is not None,
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "subject": subject,
        "data_type": data_type,
        "chapter": chapter,
        "output_dir": str(out_dir),
        "exam_json_path": str(latest_exam) if latest_exam else None,
        "exam": exam_payload,
    }

@router.get("/api/generated_exams")
def get_generated_exams(subject: str, data_type: str):
    subject = (subject or "").strip()
    data_type = (data_type or "").strip().lower()

    if not subject:
        raise HTTPException(status_code=400, detail="subject is required")
    if data_type not in {"ppt", "pdf"}:
        raise HTTPException(status_code=400, detail="data_type must be 'ppt' or 'pdf'")

    base_dir = _resolve_exam_base_dir(subject, data_type)
    items = []

    if not base_dir.is_dir():
        return {"subject": subject, "data_type": data_type, "items": items}

    for chapter_dir in sorted([p for p in base_dir.iterdir() if p.is_dir()]):
        chapter = chapter_dir.name
        json_files = sorted(chapter_dir.glob("*.json"))

        for jf in json_files:
            items.append({
                "chapter": chapter,
                "file_name": jf.name,
                "exam_json_path": str(jf),
            })

    return {
        "subject": subject,
        "data_type": data_type,
        "items": items,
    }



class DeleteGeneratedExamRequest(BaseModel):
    subject: str
    data_type: str
    chapter: str
    file_name: str


@router.post("/api/delete_generated_exam")
def delete_generated_exam(req: DeleteGeneratedExamRequest):
    subject = (req.subject or "").strip()
    data_type = (req.data_type or "").strip().lower()
    chapter = (req.chapter or "").strip()
    file_name = Path((req.file_name or "").strip()).name

    if not subject:
        raise HTTPException(status_code=400, detail="subject is required")
    if data_type not in {"ppt", "pdf"}:
        raise HTTPException(status_code=400, detail="data_type must be 'ppt' or 'pdf'")
    if not chapter:
        raise HTTPException(status_code=400, detail="chapter is required")
    if not file_name:
        raise HTTPException(status_code=400, detail="file_name is required")
    if Path(file_name).suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="file_name must be a .json file")

    exam_path = _resolve_exam_base_dir(subject, data_type) / chapter / file_name
    if not exam_path.is_file():
        raise HTTPException(status_code=404, detail=f"exam json not found: {exam_path}")

    try:
        exam_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to delete exam json: {e}")

    return {
        "ok": True,
        "subject": subject,
        "data_type": data_type,
        "chapter": chapter,
        "file_name": file_name,
        "deleted_exam_json_path": str(exam_path),
    }


@router.get("/api/load_generated_exam")
def load_generated_exam(subject: str, data_type: str, file_name: str, chapter: str):
    subject = (subject or "").strip()
    data_type = (data_type or "").strip().lower()
    file_name = (file_name or "").strip()
    chapter = (chapter or "").strip()

    if not subject:
        raise HTTPException(status_code=400, detail="subject is required")
    if data_type not in {"ppt", "pdf"}:
        raise HTTPException(status_code=400, detail="data_type must be 'ppt' or 'pdf'")
    if not chapter:
        raise HTTPException(status_code=400, detail="chapter is required")
    if not file_name:
        raise HTTPException(status_code=400, detail="file_name is required")

    exam_path = _resolve_exam_base_dir(subject, data_type) / chapter / file_name
    if not exam_path.is_file():
        raise HTTPException(status_code=404, detail=f"exam json not found: {exam_path}")

    try:
        exam = json.loads(exam_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read exam json: {e}")

    return {
        "ok": True,
        "subject": subject,
        "data_type": data_type,
        "chapter": chapter,
        "file_name": file_name,
        "exam_json_path": str(exam_path),
        "exam": exam,
    }