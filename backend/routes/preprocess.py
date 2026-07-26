from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_ROOT = PROJECT_ROOT / "data"

EXTRACT_SCRIPT = SCRIPTS_DIR / "extract_text_and_image_and_ocr.py"
MAKE_JSONL_SCRIPT = SCRIPTS_DIR / "make_chapter_jsonl.py"


class RunPreprocessRequest(BaseModel):
    folder_name: str
    data_type: str


@router.get("/api/preprocess_candidates")
def get_preprocess_candidates():
    candidates = []

    if not DATA_ROOT.exists():
        return {"candidates": candidates}

    for subject_dir in DATA_ROOT.iterdir():
        if not subject_dir.is_dir():
            continue

        folder_name = subject_dir.name

        for data_type in ("ppt", "pdf"):
            material_dir = subject_dir / data_type
            if not material_dir.exists() or not material_dir.is_dir():
                continue

            valid_exts = (".ppt", ".pptx") if data_type == "ppt" else (".pdf",)
            files = sorted(
                [p.name for p in material_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
            )

            if not files:
                continue

            candidates.append(
                {
                    "folder_name": folder_name,
                    "data_type": data_type,
                    "material_dir": str(material_dir),
                    "file_count": len(files),
                    "files": files,
                }
            )

    return {"candidates": candidates}


def run_one_command(cmd: list[str]) -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    return {
        "command": ["hidden"],
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "ok": result.returncode == 0,
    }

@router.post("/api/run_preprocess")
def run_preprocess(req: RunPreprocessRequest):
    folder_name = (req.folder_name or "").strip()
    data_type = (req.data_type or "").strip().lower()

    if not folder_name:
        raise HTTPException(status_code=400, detail="folder_name is required")
    if data_type not in {"ppt", "pdf"}:
        raise HTTPException(status_code=400, detail="data_type must be 'ppt' or 'pdf'")

    material_dir = DATA_ROOT / folder_name / data_type
    if not material_dir.exists():
        raise HTTPException(status_code=404, detail=f"material dir not found: {material_dir}")

    if not EXTRACT_SCRIPT.exists():
        raise HTTPException(status_code=404, detail=f"extract script not found: {EXTRACT_SCRIPT}")
    if not MAKE_JSONL_SCRIPT.exists():
        raise HTTPException(status_code=404, detail=f"make_chapter_jsonl script not found: {MAKE_JSONL_SCRIPT}")

    extract_cmd = [
        sys.executable,
        str(EXTRACT_SCRIPT),
        folder_name,
        data_type,
    ]
    make_jsonl_cmd = [
        sys.executable,
        str(MAKE_JSONL_SCRIPT),
        folder_name,
        data_type,
    ]

    extract_result = run_one_command(extract_cmd)
    if not extract_result["ok"]:
        return {
            "ok": False,
            "stage": "extract",
            "folder_name": folder_name,
            "data_type": data_type,
            "material_dir": str(material_dir),
            "extract": extract_result,
        }

    make_jsonl_result = run_one_command(make_jsonl_cmd)
    if not make_jsonl_result["ok"]:
        return {
            "ok": False,
            "stage": "make_chapter_jsonl",
            "folder_name": folder_name,
            "data_type": data_type,
            "material_dir": str(material_dir),
            "extract": extract_result,
            "make_chapter_jsonl": make_jsonl_result,
        }

    return {
        "ok": True,
        "stage": "done",
        "folder_name": folder_name,
        "data_type": data_type,
        "material_dir": str(material_dir),
        "extract": extract_result,
        "make_chapter_jsonl": make_jsonl_result,
    }