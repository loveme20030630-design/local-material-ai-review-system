from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import re   

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MERGE_SCRIPT = SCRIPTS_DIR / "merge_jsonl.py"
BUILD_INDEX_SCRIPT = SCRIPTS_DIR / "build_index.py"


class RunMergeRequest(BaseModel):
    folder_name: str
    data_type: str
    range_str: str | None = None
    pick_str: str | None = None


def _resolve_corpus_dir(folder_name: str, data_type: str) -> Path:
    return DATA_ROOT / folder_name / f"{data_type}_exam_corpus"


def _extract_merged_jsonl_path(stdout: str) -> Path | None:
    """
    從 merge_jsonl.py stdout 中抓最後一個 .jsonl 路徑
    [DONE]合併完成，輸出檔案: D:\\...\\xxx.jsonl
    """
    if not stdout:
        return None

    matches = re.findall(r"([A-Za-z]:\\[^\r\n]*?\.jsonl)", stdout)
    if matches:
        return Path(matches[-1].strip())

    return None


def run_subprocess_utf8(cmd: list[str], input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    """
    執行子程序並統一使用 UTF-8 收發文字。
    Windows 上若子程序 stdout 走系統預設編碼，前端可能看到中文亂碼；
    這裡強制 Python 子程序以 UTF-8 輸出，避免後端解碼時產生亂碼。
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["MERGE_JSONL_API_MODE"] = "1"

    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )

@router.get("/api/merge_subjects")
def get_merge_subjects():
    subjects: list[dict] = []

    if not DATA_ROOT.exists():
        return {"subjects": subjects}

    for subject_dir in sorted(DATA_ROOT.iterdir()):
        if not subject_dir.is_dir():
            continue

        folder_name = subject_dir.name
        available_types: list[str] = []

        for data_type in ("ppt", "pdf"):
            corpus_dir = _resolve_corpus_dir(folder_name, data_type)
            if not corpus_dir.is_dir():
                continue

            jsonls = [p for p in corpus_dir.glob("*.jsonl") if p.is_file()]
            if jsonls:
                available_types.append(data_type)

        if available_types:
            subjects.append(
                {
                    "folder_name": folder_name,
                    "available_types": available_types,
                }
            )

    return {"subjects": subjects}


@router.get("/api/merge_candidates")
def get_merge_candidates(folder_name: str, data_type: str):
    folder_name = (folder_name or "").strip()
    data_type = (data_type or "").strip().lower()

    if not folder_name:
        raise HTTPException(status_code=400, detail="folder_name is required")
    if data_type not in {"ppt", "pdf"}:
        raise HTTPException(status_code=400, detail="data_type must be 'ppt' or 'pdf'")

    corpus_dir = _resolve_corpus_dir(folder_name, data_type)
    if not corpus_dir.exists():
        raise HTTPException(status_code=404, detail=f"corpus dir not found: {corpus_dir}")

    chapter_names = sorted([p.stem for p in corpus_dir.glob("*.jsonl") if p.is_file()])

    chapters = [{"index": i + 1, "name": name} for i, name in enumerate(chapter_names)]

    return {
        "folder_name": folder_name,
        "data_type": data_type,
        "corpus_dir": str(corpus_dir),
        "count": len(chapters),
        "chapters": chapters,
    }


@router.post("/api/run_merge")
def run_merge(req: RunMergeRequest):
    folder_name = (req.folder_name or "").strip()
    data_type = (req.data_type or "").strip().lower()
    range_str = (req.range_str or "").strip() or None
    pick_str = (req.pick_str or "").strip() or None

    if not folder_name:
        raise HTTPException(status_code=400, detail="folder_name is required")
    if data_type not in {"ppt", "pdf"}:
        raise HTTPException(status_code=400, detail="data_type must be 'ppt' or 'pdf'")
    if bool(range_str) == bool(pick_str):
        raise HTTPException(status_code=400, detail="provide exactly one of range_str or pick_str")

    corpus_dir = _resolve_corpus_dir(folder_name, data_type)
    if not corpus_dir.exists():
        raise HTTPException(status_code=404, detail=f"corpus dir not found: {corpus_dir}")
    if not MERGE_SCRIPT.exists():
        raise HTTPException(status_code=404, detail=f"merge script not found: {MERGE_SCRIPT}")
    if not BUILD_INDEX_SCRIPT.exists():
        raise HTTPException(status_code=404, detail=f"build_index script not found: {BUILD_INDEX_SCRIPT}")

    selection_input = range_str or pick_str

    merge_cmd = [sys.executable, str(MERGE_SCRIPT), folder_name, data_type]

    merge_result = run_subprocess_utf8(
        merge_cmd,
        input_text=(selection_input + "\n") if selection_input else None,
    )

    merge_dir = corpus_dir / "merge_jsonl"

    if merge_result.returncode != 0:
        return {
            "ok": False,
            "stage": "merge",
            "command": merge_cmd,
            "returncode": merge_result.returncode,
            "stdout": merge_result.stdout,
            "stderr": merge_result.stderr,
            "merge_dir": str(merge_dir),
        }

    merged_jsonl_path = _extract_merged_jsonl_path(merge_result.stdout)
    if merged_jsonl_path is None:
        return {
            "ok": False,
            "stage": "merge",
            "command": merge_cmd,
            "returncode": merge_result.returncode,
            "stdout": merge_result.stdout,
            "stderr": merge_result.stderr,
            "merge_dir": str(merge_dir),
            "detail": "merge succeeded but cannot parse merged jsonl path from stdout",
        }

    build_cmd = [
        sys.executable,
        str(BUILD_INDEX_SCRIPT),
        folder_name,
        str(merged_jsonl_path),
    ]

    build_result = run_subprocess_utf8(build_cmd)

    index_name = merged_jsonl_path.stem
    index_dir = merged_jsonl_path.parent / "index" / index_name

    return {
        "ok": build_result.returncode == 0,
        "stage": "done" if build_result.returncode == 0 else "build_index",
        "merge": {
            "command": merge_cmd,
            "returncode": merge_result.returncode,
            "stdout": merge_result.stdout,
            "stderr": merge_result.stderr,
        },
        "build_index": {
            "command": build_cmd,
            "returncode": build_result.returncode,
            "stdout": build_result.stdout,
            "stderr": build_result.stderr,
        },
        "merged_jsonl_path": str(merged_jsonl_path),
        "merge_dir": str(merge_dir),
        "index_dir": str(index_dir),
    }