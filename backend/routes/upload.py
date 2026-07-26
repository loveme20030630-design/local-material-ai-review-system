from __future__ import annotations

import re
import shutil
from enum import Enum
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


class DataTypeEnum(str, Enum):
    ppt = "ppt"
    pdf = "pdf"


def sanitize_folder_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="folder_name is required")

    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    if not safe:
        raise HTTPException(status_code=400, detail="invalid folder_name")

    return safe


def normalize_data_type(data_type: DataTypeEnum) -> str:
    return data_type.value


def _target_dir(folder_name: str, data_type: str) -> Path:
    return DATA_ROOT / folder_name / data_type


@router.get("/api/upload_existing_folders")
async def upload_existing_folders():
    folders: list[dict] = []

    if not DATA_ROOT.exists():
        return {"ok": True, "folders": folders}

    for p in sorted(DATA_ROOT.iterdir()):
        if not p.is_dir():
            continue

        available_types: list[str] = []
        for data_type in ("ppt", "pdf"):
            target_dir = p / data_type
            if target_dir.is_dir():
                files = [x for x in target_dir.iterdir() if x.is_file()]
                if files:
                    available_types.append(data_type)

        folders.append(
            {
                "folder_name": p.name,
                "available_types": available_types,
            }
        )

    return {
        "ok": True,
        "folders": folders,
    }


@router.get("/api/upload_existing_files")
async def upload_existing_files(folder_name: str, data_type: DataTypeEnum):
    safe_folder = sanitize_folder_name(folder_name)
    safe_type = normalize_data_type(data_type)

    target_dir = _target_dir(safe_folder, safe_type)

    items: list[dict] = []
    if target_dir.is_dir():
        for p in sorted(target_dir.iterdir()):
            if not p.is_file():
                continue
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            items.append(
                {
                    "name": p.name,
                    "size": size,
                    "path": str(p),
                }
            )

    return {
        "ok": True,
        "folder_name": safe_folder,
        "data_type": safe_type,
        "target_dir": str(target_dir),
        "count": len(items),
        "files": items,
    }


@router.post("/api/upload_material_folder")
async def upload_material_folder(
    folder_name: str = Form(...),
    data_type: DataTypeEnum = Form(...),
    files: List[UploadFile] = File(...),
):
    safe_folder = sanitize_folder_name(folder_name)
    safe_type = normalize_data_type(data_type)

    target_dir = _target_dir(safe_folder, safe_type)
    target_dir.mkdir(parents=True, exist_ok=True)

    allowed_ext = {
        "ppt": {".ppt", ".pptx"},
        "pdf": {".pdf"},
    }

    saved_files: list[str] = []
    skipped_files: list[str] = []
    overwritten_files: list[str] = []

    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")

    for upload in files:
        original_name = Path(upload.filename or "").name
        if not original_name:
            skipped_files.append("(empty filename)")
            continue

        ext = Path(original_name).suffix.lower()
        if ext not in allowed_ext[safe_type]:
            skipped_files.append(original_name)
            continue

        save_path = target_dir / original_name
        if save_path.exists():
            overwritten_files.append(original_name)

        with save_path.open("wb") as f:
            shutil.copyfileobj(upload.file, f)

        saved_files.append(str(save_path))

    if not saved_files:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "no valid files saved",
                "allowed_extensions": sorted(allowed_ext[safe_type]),
                "skipped_files": skipped_files,
            },
        )

    return {
        "ok": True,
        "folder_name": safe_folder,
        "data_type": safe_type,
        "saved_count": len(saved_files),
        "saved_files": saved_files,
        "skipped_files": skipped_files,
        "overwritten_files": overwritten_files,
        "target_dir": str(target_dir),
    }
