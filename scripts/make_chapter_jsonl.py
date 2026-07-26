from __future__ import annotations # 避免 forward reference 問題（尚未定義的 class）

import re
from collections import defaultdict # 用於建立預設值的字典
from pathlib import Path 
from typing import Dict, List, Tuple, Optional # 用於類型註解
import sys
import json
#========= 設定專案路徑 =========
THIS_DIR = Path(__file__).resolve().parent    # 檔案相對路徑
BASE_DIR = THIS_DIR.parent
SUBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "temp"   # 如果沒有其他科目訂個預設資料夾
DATA_DIR = BASE_DIR/"data"/SUBJECT_NAME  # 考試語料庫目錄

PAGE_IMG_RE = re.compile(
    # 匹配頁面圖像文件名
    # 範例：page_001_img_01.png
    r'^page_(\d+)_img_(\d+)\.(png|jpg|jpeg)$',
    re.IGNORECASE
    ) 
RE_PAGE_HEADER = re.compile(
    r'^=+\s*Page\s+(\d+)\s*=+$',
    re.IGNORECASE
)

def is_valid_doc_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    name = p.name.strip()
    if not name:
        return False
    if name.startswith("_"):
        return False
    return True

def pick_text_path(doc_dir: Path) -> Optional[Path]:
    """
    優先找 {資料夾名}.txt；找不到就挑第一個 .txt
    """
    cand = doc_dir / f"{doc_dir.name}.txt"
    if cand.is_file():
        return cand
    text = sorted(doc_dir.glob("*.txt"))
    return text[0] if text else None

def get_chapter_dir() ->  tuple[list[dict[str, object]], Path]:
    mode = sys.argv[2] if len(sys.argv) > 2 else None

    context: list[dict[str, object]] = []

    DATA_PATH = DATA_DIR / ("ppt_raw_data" if mode == "ppt" else "pdf_raw_data")
    save_dir = DATA_DIR / ("ppt_exam_corpus" if mode == "ppt" else "pdf_exam_corpus")
    save_dir.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.is_dir():
        raise RuntimeError(f"[ERROR]資料目錄不存在: {DATA_PATH}")

    for doc_dir in sorted(DATA_PATH.iterdir()):
        if not is_valid_doc_dir(doc_dir):
            continue

        doc_id = doc_dir.name  
        text_path = pick_text_path(doc_dir)
        raw_img_dir = doc_dir / "raw_img"
        ocr_dir = doc_dir / "ocr"

        if text_path is None or (not text_path.is_file()):
            print(f"[SKIP]跳過：找不到: {text_path}", doc_dir)
            continue

        if not raw_img_dir.is_dir():
            raw_img_dir = doc_dir / "raw_img"
            print(f"[SKIP]跳過異常圖像目錄:{text_path}", raw_img_dir)

        context.append({
            "doc_id": doc_id,
            "text_path": text_path,
            "raw_img_dir": raw_img_dir,
            "ocr_dir": ocr_dir
        })

    return context, save_dir

def scan_page_assets(raw_img_dir: Path, ocr_dir: Path)->Dict[int, list[Dict[str, Path| int | None]]]:   

    pages = defaultdict(list)

    if not raw_img_dir.is_dir():
        return {}
    
    ocr_exists = ocr_dir.is_dir()
    
    for img_path  in sorted(raw_img_dir.iterdir()):
        if not img_path .is_file():
            continue

        m = PAGE_IMG_RE.match(img_path.name)
    
        
        if not m:
            continue
        
        page_number = int(m.group(1))
        img_index = int(m.group(2))
        img_id = img_path.stem

        ocr_path = None

        if ocr_exists:
            candidate = ocr_dir / f"{img_id}.txt"
            if candidate.is_file():
                ocr_path = candidate
            else:
                print(f"[SKIP]找不到對應 OCR 檔案，跳過: {candidate}")

        pages[page_number].append({
            "img_id": img_id,
            "img_index": img_index,
            "img_raw_path": img_path,
            "ocr_path": ocr_path
        })
    for page_number in pages:
        pages[page_number].sort(key=lambda x: x["img_index"])
    return dict(pages)

def parse_page_txt(txt_path: Path) -> Dict[int, str]:
    
    if not txt_path.is_file():
        return {}

    raw_text = txt_path.read_text(encoding="utf-8")

    page_line : Dict[int, List[str]] = {}
    cur_page : Optional[int] = None

    for line in raw_text.splitlines():
        m = RE_PAGE_HEADER.match(line.strip())
        if m:
            cur_page = int(m.group(1))
            page_line[cur_page] = []
            continue

        # 在遇到標頭前的雜訊略過
        if cur_page is None:
            continue

        page_line[cur_page].append(line)

    final_texts: Dict[int, str] = {}
    for page_num, lines in page_line.items():
        final_texts[page_num] = "\n".join(lines).strip()

    return final_texts

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        return "\n".join(str(i) for i in text)
    return str(text)

def read_ocr_text(ocr_path: Optional[Path]) -> str:
    if ocr_path is None:
        return ""
    if not isinstance(ocr_path, Path):
        return ""
    if not ocr_path.is_file():
        return ""
    return ocr_path.read_text(encoding="utf-8", errors="ignore").strip()

def to_project_relative_path(path: Optional[Path]) -> str:
    if path is None:
        return ""
    if not isinstance(path, Path):
        return ""
    try:
        return path.resolve().relative_to(BASE_DIR).as_posix()
    except Exception:
        return path.as_posix()

def write_json(obj : object, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_page_jsonl(doc_id: str, pages: Dict[int, list[Dict]], texts: Dict[int, str]) -> List[Dict]:
    final_pages = []
    all_page_numbers = sorted(set(texts.keys()) | set(pages.keys()))
    for page_number in sorted(all_page_numbers):
        assets = pages.get(page_number, [])
        final_pages.append({
            "doc_id": doc_id,
            "page": page_number,
            "text": texts.get(page_number, ""),
            "images": [
                to_project_relative_path(a.get("img_raw_path"))
                for a in assets
                if a.get("img_raw_path") is not None
            ],
            "ocr": "\n\n".join(
                t for t in (read_ocr_text(a.get("ocr_path"))for a in assets)if t
            ).strip(),
        })

    return final_pages

def main() -> None:
    context, save_dir = get_chapter_dir()
    if not context:
        raise RuntimeError("[ERROR] 沒有可轉換成 JSONL 的章節資料")

    written_count = 0

    for ctx in context:
        page = scan_page_assets(ctx["raw_img_dir"], ctx["ocr_dir"])
        text = parse_page_txt(ctx["text_path"])
        if (not page) and (not text):
            print(f"[SKIP]章節無頁面資源，跳過: {ctx['doc_id']}")
            continue

        final_jsonl = build_page_jsonl(ctx["doc_id"], page, text)
        if not final_jsonl:
            print(f"[SKIP]章節沒有可寫入 JSONL 的頁面，跳過: {ctx['doc_id']}")
            continue

        out_path = save_dir / f"{ctx['doc_id']}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for obj in final_jsonl:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        written_count += 1
        print("[DONE]", out_path, "n=", len(final_jsonl))

    if written_count == 0:
        raise RuntimeError("[ERROR] 沒有成功產生任何章節 JSONL")

if __name__ == "__main__":
    main()