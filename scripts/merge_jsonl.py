import json
from pathlib import Path
import sys
import os
import re
from typing import Optional, List

THIS_DIR = Path(__file__).resolve().parent    # 檔案相對路徑
BASE_DIR = THIS_DIR.parent
DATA_DIR = BASE_DIR/"data"

SUBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "temp"   # 如果沒有其他科目訂個預設資料夾
SUBJECT_BASE = DATA_DIR/SUBJECT_NAME

# ========= 常見章節名稱 =========
_ROMAN_RE = re.compile(r"^\(\s*([IVXLCDM]+)\s*\)", re.IGNORECASE)
_CN_CHAPTER_RE = re.compile(r"第\s*([一二三四五六七八九十零\d]+)\s*(章|節|講|回|單元|章節)")
_EN_CHAPTER_RE = re.compile(r"\b(?:CH|Chapter|Chap|Lesson|Lecture|Unit)\s*[-_ ]?\s*(\d{1,3})\b", re.IGNORECASE)
_TRAILING_SLOT_RE = re.compile(r"(?:^|[^0-9])[-_ ](\d{1,3})(?=\s*\(|\s*\[|\.|_|-|$)")

_ROMAN_MAP = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
CH_RE = re.compile(r'^CH(\d{2})(?:[_\-\s\(].*)?$', re.IGNORECASE)

def get_chapter_dir() -> Path:
    mode = sys.argv[2] if len(sys.argv) > 2 else None

    if mode == "ppt":
        return SUBJECT_BASE / "ppt_exam_corpus"
    if mode == "pdf":
        return SUBJECT_BASE / "pdf_exam_corpus"

    ppt_dir = SUBJECT_BASE / "ppt_exam_corpus"
    pdf_dir = SUBJECT_BASE / "pdf_exam_corpus"

    if ppt_dir.exists() and not pdf_dir.exists():
        return ppt_dir
    if pdf_dir.exists() and not ppt_dir.exists():
        return pdf_dir
    if ppt_dir.exists():
        return ppt_dir

    return pdf_dir
def list_chapter_dirs(jsonl_dir:Path) -> list[str]:
    """
    掃描 jsonl_dir 裡面的各章節的 jsonl 檔案，並回傳章節 stem 
    """

    item: list[str] = []
    for p in jsonl_dir.glob("*.jsonl"):
        if not p.is_file():
            continue
        stem = p.stem.strip()
        if not stem:
            continue
        if re.search(r"\)\s*-\s*\(", stem):# 排除已合併檔案
            continue
        item.append(stem)
    return sorted(item, key=natural_sort_key)

def roman_to_int(s: str) -> int | None:
    """
    將羅馬數字轉換成整數，若格式不正確回傳None
    """
    s = s.upper().strip()
    if not s:
        return None
    total = 0
    prev = 0
    for doc_id in reversed(s):
        if doc_id not in _ROMAN_MAP:
            return None
        val = _ROMAN_MAP[doc_id]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total

def cn_num_to_int(s: str) -> int | None:
    """
    中文數字轉int
    """
    s = s.upper().strip()
    if not s:
        return None
    
    if s.isdigit():
        return int(s)
    
    m = {"零":0,"一":1,"二":2,"兩":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}
    
    total = 0
    num = 0
    saw = False
    for doc_id in s:
        if doc_id in m:
            num = m[doc_id]
            saw = True
        elif doc_id == "十":
            total += (num if num else 1)*10
            num = 0
            saw = True
        else:
            return None
    total += num
    return total if saw else None

def extract_order_hint(stem: str) -> tuple[int, int]:
    """
    嘗試從檔名排「章節順序」
    回傳：(優先級, 數值)

    優先級設計：
    0 = 英文章節
    1 = 中文章節
    2 = 羅馬數字
    3 = 尾端數字
    """

    m = _EN_CHAPTER_RE.search(stem)
    if m:
        return(0, int(m.group(1)))
    
    m = _CN_CHAPTER_RE.search(stem)
    if m:
        n = cn_num_to_int(m.group(1))
        if n is not None:
            return (1, n)
    
    m = _ROMAN_RE.match(stem)
    if m:
        n = roman_to_int(m.group(1))
        if n is not None:
            return (2, n)
        
    m = _TRAILING_SLOT_RE.search(stem)    
    if m:
        return(3, int(m.group(1)))
    
    return None

def natural_sort_key(stem: str):
    """
    統一的排序 key：
    - 能抽出章節數字 ： 上面的規則排序
    - 抽不出　:　自然排序（數字/文字混排）
    """
    hint = extract_order_hint(stem)
    if hint is not None:
        pri, n = hint
        return(0, pri, n, stem.lower())
    
    # 自然排序
    parts = re.split(r"(\d+)", stem)
    keys: list[object] = []
    for p in parts:
        if p =="":
            continue
        keys.append(int(p) if p.isdigit() else p.lower())
    return (1, keys)

def doc_id_index(all_doc_id_dirs: list[str], doc_id: str) -> int:
    return all_doc_id_dirs.index(doc_id)

def print_selection_table(stem: list[str]) -> None:
    """
    print 章節對應表：
    index -> stem
    (UI 使用)
    """
    # 讓章節格式能夠對齊
    width = len(str(len(stem)))
    print("\n[CHAPTER LIST]\n")
    for i, s in enumerate(stem, start=1):
        print(f"{i:>{width}}.{s}")
    print("")

def pick_by_range(item: list[str], range_str: str) -> list[str]:
    """
    選擇連續章節，例如 1-4
    """
    m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", range_str)
    if not m:
        raise RuntimeError(f"[ERROR] range格式錯誤: {range_str}")
    
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else a

    if a>b:
        a, b = b, a
    
    s = a - 1
    e = min(b - 1, len(item) - 1)

    if s >=  len(item):
        raise RuntimeError(f"[ERROR] range 起點超出範圍")

    return item[s:e+1]  

def pick_by_pick(item: list[str], pick_str: str) -> list[str]:
    """
    挑不連續章節，例如1, 3, 5
    """
    parts = [p.strip() for p in pick_str.split(",") if p.strip()]
    seen = set()
    out: list[str] =[]
    
    for p in parts:
        if not p.isdigit():
            raise RuntimeError(f"[ERROR] pick 格式錯誤: {pick_str}")

        idx = int(p) - 1
        if idx < 0 or idx >= len(item):
            raise RuntimeError("[ERROR] pick 超出範圍")

        if idx not in seen:
            seen.add(idx)
            out.append(item[idx])
    return out

def pick_by_start_end(items: list[str], start: Optional[str], end: Optional[str]) -> list[str]:
    """
    以章節 stem 名稱指定區間
    """
    if start is None and end is None:
        return items

    if start and start not in items:
        raise RuntimeError(f"[ERROR]找不到起始章節 {start}")

    if end and end not in items:
        raise RuntimeError(f"[ERROR]找不到結束章節 {end}")

    s = items.index(start) if start else 0
    e = items.index(end) if end else len(items) - 1

    if s > e:
        s, e = e, s

    return items[s:e+1]    

def extract_chapter_label(stem: str) -> str|None:
    """
    從檔名抽「章節標籤」用在輸出檔名：
    - (II) ...  -> II
    - Chapter 3 -> 3
    - 第三章    -> 3
    - CH04      -> 04
    抽不到就回 None
    """
    m = _ROMAN_RE.match(stem)
    if m:
        return m.group(1).upper()

    m = _EN_CHAPTER_RE.search(stem)
    if m:
        return str(int(m.group(1)))

    m = _CN_CHAPTER_RE.search(stem)
    if m:
        n = cn_num_to_int(m.group(1))
        if n is not None:
            return str(n)

    m = CH_RE.match(stem.strip().upper())
    if m:
        return m.group(1) 

    return None

def extract_chapter_tag(stem: str) -> tuple[str, str] | None:
    """
    回傳 (fmt, tag)
    fmt:
      - "roman"  -> "II"
      - "cn"     -> "第一章"
      - "ch"     -> "CH01"
      - "en"     -> "Chapter 3"
    tag:
      - 章節標籤字串（保持原格式）
    """
    s = stem.strip()

    m = _ROMAN_RE.match(s)
    if m:
        return ("roman", m.group(1).upper())

    m = _CN_CHAPTER_RE.search(s)
    if m:
        n = cn_num_to_int(m.group(1))
        if n is not None:
            return ("cn", f"第{m.group(1)}章")  
        
    m = CH_RE.match(s.upper())
    if m:
        num = m.group(1)
        return ("ch", f"CH{num}")

    m = _EN_CHAPTER_RE.search(s)
    if m:
        return ("en", f"Chapter {int(m.group(1))}")

def build_merge_out_name(
    selected: list[str],
    pick_str: Optional[str] = None,
    range_str: Optional[str] = None,
) -> str:
    if not selected:
        raise RuntimeError("[ERROR] selected 為空，無法建立輸出檔名")

    tags: list[str] = []
    for stem in selected:
        tag_info = extract_chapter_tag(stem)
        if tag_info is None:
            tags.append("")
        else:
            tags.append(tag_info[1].replace(" ", "_"))

    # 不連續選章：優先用章節 tag；抽不到就退成 pick_1_3_5
    if pick_str:
        clean_tags = [t for t in tags if t]
        if len(clean_tags) == len(selected):
            return f"{'_'.join(clean_tags)}.jsonl"

        safe_pick = re.sub(r"\s+", "", pick_str).replace(",", "_")
        return f"pick_{safe_pick}.jsonl"

    # 連續區間：優先用首尾 tag；抽不到才退成 range_1_5
    first_tag = tags[0] if tags else ""
    last_tag = tags[-1] if tags else ""

    if first_tag and last_tag:
        return f"{first_tag}-{last_tag}.jsonl"

    if range_str:
        safe_range = re.sub(r"\s+", "", range_str).replace("-", "_")
        return f"range_{safe_range}.jsonl"
 
    return "merged.jsonl"

def choose_chapter_dir(
        all_doc_id_dirs: list[str],
        start_doc_id: Optional[str],
        end_doc_id: Optional[str],
        range_str: Optional[str],
        pick_str: Optional[str],
) -> list[str]:
    """
    選章優先順序：
    1. pick（不連續）
    2. range（連續）
    3. start/end
    """
    if pick_str:
        return pick_by_pick(all_doc_id_dirs, pick_str)

    if range_str:
        return pick_by_range(all_doc_id_dirs, range_str)

    return pick_by_start_end(all_doc_id_dirs, start_doc_id, end_doc_id)

def merge_jsonl(
        jsonl_dir: Path,
        all_doc_id_dirs: list[str],
        start_doc_id: Optional[str],
        end_doc_id: Optional[str],
        range_str: Optional[str],
        pick_str: Optional[str],
) -> Path:
    choose_doc_id_dirs = choose_chapter_dir(all_doc_id_dirs, 
                                            start_doc_id, 
                                            end_doc_id, 
                                            range_str, 
                                            pick_str) 
    
    out_name = build_merge_out_name(choose_doc_id_dirs,
                                    pick_str=pick_str,
                                    range_str=range_str,)

    out_dir = jsonl_dir / "merge_jsonl"
    out_dir.mkdir(parents = True, exist_ok = True)
    out_path = out_dir / out_name 

    try:
        with out_path.open("w", encoding="utf-8") as out_file:
            for doc_id in choose_doc_id_dirs:
                jsonl_path = jsonl_dir / f"{doc_id}.jsonl"
                
                if not jsonl_path.is_file():
                    raise RuntimeError(f"[ERROR] 選取章節缺少 jsonl 檔案: {jsonl_path}")
                
                with jsonl_path.open("r", encoding="utf-8") as in_file:
                    for line in in_file:
                        out_file.write(line)
        print(f"[DONE]合併完成，輸出檔案: {out_path}")
    except Exception as e:
        raise RuntimeError(f"[ERROR]合併jsonl檔案時發生錯誤: {e}")
    return out_path

def main():
    data_jsonl_dir = get_chapter_dir()
    all_doc_id_dirs = list_chapter_dirs(data_jsonl_dir)
    all_doc_id_dirs = sorted(all_doc_id_dirs, key=natural_sort_key)

    start_doc_id = sys.argv[3] if len(sys.argv) > 3 else None
    end_doc_id = sys.argv[4] if len(sys.argv) > 4 else None

    if start_doc_id is None and end_doc_id is None:
        api_mode = os.environ.get("MERGE_JSONL_API_MODE") == "1"
        if not api_mode:
            print_selection_table(all_doc_id_dirs)
            s = input("選取章節: 若輸入1-5則合併1到5章節，若選擇1, 3, 5則合併選取章節;ENTER鍵為直接退出").strip()
        else:
            # API 已由前端提供章節範圍，不需要再輸出互動提示，避免技術紀錄混入提示文字。
            s = sys.stdin.readline().strip()

        if not s:
            print("[EXIT] 未選擇章節，退出")
            return

        range_str = s if re.fullmatch(r"\s*\d+\s*(?:-\s*\d+\s*)?\s*", s) else None
        pick_str = s if (range_str is None and re.fullmatch(r"\s*\d+\s*(?:,\s*\d+\s*)+\s*", s)) else None

        if range_str is None and pick_str is None:
            raise RuntimeError(f"[ERROR]輸入格式錯誤：{s}（只能是 1-4 或 1,3,5）")

        merge_jsonl(
            jsonl_dir=data_jsonl_dir,
            all_doc_id_dirs=all_doc_id_dirs,
            start_doc_id=None,
            end_doc_id=None,
            range_str=range_str,
            pick_str=pick_str,
        )
        return

    merge_jsonl(
        jsonl_dir=data_jsonl_dir,
        all_doc_id_dirs=all_doc_id_dirs,
        start_doc_id=start_doc_id,
        end_doc_id=end_doc_id,
        range_str=None,
        pick_str=None,
    )

if __name__ == "__main__":
    main()