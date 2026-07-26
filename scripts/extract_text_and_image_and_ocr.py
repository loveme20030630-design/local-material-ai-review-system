from pathlib import Path
import sys
import re

from pptx import Presentation
import pytesseract      # 讀取ppt圖片
from PIL import Image, ImageFile, UnidentifiedImageError
from pdf2image import convert_from_path
import pdfplumber    # 讀取pdf文字

# 確保圖片沒問題
ImageFile.LOAD_TRUNCATED_IMAGES = True    
#========= 設定專案路徑 =========
THIS_DIR = Path(__file__).resolve().parent    # 檔案相對路徑
BASE_DIR = THIS_DIR.parent
DATA_DIR = BASE_DIR/"data"

SUBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "temp"   # 如果沒有其他科目訂個預設資料夾
SUBJECT_BASE = DATA_DIR/SUBJECT_NAME

SAVE_DIR = SUBJECT_BASE/"total_raw_data"

def get_project_root() -> Path:
    
    global SUBJECT_DATA_DIR, SAVE_DIR

    mode = sys.argv[2] if len(sys.argv) > 2 else None

    if mode == "ppt":
        SUBJECT_DATA_DIR = SUBJECT_BASE/"ppt" 
        SAVE_DIR = SUBJECT_BASE/"ppt_raw_data"
        check_data_dir_append()

    if mode == "pdf":
        SUBJECT_DATA_DIR = SUBJECT_BASE/"pdf" 
        SAVE_DIR = SUBJECT_BASE/"pdf_raw_data"
        check_data_dir_append()
    return SUBJECT_DATA_DIR, SAVE_DIR

def get_save_dir(dir_id: str)->None:
    save_dir = SAVE_DIR/dir_id
    (save_dir/"raw_img").mkdir(parents = True, exist_ok = True)
    (save_dir/"ocr").mkdir(parents = True, exist_ok = True)
    return save_dir

def check_data_dir_append()->None:     
    for i in [DATA_DIR, SUBJECT_DATA_DIR,SAVE_DIR]:
        i.mkdir(parents = True, exist_ok = True)

def get_ppt_txt(ppt_path:Path)->None:
    save_dir = get_save_dir(ppt_path.stem)
    output_path=save_dir/f"{ppt_path.stem}.txt"
    if output_path.exists():
        print(f"[SKIP] {SUBJECT_NAME} {ppt_path.stem}.txt 已存在")
        return
    
    pres=Presentation(ppt_path)         # 讀取 ppt

    page_output=[]

    # enumerate 用來確認頁碼，page 為該頁內容
    for idx, page in enumerate(pres.slides, start = 1):    
        text_list = []
        for shape in page.shapes:
            if hasattr(shape, "text") and shape.text:     # 確認屬性
                text_list.append(shape.text)

        if not text_list:
            text_list.append("這頁是沒有txt的")    

        page_block = f"====== Page {idx} ======\n"+"\n".join(text_list)
        page_output.append(page_block)

    # 完整頁數寫入，用兩行隔開
    final_text = "\n\n".join(page_output)

    # 寫入txt
    output_path.write_text(final_text, encoding = "utf-8")
    print(f"[DONE] {SUBJECT_NAME} {ppt_path.name} to {output_path.relative_to(BASE_DIR)}")

def get_ppt_raw_img(ppt_path:Path)->None:
    save_dir = get_save_dir(ppt_path.stem)
    raw_image_dir = save_dir/"raw_img"
    if raw_image_dir.exists() and any(raw_image_dir.iterdir()):# 透過iterdir做路徑搜尋
        print(f"[SKIP] {SUBJECT_NAME} {ppt_path.stem} img 已存在")
        return
    
    pre = Presentation(ppt_path)

    raw_img = save_dir/"raw_img"
    raw_img.mkdir(parents=True, exist_ok=True)

    print(f"[IMG] {SUBJECT_NAME} {ppt_path.name} to {raw_img.relative_to(BASE_DIR)}")

    for page_idx, page_ in enumerate(pre.slides, start = 1):    
        raw_img_idx = 0
        for shape in page_.shapes:
            if not hasattr(shape, "image"):
                continue

            raw_img_idx += 1

            raw_image = shape.image
            # 將圖片轉成二進位檔案完整擷取原始圖片
            raw_img_bytes = raw_image.blob
            # 避免格式錯誤，不用ext可能會出現NONE
            ext = (raw_image.ext or "png").lower()

            raw_img_path = raw_img / f"page_{page_idx:03d}_img_{raw_img_idx:02d}.{ext}"

            raw_img_path.write_bytes(raw_img_bytes)

    # 若沒有任何圖片被寫入，避免使用未定義的 raw_img_path
    if any(raw_img.iterdir()):
        print(f"[DONE] {SUBJECT_NAME} img {ppt_path.name} to {raw_img.relative_to(BASE_DIR)}")
    else:
        print(f"[WARN] {SUBJECT_NAME} {ppt_path.name} 沒有抓到任何圖片")

NOISE_PATTERNS = [
    r"^R\d+i-g\d+ht-\d+$",
    r"^(Icon-\d+\s*)+$",
    r"^第\d+頁$",
    r"^S\d+u-\d+bNo\b.*$",
    r"^SubNo\b.*$",
    r"^SubName\b.*$",

    # 整行由「R+數字」大量重複組成（含空白/連字號都允許）
    r"^(?:R\d[\w\-]*\s*){3,}$",  # 例如：R4R4R4i-i-i-...

    # 充滿 --- ___ ||| 這類分隔符，且長度至少8（避免誤刪正常內容）
    r"^[\-\_\|\=\s]{8,}$",

    # 混合字母/數字/連字號
    r"^[A-Za-z0-9\-\s]{12,}$",
]
def noise_check(line: str) -> bool:
    line = line.strip()
    if not line:
        return False

    # 有中文直接保留
    if re.search(r"[\u4e00-\u9fff]", line):
        return False
    
    if re.fullmatch(r"\d+[-_]{2,}\d+", line):
        return True

    if re.fullmatch(r"(?:\d+-\d+\s*)+", line):
        return True

    if re.search(r"(R\d){3,}", line):
        return True

    # 太短的其他東西就放過
    if len(line) < 10:
        return False

    # ASCII 重複率過高
    if re.fullmatch(r"[A-Za-z0-9\-\s]+", line):
        s = line.replace(" ", "")
        redundancy_ratio = len(set(s)) / max(1, len(s))
        if redundancy_ratio < 0.35:
            return True

    if line.count("-") >=3:
        return True

    return False
def noise_clean(line: str) -> bool:
    line= line.strip()

    if not line:
        return True
    
    if noise_check(line):
        return True

    for pattern in NOISE_PATTERNS:
        if re.match(pattern, line):
            if pattern ==  r"^[A-Za-z0-9\-\s]{12,}$":
                return noise_check(line)
            return True
    return False
def clean_pdf_text(text: str) -> str:
    lines = text.splitlines()
    clean=[]
    for line in lines:
        line = re.sub(r"\s{2,}", " ",line).strip()
        line = re.sub(r"(R\d){3,}[\w\-]*", "", line)
        line = re.sub(r"\b\d{3}[-_]{3}\d{3}\b", "", line)

        if noise_clean(line):
            continue
        if re.fullmatch(r"\d+", line):
            continue

        if not line:
            continue

        clean.append(line)
    return "\n".join(clean).strip()

def get_pdf_txt(pdf_path:Path)->None:
    save_dir = get_save_dir(pdf_path.stem)
    output_path=save_dir/f"{pdf_path.stem}.txt"
    if output_path.exists():
        print(f"[SKIP] {SUBJECT_NAME} {pdf_path.stem}.txt 已存在")
        return
    
    slide_output = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                # 清理雜訊
                text = clean_pdf_text(text)
                try:
                    text = text.strip()
                except:
                    print(f"[WARN] {pdf_path.name} 無法擷取 {idx} 頁文字")
                
                if not text:
                    text = "這頁是沒有txt的"

                slide_block = f"====== Page {idx} ======\n{text}"
                slide_output.append(slide_block)
    except Exception as e:
        raise RuntimeError(f"[ERROR] {SUBJECT_NAME} 無法讀取 {pdf_path.name}，錯誤訊息: {e}") from e
    
    final_text = "\n\n".join(slide_output)
    # 寫入txt
    output_path.write_text(final_text, encoding = "utf-8")
    print(f"[DONE] {SUBJECT_NAME} {pdf_path.name} to {output_path.relative_to(BASE_DIR)}")

def pdf_has_effective_text(pdf_path: Path) -> bool:
    """
    判斷 get_pdf_txt() 產出的 txt 是否真的有可用文字。
    - 只有 Page header / 「這頁是沒有txt的」不算有效文字
    - 純文字 PDF 沒有圖片時，可用這個避免誤判成 fatal error
    """
    save_dir = get_save_dir(pdf_path.stem)
    txt_path = save_dir / f"{pdf_path.stem}.txt"
    if not txt_path.is_file():
        return False

    try:
        raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False

    cleaned = re.sub(r"^=+\s*Page\s+\d+\s*=+$", "", raw, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = cleaned.replace("這頁是沒有txt的", "")
    return bool(cleaned.strip())

def get_pdf_raw_img(pdf_path: Path) -> None:
    save_dir = get_save_dir(pdf_path.stem)
    raw_img_dir = save_dir / "raw_img"

    if raw_img_dir.exists() and any(raw_img_dir.iterdir()):
        print(f"[SKIP] {SUBJECT_NAME} {pdf_path.stem} img 已存在")
        return

    raw_img_dir.mkdir(parents=True, exist_ok=True)
    print(f"[IMG] {SUBJECT_NAME} {pdf_path.name} to {raw_img_dir.relative_to(BASE_DIR)}")

    # 把 PDF 每一頁轉成 bitmap。
    # 注意：整頁圖片型 PDF 的文字不會被 pdfplumber.extract_text() 讀到，
    # 所以這裡保留原本「擷取內嵌圖片」邏輯，但若該頁最後沒有任何圖片被存下來，
    # 就 fallback 存整頁圖，讓 get_img_ocr() 後續可以 OCR。
    try:
        pil_pages = convert_from_path(str(pdf_path), dpi=200)
    except Exception as e:
        raise RuntimeError(f"[ERROR] {SUBJECT_NAME} {pdf_path.name} 轉圖失敗 {e}") from e

    count = 0

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, (page, pil_page) in enumerate(zip(pdf.pages, pil_pages), start=1):
                page_saved_count = 0
                images = page.images or []

                page_w, page_h = page.width, page.height
                img_w, img_h = pil_page.size

                scale_x = img_w / page_w
                scale_y = img_h / page_h

                page_img_counter = 0
                extract_log = raw_img_dir / "extract_log.txt"

                for img_idx, img in enumerate(images, start=1):
                    x0 = img["x0"]
                    top = img["top"]
                    x1 = img["x1"]
                    bottom = img["bottom"]

                    left = int(x0 * scale_x)
                    upper = int(top * scale_y)
                    right = int(x1 * scale_x)
                    lower = int(bottom * scale_y)

                    crop_w = right - left
                    crop_h = lower - upper

                    if crop_w < 80 or crop_h < 80:
                        with extract_log.open("a", encoding="utf-8") as f:
                            f.write(f"{pdf_path.name} page {page_idx} 圖片 {img_idx} 占比過小: {crop_w}x{crop_h}\n")
                        continue

                    page_area_px = img_w * img_h
                    img_area_px = crop_w * crop_h
                    area_ratio = img_area_px / page_area_px

                    if area_ratio > 0.9:
                        with extract_log.open("a", encoding="utf-8") as f:
                            f.write(f"{pdf_path.name} page {page_idx} 圖片 {img_idx} 占比過大( {area_ratio:.2%})，不採用局部裁切，將使用整頁圖片做 OCR\n")
                        continue

                    out = pil_page.crop((left, upper, right, lower))
                    page_img_counter += 1
                    out_path = raw_img_dir / f"page_{page_idx:03d}_img_{page_img_counter:02d}.png"

                    try:
                        out.save(out_path)
                        count += 1
                        page_saved_count += 1
                    except Exception as e:
                        print(f"[ERROR] 儲存圖片失敗 {e}")
                        continue

                # fallback：沒有任何 embedded image 成功存下來，就整頁存圖。
                if page_saved_count == 0:
                    out_path = raw_img_dir / f"page_{page_idx:03d}_img_01.png"
                    try:
                        pil_page.save(out_path)
                        count += 1
                        print(f"[FALLBACK] {SUBJECT_NAME} {pdf_path.name} page {page_idx} saved full page image")
                    except Exception as e:
                        print(f"[ERROR] 儲存整頁圖片失敗 page {page_idx}: {e}")
                        continue

    except Exception as e:
        raise RuntimeError(f"[ERROR] {SUBJECT_NAME} {pdf_path.name} 擷取圖片失敗 {e}") from e

    if count > 0:
        print(f"[DONE] {SUBJECT_NAME} img {pdf_path.name} ({count} images)")
    else:
        if pdf_has_effective_text(pdf_path):
            print(f"[WARN] {SUBJECT_NAME} {pdf_path.name} 沒有抓到任何圖片，但已抽到有效文字，繼續流程")
        else:
            raise RuntimeError(f"[ERROR] {SUBJECT_NAME} {pdf_path.name} 沒有抓到任何圖片，也沒有有效文字")

def get_img_ocr(file_path:Path,)->None:
    save_dir = get_save_dir(file_path.stem)
    raw_img_dir = save_dir/"raw_img"
    ocr_img_dir = save_dir/"ocr"

    if (not raw_img_dir.exists() or not any(raw_img_dir.iterdir())):
        print(f"[SKIP] {SUBJECT_NAME} {file_path.stem} 沒抓到ocr，跳過")
        return

    ocr_img_dir.mkdir(parents=True, exist_ok=True)

    # 避免讀到非指定檔案的其他內容
    new_counter = 0
    skip_counter= 0
    fail_counter = 0
    for ocr_img_path in sorted(raw_img_dir.iterdir()):
        if not ocr_img_path.is_file():
            continue
        RASTER_EXT = {"png","jpg","jpeg"}
        ext = ocr_img_path.suffix.lstrip(".").lower()
        if ext not in RASTER_EXT: # 僅處理常見圖片避免出錯，lstrip避免附檔名錯誤
            skip_counter += 1
            print(f"[SKIP] {SUBJECT_NAME} {ocr_img_path.name} 非圖片{ext}格式，跳過 ({skip_counter})")
            continue
        ocr_txt_path = ocr_img_dir/f"{ocr_img_path.stem}.txt"
        if (ocr_img_dir/f"{ocr_img_path.stem}.txt").exists():
            skip_counter += 1
            print(f"[SKIP] {SUBJECT_NAME} {ocr_img_path.name} 已存在，跳過 ({skip_counter})")
            continue

        try:
            with Image.open(ocr_img_path) as im:
                im.verify()  # 驗證圖片完整性
            with Image.open(ocr_img_path) as img:
                img = img.convert("RGB")  # 確保圖片是RGB格式
                text = pytesseract.image_to_string(img, lang="eng+chi_tra")

            ocr_txt_path.write_text(text.strip(), encoding="utf-8")
            new_counter += 1
            print(f"[DONE] {SUBJECT_NAME} ocr to {ocr_txt_path.relative_to(BASE_DIR)}")

        except (UnidentifiedImageError, OSError, ValueError) as e:
            print(f"[ERROR] {SUBJECT_NAME} 圖片無法開啟或損毀{ocr_img_path.name}")
            fail_counter += 1
            continue

        except Exception as e:
            print(f"[ERROR] {SUBJECT_NAME} 圖片開啟失敗{ocr_img_path.name}")
            fail_counter += 1
            continue

    if new_counter ==0 and fail_counter ==0 and skip_counter ==0:
        print(f"{SUBJECT_NAME} 沒抓到任何ocr")
    if fail_counter >0:
        print(f"{SUBJECT_NAME} 有 {fail_counter} 張圖片抓取ocr失敗")
    if skip_counter >0:
        print(f"{SUBJECT_NAME} 有 {skip_counter} 張圖片被跳過")

def run_ppt()-> None:   
    ppt_files = list(SUBJECT_DATA_DIR.glob("*.pptx"))
    if not ppt_files:
        raise RuntimeError(f"[ERROR] 找不到 pptx 檔案：{SUBJECT_DATA_DIR}")
    for ppt_path in ppt_files:
        get_project_root()
        get_ppt_txt(ppt_path)
        get_ppt_raw_img(ppt_path)
        get_img_ocr(ppt_path)      

def run_pdf()-> None:
    pdf_files = list(SUBJECT_DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"[ERROR] 找不到 pdf 檔案：{SUBJECT_DATA_DIR}")
    for pdf_path in pdf_files:
        get_project_root()
        get_pdf_txt(pdf_path)
        get_pdf_raw_img(pdf_path)
        get_img_ocr(pdf_path)

def main()->None:
    mode = sys.argv[2] if len(sys.argv) > 2 else None

    if not mode:
        print("請改用ppt或pdf")
        sys.exit(1)
    
    get_project_root()
    
    if mode == "ppt":
        run_ppt()
    elif mode == "pdf":
        run_pdf()

if __name__== "__main__":
    main()
