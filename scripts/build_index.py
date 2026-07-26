from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

THIS_DIR = Path(__file__).resolve().parent          
BASE_DIR = THIS_DIR.parent                          
DATA_PATH = BASE_DIR / "data"                      

SUBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "temp"
SUBJECT_PATH = DATA_PATH / SUBJECT_NAME       

def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def mean_pooling(last_hidden_state: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm

def encode_texts(
    texts: list[str],
    model_name: str,
    batch_size: int = 32,
    max_length: int = 512,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    vecs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            pooled = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            vec = pooled.detach().cpu().numpy().astype(np.float32)
            vecs.append(vec)

    emb = np.vstack(vecs).astype("float32")
    emb = l2_normalize(emb).astype("float32")
    return emb

def resolve_jsonl_path(arg: str) -> Path:
    p = Path(arg)
    if p.is_absolute():
        return p.resolve()
    return (SUBJECT_PATH / p).resolve()


def build_index_text(page: dict) -> str:
    """
    建 index 用文字：一般 PDF/PPT 用 text；整頁圖片型 PDF 則用 OCR。
    若 text 和 ocr 都存在，就合併，避免圖片中的關鍵字被漏掉。
    """
    text = (page.get("text") or "").strip()
    ocr = (page.get("ocr") or "").strip()

    if text == "這頁是沒有txt的":
        text = ""

    return "\n\n".join(x for x in [text, ocr] if x).strip()

def main() -> None:
    if len(sys.argv) < 3:
        raise RuntimeError("用法：python build_index.py <SUBJECT_NAME> <JSONL_PATH>")

    jsonl_path = resolve_jsonl_path(sys.argv[2])
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"[ERROR] 找不到檔案：{jsonl_path}")

    index_name = jsonl_path.stem                     
    index_root = jsonl_path.parent / "index"         # .../pdf_exam_corpus/index
    save_dir = index_root / index_name              
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] {jsonl_path}")
    pages = load_jsonl(jsonl_path)
    if not pages:
        raise RuntimeError("[ERROR] jsonl 為空")

    texts: list[str] = []
    page_items: list[dict] = []

    for p in pages:
        text = build_index_text(p)
        if not text:
            continue
        texts.append(text)
        page_items.append(
            {
                "doc_id": p.get("doc_id"),
                "page": p.get("page"),
                "images": p.get("images", []),
                "ocr": p.get("ocr", ""),
            }
        )

    if not texts:
        raise RuntimeError("[ERROR] 沒有可用文字頁面")
    

    print(f"[DOCS] {len(texts)} pages")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = encode_texts(texts, model_name=model_name)

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    if index.ntotal != len(texts):
        raise RuntimeError("index/texts mismatch")

    faiss_path = save_dir / "vectors.faiss"
    pages_path = save_dir / "pages.json"            
    texts_path = save_dir / "texts.json"
    info_path = save_dir / "index.info.json"

    faiss.write_index(index, str(faiss_path))
    pages_path.write_text(json.dumps(page_items, ensure_ascii=False, indent=2), encoding="utf-8")
    texts_path.write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8")

    info = {
        "subject": SUBJECT_NAME,
        "source": str(jsonl_path),
        "count": int(index.ntotal),
        "dim": dim,
        "model": model_name,
        "output_dir": str(save_dir),
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] index={save_dir}")
    print("       vectors.faiss / pages.json / texts.json / index.info.json")

if __name__ == "__main__":
    main()
