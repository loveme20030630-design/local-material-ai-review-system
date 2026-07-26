# 本地化教材檢索與人工智慧輔助複習系統

本專題針對使用大型語言模型輔助複習時，回答內容可能偏離指定教材的問題，
設計一套以使用者上傳之 PPT 與 PDF 教材為資料來源的本地化複習系統。

## 專題目標

- 擷取 PPT 與 PDF 的文字及圖片內容
- 使用 OCR 補充圖片中的文字資訊
- 建立教材向量索引
- 根據指定教材產生重點、提示與複習題
- 支援教材內容檢索與問答
- 保留章節與頁面資訊，提高結果的可追溯性
- 在教材中找不到答案時提供未命中提示

## 系統功能

- 教材上傳
- 教材前處理
- 文字、圖片與 OCR 內容擷取
- 章節資料整併
- FAISS 向量索引建立
- 教材內容搜尋與問答
- 章節提示產生
- 練習題與測驗產生
- 教材檢視

## 專案結構

```text
backend/        FastAPI 後端與 API 路由
scripts/        教材前處理、索引、題目與提示產生程式
web/            前端網頁
data/           執行時使用的教材資料夾，公開倉庫不含教材
requirements.txt
start_system.bat
stop_system.bat
```

## 使用技術

- Python
- FastAPI
- Uvicorn
- FAISS
- Tesseract OCR
- python-pptx
- pdf2image
- pdfplumber
- PyTorch
- Transformers
- Ollama
- HTML、CSS、JavaScript

## 執行環境

本專案主要在 Windows 與 Conda 環境下開發。

### 1. 建立並啟用 Conda 環境

```bash
conda create -n llmenv python=3.11
conda activate llmenv
```

### 2. 安裝 Python 套件

```bash
pip install -r requirements.txt
```

PyTorch 請依本機 CUDA 環境安裝對應版本。

FAISS 透過 Conda 安裝：

```bash
conda install -c conda-forge faiss-cpu=1.9.0
```

### 3. 安裝外部工具

執行教材前處理前，需另行安裝：

- Tesseract OCR
- Poppler
- Ollama
- 專案使用的本地語言模型

### 4. 啟動系統

執行：

```text
start_system.bat
```

系統會啟動：

- FastAPI 後端：`http://127.0.0.1:8000`
- 前端伺服器：`http://127.0.0.1:5500`
- 啟動頁面：`http://127.0.0.1:5500/launcher.html`

停止系統時執行：

```text
stop_system.bat
```

## 資料與著作權說明

本公開倉庫不包含：

- 原始課程教材
- OCR 後的教材全文
- 使用者上傳資料
- FAISS 向量索引
- 模型權重
- 第三方工具安裝檔

使用者需自行準備具有合法使用權限的 PPT 或 PDF 教材。

## 專題成果

本系統已完成教材上傳、前處理、向量索引、內容檢索、提示產生、
題目產生與前端操作介面等功能。

後續可加入：

- 更完整的圖片與圖表理解
- 題目難度標記
- 錯題回放
- 多模型效能比較
- 回答來源與頁碼顯示優化

## 作者

劉建廷  
資訊工程學系畢業專題
