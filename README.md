# 本地化教材檢索與人工智慧輔助複習系統

本專題以使用者自行匯入的 PPT、PDF 課程教材為資料來源，建置一套可在本機執行的教材檢索與 AI 輔助複習系統。

系統會擷取教材中的文字、圖片與 OCR 內容，整理為結構化資料，並依使用者指定的複習範圍建立向量索引。完成索引後，可進行教材段落查詢、AI 內容整理、關鍵字輔助查詢、選擇題生成、練習與測驗。

本系統將「教材檢索」與「生成式整理」分開處理：

* **找段落**：直接從教材資料中搜尋，不使用生成式語言模型改寫內容。
* **整理資料**：先檢索教材內容，再將取得的教材片段作為 context 交由本地語言模型整理。

查詢與題目結果會保留教材頁碼與來源資訊，使用者可開啟系統內的對應教材頁面，查看該頁的文字、OCR 與圖片內容，以進一步確認檢索或作答依據。

---

## 專題目標

本專題希望建立一套以指定教材內容為核心的本地化複習工具，主要目標包括：

* 支援 PPT、PDF 教材匯入
* 擷取教材中的文字、圖片與 OCR 內容
* 將教材轉換為可供檢索的結構化資料
* 讓使用者自行選擇複習章節與範圍
* 針對指定範圍建立向量索引
* 支援直接文字比對與向量檢索
* 將教材檢索與生成式內容整理分離
* 使用本地語言模型進行教材限定的內容整理
* 提供關鍵字輔助查詢
* 依教材內容生成四選一題目
* 提供練習模式、測驗模式與錯題檢視
* 保存教材頁碼、圖片與 OCR 等來源資訊
* 讓使用者由查詢或題目結果開啟對應教材頁面確認內容
* 盡可能在本機環境完成教材處理、檢索與模型推論

---

## 系統主要流程

```text
PPT / PDF 教材
        │
        ▼
文字、圖片擷取與 OCR
        │
        ▼
教材內容結構化
     JSONL
        │
        ▼
選擇複習範圍
單章 / 連續章節 / 多章節
        │
        ▼
建立 FAISS 向量索引
        │
        ├─────────────────────┐
        │                     │
        ▼                     ▼
    找段落                整理資料
  不使用 LLM             Retrieval + LLM
        │                     │
        ├──────────┬──────────┤
        │          │          │
        ▼          ▼          ▼
     關鍵字      AI 出題    教材限定回答
                   │
                   ▼
              練習 / 測驗
                   │
                   ▼
          答案、解析與來源
                   │
                   ▼
              來源頁面
       TEXT / OCR / 圖片 / 頁碼
```

---

# 已完成功能

## 1. 教材匯入

系統提供教材匯入介面，可建立不同科目的資料夾，並依教材格式管理 PPT 與 PDF 檔案。

目前支援：

* 建立科目教材資料夾
* 選擇 PPT 或 PDF 類型
* 上傳教材檔案
* 查看既有教材與資料夾
* 依科目與教材格式管理後續處理資料

對應前端：

```text
web/upload.html
```

對應後端：

```text
backend/routes/upload.py
```

---

## 2. 教材前處理

教材匯入後，可透過前處理流程擷取教材內容。

目前處理內容包含：

* PPT 文字擷取
* PDF 文字擷取
* 教材圖片擷取
* Tesseract OCR
* 圖片中的文字資訊補充
* 教材頁碼保存
* 教材識別資訊保存
* 圖片路徑保存
* OCR 結果保存
* 建立章節 JSONL

每個教材頁面會保存與該頁相關的資料，例如：

```text
doc_id
page
text
ocr
images
```

因此後續檢索結果除了文字外，也可以保留頁面與圖片等資訊。

主要程式：

```text
scripts/extract_text_and_image_and_ocr.py
scripts/make_chapter_jsonl.py
backend/routes/preprocess.py
```

---

## 3. 自訂複習範圍

系統並非固定對整份教材建立單一索引。

使用者可以依照目前需要複習的範圍，自行選擇教材章節，再將所選內容合併並建立對應索引。

支援：

* 單一章節
* 連續章節
* 多個指定章節

例如：

```text
CH01
CH03-CH05
CH01 + CH04 + CH08
```

系統會將指定範圍的 JSONL 資料整併，再建立該範圍專用的向量索引。

主要程式：

```text
web/merge_json.html
backend/routes/merge_json.py
scripts/merge_jsonl.py
scripts/build_index.py
```

---

## 4. 向量索引

目前向量索引流程使用：

```text
sentence-transformers/all-MiniLM-L6-v2
```

文字編碼流程包含：

1. Tokenization
2. Transformer Encoding
3. Attention Mask Mean Pooling
4. L2 Normalization
5. FAISS IndexFlatIP

索引建立後，除了向量本身，也會另外保存教材頁面與來源相關資訊。

主要使用：

* PyTorch
* Transformers
* FAISS
* NumPy

---

# 教材查詢

## 5. 找段落：不使用生成式語言模型

一般教材查詢功能的目的，是讓使用者直接尋找教材中已有的內容，而不是由語言模型重新生成答案。

前端對應按鈕：

```text
找段落
```

查詢流程為：

```text
使用者輸入查詢
        │
        ▼
直接文字比對
        │
        ├── 找到 → 顯示直接命中結果
        │
        └── 未找到
                │
                ▼
             向量檢索
                │
                ▼
          顯示相關教材頁面
```

查詢結果可包含：

* TEXT
* OCR
* 教材圖片
* doc_id
* 教材頁碼
* 檢索排序結果

此模式不需要生成式語言模型重新整理教材內容。

因此使用者可以先查看教材原始擷取內容，再自行判斷結果是否符合需求。

主要程式：

```text
scripts/search_index_qa.py
backend/routes/search.py
web/search.html
```

---

## 6. 整理資料：Retrieval + Local LLM

除了直接尋找教材段落之外，系統也提供 AI 內容整理功能。

前端對應按鈕：

```text
整理資料
```

此模式會先從目前指定的複習範圍中取得相關教材內容，再將檢索結果組成 context，交由本地語言模型進行整理。

流程為：

```text
使用者問題
    │
    ▼
Hybrid Retrieval
    │
    ▼
Top-k 教材片段
    │
    ▼
組成 CONTEXT
    │
    ▼
Local LLM
    │
    ▼
教材限定回答
```

目前系統會依問題內容判斷不同整理方式，包括：

* 一般問答
* 重點整理
* 項目比較
* 流程／步驟整理

例如使用者詢問：

```text
整理影像銳化的重點
```

系統會以整理模式呈現教材中的相關內容。

若詢問：

```text
比較 Spatial Domain 與 Frequency Domain
```

則會使用比較格式整理目前檢索到的教材內容。

---

## 7. 教材限定回答

AI 整理模式的提示規則會要求模型：

* 只能根據本次提供的 CONTEXT 回答
* 不主動使用教材外知識補充
* 重要概念盡量附上來源頁碼
* CONTEXT 找不到的內容不自行補足
* 比較題找不到的項目標示未找到
* 流程題不將模型推論描述成教材唯一標準流程

目前主要透過 prompt 約束生成結果。

因此需要注意：

> 系統尚未建立逐句來源支持驗證機制。

目前所顯示的來源，代表本次檢索取得並提供給模型的候選教材內容，不代表生成回答的每一句文字都已完成自動證據驗證。

---

# 關鍵字功能

## 8. 關鍵字輔助查詢

系統可以針對目前選定的教材範圍產生複習關鍵字。

使用者可以：

* 載入已產生的關鍵字
* 生成新的關鍵字
* 選擇是否使用本地語言模型輔助
* 點擊關鍵字直接填入搜尋欄位
* 完全不使用關鍵字，自行輸入查詢內容

關鍵字的角色是：

```text
輔助使用者找到查詢方向
```

而不是限制使用者只能透過預先產生的關鍵字進行搜尋。

主要程式：

```text
scripts/make_chapter_tips.py
backend/routes/tips.py
web/search.html
```

---

# AI 選擇題與測驗

## 9. 選擇題生成

系統可以依目前建立的複習範圍產生四選一題目。

每一題主要包含：

* 題幹
* A / B / C / D 四個選項
* 正確答案
* 答案解析
* 教材來源
* 來源對應資訊

出題時，模型被要求只能根據提供的教材 context 建立題目。

若教材資訊不足以建立符合條件的題目，則不應自行使用教材外知識補充。

主要程式：

```text
scripts/make_exam_json.py
backend/routes/exam.py
```

---

## 10. 練習模式

練習模式適合逐題複習。

每題作答後即可立即查看：

* 是否答對
* 正確答案
* 答案解析
* 教材來源
* 對應教材內容

使用者可以依據解析中的來源標籤查看相關教材內容。

---

## 11. 測驗模式

測驗模式在作答期間不立即公布答案。

使用者完成整份題目並提交後，系統才統一顯示：

* 答對題數
* 錯題數
* 正確率
* 每題正確答案
* 答案解析
* 教材來源

此外也提供：

* 題號導航
* 錯題導航
* 僅顯示錯題
* 完整題目回顧

---

## 12. 歷史測驗紀錄

前端會保存作答紀錄，使用者可以重新查看先前完成的測驗。

紀錄內容包含：

* 作答模式
* 題目
* 選項
* 使用者答案
* 正確答案
* 答案解析
* 教材來源
* 答對題數
* 錯題數
* 正確率

使用者也可以只查看歷史測驗中的錯題。

前端主要程式：

```text
web/practice_and_test.html
```

---

# 教材來源頁面

## 13. 來源資訊保存

教材在前處理階段即保留：

* 教材識別資訊
* 頁碼
* 文字
* OCR
* 圖片

因此查詢結果或題目解析可以建立來源頁面的定位資訊。

---

## 14. Viewer

查詢結果與題目來源中的：

```text
開啟來源頁
```

會開啟系統內的 `viewer.html`，並透過：

```text
subject
data_type
chapter
doc_id
page
```

定位到對應教材頁面。

Viewer 可顯示：

* 該頁文字
* OCR 辨識結果
* 教材圖片
* 教材頁碼
* doc_id

另外提供：

* 頁碼列表
* 上一頁
* 下一頁
* OCR 頁面提示
* 圖片檢視
* 行動裝置版面調整

需要特別說明：

> Viewer 顯示的是系統前處理後保存的教材頁面資料，不會直接開啟原始 PPT 或 PDF 檔案。

其目的在於讓使用者能從查詢或題目結果，快速查看系統所保存的對應教材頁面內容。

這對 OCR 特別重要。

若 OCR 對公式、符號、圖片文字或複雜版面辨識不完整，使用者仍可以直接查看該頁所保存的原始擷取圖片進行人工確認。

主要程式：

```text
backend/routes/materials.py
web/viewer.html
```

---

# 前端介面

目前前端主要包含：

```text
launcher.html
upload.html
preprocess.html
merge_json.html
search.html
practice_and_test.html
viewer.html
```

功能對應如下：

| 頁面                       | 功能            |
| ------------------------ | ------------- |
| `launcher.html`          | 系統入口          |
| `upload.html`            | 教材匯入          |
| `preprocess.html`        | 教材前處理         |
| `merge_json.html`        | 選擇複習範圍、建立索引   |
| `search.html`            | 找段落、AI 整理、關鍵字 |
| `practice_and_test.html` | 題目生成、練習、測驗    |
| `viewer.html`            | 教材來源頁面查看      |

手機版主要提供查詢、來源查看與練習相關功能。

教材匯入、前處理及複習範圍建立等管理操作，以電腦端操作為主。

---

# 專案結構

```text
local-material-ai-review-system/
│
├─ backend/
│  ├─ app.py
│  │
│  └─ routes/
│     ├─ upload.py
│     ├─ preprocess.py
│     ├─ merge_json.py
│     ├─ search.py
│     ├─ tips.py
│     ├─ exam.py
│     └─ materials.py
│
├─ scripts/
│  ├─ extract_text_and_image_and_ocr.py
│  ├─ make_chapter_jsonl.py
│  ├─ merge_jsonl.py
│  ├─ build_index.py
│  ├─ search_index_qa.py
│  ├─ make_chapter_tips.py
│  └─ make_exam_json.py
│
├─ web/
│  ├─ launcher.html
│  ├─ upload.html
│  ├─ preprocess.html
│  ├─ merge_json.html
│  ├─ search.html
│  ├─ practice_and_test.html
│  └─ viewer.html
│
├─ data/
│  └─ 執行時使用的教材、
│     結構化資料與向量索引
│
├─ requirements.txt
├─ start_system.bat
└─ stop_system.bat
```

---

# 使用技術

## 後端與資料處理

* Python
* FastAPI
* Uvicorn
* PyTorch
* Transformers
* NumPy

## 檢索

* `sentence-transformers/all-MiniLM-L6-v2`
* FAISS
* Exact Text Match
* Vector Retrieval
* Hybrid Retrieval

## AI

* Ollama
* Local LLM
* Retrieval-Augmented Generation

## 教材處理

* Tesseract OCR
* python-pptx
* pdfplumber
* pdf2image
* Pillow

## 前端

* HTML
* CSS
* JavaScript

---

# 執行環境

本專案主要於 Windows 與 Conda 環境開發。

## 1. 建立 Conda 環境

```bash
conda create -n llmenv python=3.11
conda activate llmenv
```

## 2. 安裝 Python 套件

```bash
pip install -r requirements.txt
```

PyTorch 請依本機 CUDA 或 CPU 環境安裝適合版本。

FAISS 可使用 Conda 安裝，例如：

```bash
conda install -c conda-forge faiss-cpu=1.9.0
```

---

## 3. 外部工具

部分功能需要另外安裝：

* Tesseract OCR
* Poppler
* Ollama
* 本地語言模型

---

## 4. 啟動系統

Windows 環境下可執行：

```text
start_system.bat
```

啟動後預設包含：

```text
FastAPI Backend
http://127.0.0.1:8000

Frontend
http://127.0.0.1:5500

System Launcher
http://127.0.0.1:5500/launcher.html
```

停止系統：

```text
stop_system.bat
```

---

# 公開 Repository 說明

由於課程教材涉及著作權，加上使用者資料、模型與執行後索引檔案可能具有較大容量，因此公開 Repository 不包含實際執行時的完整資料。

公開版本不包含：

* 原始課程教材
* 完整 OCR 後教材內容
* 使用者上傳教材
* 執行後建立的 FAISS 向量索引
* 本地語言模型權重
* 第三方工具安裝檔

目前 Repository 主要提供：

* 系統原始碼
* 教材前處理流程
* 向量索引建立流程
* 教材檢索程式
* RAG 整理流程
* 關鍵字功能
* AI 選擇題生成
* 練習與測驗介面
* 教材來源頁面 Viewer
* 前後端整合方式

實際教材查詢與操作案例另於專題成果文件中呈現。

---

# 目前限制

本專題目前以完整系統整合與實際教材操作為主要目標，仍存在以下限制。

## OCR

OCR 辨識品質會受到：

* 圖片解析度
* 字體
* 數學公式
* 特殊符號
* 表格
* 複雜排版

等因素影響。

因此系統同時保存教材圖片，供使用者人工確認。

---

## 檢索

向量檢索結果會受到：

* 教材文字品質
* OCR 品質
* Embedding 模型
* 查詢方式
* Top-k 設定
* 教材範圍

等因素影響。

語意相近的檢索結果也不代表一定符合使用者真正需要的證據內容。

---

## 生成式內容

本地語言模型仍可能：

* 整理不完整
* 誤解教材內容
* 產生不穩定結果
* 無法完全遵守提示規則

目前系統透過 prompt 要求模型只能依據 CONTEXT 回答，但尚未建立完整的逐句來源支持驗證。

---

## 評估

目前尚未完成：

* 大規模人工標註檢索資料集
* 大規模檢索準確率評估
* 生成品質量化評估
* 系統學習成效研究
* 完整 Evidence Attribution 評估

目前主要以實際教材操作、功能測試及人工確認進行系統驗證。

---

# 後續研究方向

在完成教材處理、檢索、內容整理、選擇題與來源頁面等系統功能後，本專題後續希望進一步分析：

> 當檢索設定發生改變時，即使系統仍找到相同文件，實際提供給模型的證據內容是否仍然一致？

後續預計研究：

* 不同 `top-k` 對檢索證據的影響
* 不同 `chunk size` 的影響
* 不同 `overlap` 的影響
* Document-level 與 Evidence-level 結果差異
* 必要證據是否仍被保留
* 證據排名變化
* 檢索失效來源
* 來源頁面與證據定位
* 系統延遲與檢索設定之間的取捨

目前研究延伸方向為：

**檢索增強生成系統於檢索設定變動下之證據連續性與失效來源分析**

希望從既有系統實作進一步延伸至可量化的檢索與證據穩定性研究。

---

# 作者

**劉泓億**

亞洲大學
資訊工程學系

畢業專題：

**本地化教材檢索與人工智慧輔助複習系統**
