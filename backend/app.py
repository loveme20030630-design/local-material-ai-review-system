from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes.materials import router as materials_router
from backend.routes.upload import router as upload_router
from backend.routes.preprocess import router as preprocess_router
from backend.routes.merge_json import router as merge_json_router
from backend.routes.exam import router as exam_router
from backend.routes.tips import router as tips_router
from backend.routes.search import router as search_router

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

app = FastAPI(title="LLM Project API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if DATA_ROOT.is_dir():
    app.mount("/static", StaticFiles(directory=str(DATA_ROOT)), name="static")

app.include_router(materials_router)
app.include_router(upload_router)
app.include_router(preprocess_router)
app.include_router(merge_json_router)
app.include_router(exam_router)
app.include_router(tips_router)
app.include_router(search_router)

@app.get("/")
def root():
    return {"status": "ok"}