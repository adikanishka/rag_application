# app.py (single combined FastAPI app)
import os
import glob
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from rag_pipeline import build_vector_store, get_answer

app = FastAPI()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Startup event
@app.on_event("startup")
def startup_event():
    print("Cleaning uploads folder...")
    cleanup_uploads()
    print("Building default vector store...")
    build_vector_store()
    print("Default vector store built!")

def cleanup_uploads():
    files = glob.glob(os.path.join(UPLOAD_DIR, "*"))
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Failed to delete {f}: {e}")

def parse_bool(value: str) -> bool:
    return str(value).lower() in ("true", "1", "yes")

# Serve frontend
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API route
@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    use_custom: str = Form("false"),
    file: UploadFile = File(None)
):
    use_custom_bool = parse_bool(use_custom)
    custom_path = None

    if use_custom_bool:
        if not file:
            raise HTTPException(status_code=400, detail="use_custom is True but no file uploaded.")
        custom_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(custom_path, "wb") as f:
            f.write(await file.read())

    answer = get_answer(question, use_custom=use_custom_bool, custom_path=custom_path)
    return JSONResponse(content={"answer": answer})



