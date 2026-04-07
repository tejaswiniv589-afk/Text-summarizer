import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import subprocess
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, HTMLResponse
import uvicorn
from text_summarizer.pipeline.prediction import PredictionPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ✅ Load model ONCE when server starts — not on every request
pipeline = PredictionPipeline()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train")
async def training():
    try:
        result = subprocess.run(
            ["python", "main.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return Response("Training successful!!")
        else:
            return Response(f"Training failed:\n{result.stderr}")
    except Exception as e:
        return Response(f"Error occurred: {e}")

@app.post("/predict", response_class=HTMLResponse)
async def predict_route(request: Request, text: str = Form(...)):
    try:
        # ✅ Use the already-loaded pipeline — no reload
        summary = pipeline.predict(text)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "summary": summary,
            "original_text": text
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e),
            "original_text": text
        })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)