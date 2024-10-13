import os
import shutil

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/process")
def process(file: UploadFile = File(...)):
    file_name = file.filename.rsplit(".", 1)[0] + ".mp4"
    with open(file_name, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    response = {
        "player": "Test Player",
        "file_name": "test_player.mp4",
    }
    
    return JSONResponse(content=response)

@app.get("/download/{file_name}")
def download(file_name: str):
    if os.path.exists(file_name):
        return FileResponse(path=file_name, media_type="video/mp4", filename=file_name)
    return JSONResponse(content={"message": "File not found"}, status_code=404)