import shutil

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

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
    
    return {"message": f"File {file_name} uploaded"}
