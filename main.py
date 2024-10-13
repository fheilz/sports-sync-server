# main.py

import os
import shutil
import tempfile
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from inference import process_video

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "API is running"}


@app.post("/process")
def process(file: UploadFile = File(...)):
    file_name = "user_videos_processed/" + file.filename.rsplit(".", 1)[0] + ".npy"
    with open(file.filename, "wb") as f:
        shutil.copyfileobj(file.file, f)

    predicted_player = process_video(file_name)
    player_name = f"{predicted_player.split('_')[0].capitalize()} {predicted_player.split('_')[1].capitalize()}"
    
    # Clean up: remove the temporary video file
    if os.path.exists(file.filename):
        os.remove(file.filename)
    
    # Prepare and return the response
    response = {
        "player": player_name,
        "file_name": f"{predicted_player}.mp4",
    }
    
    return JSONResponse(content=response)

@app.get("/download/{file_name}")
def download(file_name: str):
    if os.path.exists(f"athlete_videos_processed/{file_name}"):
        return FileResponse(path=f"athlete_videos_processed/{file_name}", media_type="video/mp4", filename=file_name)
    return JSONResponse(content={"message": "File not found"}, status_code=404)