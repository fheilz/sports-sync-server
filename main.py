# main.py

import os
import shutil
import uuid
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
async def process_file(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.lower().endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported.")
    
    # Generate a unique temporary file name
    unique_id = uuid.uuid4().hex
    temp_dir = tempfile.gettempdir()
    temp_video_path = os.path.join(temp_dir, f"{unique_id}.mp4")
    
    try:
        # Save the uploaded file to the temporary directory
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the video and get the predicted player name
        predicted_player = process_video(temp_video_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {e}")
    
    finally:
        # Clean up: remove the temporary video file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
    
    # Prepare and return the response
    response = {
        "player": predicted_player,
        "file_name": file.filename
    }
    
    return JSONResponse(content=response)

