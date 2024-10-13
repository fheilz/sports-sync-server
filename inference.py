# inference.py

import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import mediapipe as mp

# Model definition
class SwingClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SwingClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_joints, num_coordinates)
        batch_size, sequence_length, num_joints, num_coordinates = x.shape
        
        # Flatten the last two dimensions (num_joints * num_coordinates)
        x = x.view(batch_size, sequence_length, num_joints * num_coordinates)
        
        # Pass through LSTM
        _, (h_n, _) = self.lstm(x)
        
        # Pass the last hidden state through the fully connected layer
        out = self.fc(h_n[-1])
        return out

# Function to load the trained model
def load_model(model_path, input_size, hidden_size, num_classes):
    model = SwingClassifier(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Normalize pose data
def normalize_data(pose_data):
    # Center the pose around the root joint (assumed to be the first joint)
    root_joint = pose_data[:, 0, :]
    pose_data = pose_data - root_joint[:, np.newaxis, :]
    
    # Scale normalization
    scale = np.sqrt(np.sum(np.square(pose_data), axis=(1, 2), keepdims=True))
    pose_data /= scale
    return pose_data

# Pad or truncate the sequence to the target length
def pad_sequence(sequence, target_length):
    current_length = sequence.shape[0]
    if current_length < target_length:
        padding = np.zeros((target_length - current_length, sequence.shape[1], sequence.shape[2]))
        sequence = np.vstack((sequence, padding))
    else:
        sequence = sequence[:target_length]
    return sequence

# Retrieve player names from the data directory
def get_player_names(data_dir):
    player_names = []
    for file in os.listdir(data_dir):
        if file.endswith('.npy'):
            player_name = file.split('.')[0]  # Extract player name from filename
            if player_name not in player_names:
                player_names.append(player_name)
    return player_names

# Function to extract pose data from video
def extract_pose_from_video(video_path, target_length=160):
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    pose_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB before processing.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_estimator.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_data = []
            for lm in landmarks:
                frame_data.extend([lm.x, lm.y, lm.z])
            pose_sequence.append(frame_data)
        else:
            # If no landmarks detected, append zeros
            pose_sequence.append([0.0] * 51)  # 17 joints * 3 coordinates

    cap.release()
    pose_estimator.close()

    # Convert to numpy array
    pose_array = np.array(pose_sequence)

    # Pad or truncate to target_length
    pose_array = pad_sequence(pose_array, target_length)

    # Reshape to (sequence_length, num_joints, num_coordinates)
    pose_array = pose_array.reshape(target_length, 17, 3)  # Assuming 17 joints

    return pose_array

# Load model and player names at module load
MODEL_PATH = './model.pt'
ATHLETE_DATA_DIR = './athlete_videos_processed'

player_names = get_player_names(ATHLETE_DATA_DIR)
num_classes = len(player_names)
input_size = 17 * 3  # Number of joints * coordinates (x, y, z)
hidden_size = 64     # Must match the hidden size used during training

model = load_model(MODEL_PATH, input_size, hidden_size, num_classes)
print("Model loaded successfully.")
print(f"Detected players: {player_names}")

# Function to process video and return predicted player name
def process_video(video_path):
    """
    Processes the input video file and returns the predicted player name.

    Args:
        video_path (str): Path to the .mp4 video file.

    Returns:
        str: Predicted player name.
    """
    # Extract pose data from the video
    pose_data = extract_pose_from_video(video_path)
    
    # Normalize the pose data
    pose_data = normalize_data(pose_data)
    
    # Convert to tensor and add batch dimension
    processed_pose = torch.FloatTensor(pose_data).unsqueeze(0)  # Shape: (1, sequence_length, num_joints, num_coordinates)
    
    # Run the model to get the prediction
    with torch.no_grad():
        output = model(processed_pose)
        _, predicted = torch.max(output, 1)
        predicted_index = predicted.item()
    
    # Ensure the predicted index is within the range of player_names
    if predicted_index >= len(player_names):
        raise IndexError(f"Predicted index {predicted_index} is out of range for player names.")
    
    return player_names[predicted_index]

