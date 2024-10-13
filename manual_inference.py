import os
import numpy as np
import torch
import torch.nn as nn

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

# Function to preprocess video data
def preprocess_video_data(video_data, target_length=160):
    video_data = normalize_data(video_data)
    video_data = pad_sequence(video_data, target_length)
    return torch.FloatTensor(video_data).unsqueeze(0)  # Add batch dimension

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

# Global variables for model and player names
MODEL_PATH = './model.pt'
ATHLETE_DATA_DIR = './athlete_videos_processed'
USER_VIDEO_DIR = './user_videos_processed'

# Initialize model and player names at startup
player_names = get_player_names(ATHLETE_DATA_DIR)
num_classes = len(player_names)
input_size = 17 * 3  # Number of joints * coordinates (x, y, z)
hidden_size = 64     # Must match the hidden size used during training

model = load_model(MODEL_PATH, input_size, hidden_size, num_classes)
print("Model loaded successfully.")
print(f"Detected players: {player_names}")

# Function to run the model on a single video and get the predicted player name
def process_video(video_path):
    """
    Processes the input video and returns the predicted player name.

    Args:
        video_path (str): Path to the .npy video file.

    Returns:
        str: Predicted player name.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' does not exist.")
    
    # Load video data
    video_data = np.load(video_path)
    
    # Preprocess the video data
    processed_video = preprocess_video_data(video_data)
    
    # Run the model to get the prediction
    with torch.no_grad():
        output = model(processed_video)
        _, predicted = torch.max(output, 1)
        predicted_index = predicted.item()
    
    # Ensure the predicted index is within the range of player_names
    if predicted_index >= len(player_names):
        raise IndexError(f"Predicted index {predicted_index} is out of range for player names.")
    
    return player_names[predicted_index]

# Main execution
if __name__ == "__main__":
    # Example usage: Process all videos in the user_videos_processed directory
    for video_file in os.listdir(USER_VIDEO_DIR):
        if video_file.endswith('.npy'):
            video_path = os.path.join(USER_VIDEO_DIR, video_file)
            try:
                predicted_player = process_video(video_path)
                print(f"The swing in '{video_file}' is most similar to: {predicted_player}")
            except Exception as e:
                print(f"Error processing '{video_file}': {e}")

