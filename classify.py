import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

torch.manual_seed(1)

# Data Augmentation
def augment_data(pose_data):
    # Flip along the y-axis (mirroring)
    mirrored_data = pose_data.copy()
    mirrored_data[:, :, 0] = -mirrored_data[:, :, 0]  # Flip x-coordinate
    return [pose_data, mirrored_data]

# Data Processing
def load_and_preprocess_data(data_dir, target_length=150):
    data = []
    labels = []
    player_names = []

    for file in os.listdir(data_dir):
        if file.endswith('.npy'):
            player_name = file.split('.')[0]  # Remove .npy extension
            if player_name not in player_names:
                player_names.append(player_name)

            pose_data = np.load(os.path.join(data_dir, file))
            pose_data = normalize_data(pose_data)
            pose_data = pad_sequence(pose_data, target_length)  # Pad sequences to the target length
            
            augmented_data = augment_data(pose_data)  # Augment the data
            for augmented_pose in augmented_data:
                data.append(augmented_pose)
                labels.append(player_names.index(player_name))
    
    return data, labels, player_names

def normalize_data(pose_data):
    # Center the pose around the root joint (usually hip)
    root_joint = pose_data[:, 0, :]
    pose_data = pose_data - root_joint[:, np.newaxis, :]
    
    # Scale normalization
    scale = np.sqrt(np.sum(np.square(pose_data), axis=(1, 2)))
    pose_data /= scale[:, np.newaxis, np.newaxis]
    
    return pose_data

# Padding function to make all sequences the same length
def pad_sequence(sequence, target_length):
    if len(sequence) < target_length:
        padding = np.zeros((target_length - len(sequence), sequence.shape[1], sequence.shape[2]))
        sequence = np.vstack((sequence, padding))
    return sequence[:target_length]

# Dataset and DataLoader
class SwingDataset(Dataset):
    def __init__(self, data, labels):
        # data and labels are lists
        self.data = [torch.FloatTensor(d) for d in data]  # Convert each sequence to a tensor
        self.labels = torch.LongTensor(labels)  # Convert labels to a tensor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SwingClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SwingClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_joints, 3)
        batch_size, sequence_length, num_joints, num_coordinates = x.shape
        
        # Flatten the last two dimensions (num_joints * num_coordinates)
        x = x.view(batch_size, sequence_length, num_joints * num_coordinates)
        
        # x shape after view: (batch_size, sequence_length, input_size)
        _, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        out = self.fc(out)
        return out

# Training and Evaluation Functions
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}")

def classify_new_swing(model, new_swing_data, player_names):
    model.eval()
    with torch.no_grad():
        new_swing_tensor = torch.FloatTensor(new_swing_data).unsqueeze(0)
        output = model(new_swing_tensor)
        _, predicted = torch.max(output, 1)
        return player_names[predicted.item()]

# Main Execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load and preprocess reference data (7 players)
    data_dir = "./athlete_videos_processed"
    data, labels, player_names = load_and_preprocess_data(data_dir, target_length=160)  # Set target length
    print(f"Loaded {len(data)} samples for {len(player_names)} players")
    print("Player names:", player_names)

    # Split data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Create DataLoaders
    train_loader = DataLoader(SwingDataset(train_data, train_labels), batch_size=8, shuffle=True)
    val_loader = DataLoader(SwingDataset(val_data, val_labels), batch_size=8)

    # Initialize model
    input_size = 17 * 3  # num_joints * 3 (x, y, z)
    hidden_size = 64  # Adjusted hidden size for better training
    num_classes = len(player_names)
    model = SwingClassifier(input_size, hidden_size, num_classes)

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer, num_epochs=1000)

    torch.save(model.state_dict(), 'model.pt')

    # Classify new user swings
    user_data_dir = "./user_videos_processed"
    for user_file in os.listdir(user_data_dir):
        if user_file.endswith('.npy'):
            user_swing_data = np.load(os.path.join(user_data_dir, user_file))
            user_swing_data = normalize_data(user_swing_data)
            user_swing_data = pad_sequence(user_swing_data, target_length=160)
            most_similar_player = classify_new_swing(model, user_swing_data, player_names)
            print(f"The swing in '{user_file}' is most similar to: {most_similar_player}")
