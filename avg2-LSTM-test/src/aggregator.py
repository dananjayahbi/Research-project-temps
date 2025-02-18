import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=7, hidden_size=32, num_layers=2, output_size=7):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take last timestep output
        return out

class EmotionAggregator:
    def __init__(self, window_size=10, save_path="emotion_data.json"):
        self.window_size = window_size
        self.emotion_records = []
        self.emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.model = LSTMPredictor()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.save_path = save_path

    def add_emotion(self, emotion_dict):
        """Store emotion records as a sequence for time-series learning."""
        self.emotion_records.append(list(emotion_dict.values()))
        if len(self.emotion_records) > self.window_size:
            self.emotion_records.pop(0)  # Keep only the last N records
            self.train_lstm()
            self.save_to_json()

    def train_lstm(self):
        """Train LSTM on emotion sequences."""
        if len(self.emotion_records) < self.window_size:
            return
        
        data = torch.tensor([self.emotion_records], dtype=torch.float32)  # Shape: (1, seq_len, 7)
        target = torch.tensor(self.emotion_records[-1], dtype=torch.float32)  # Predict last step

        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        
        print(f"LSTM Training Loss: {loss.item():.4f}")

    def predict_next_emotion(self):
        """Predict the next emotional state using the trained LSTM."""
        if len(self.emotion_records) < self.window_size:
            return None  # Not enough data for prediction

        data = torch.tensor([self.emotion_records], dtype=torch.float32)  # Shape: (1, seq_len, 7)
        predicted = self.model(data).detach().numpy()[0]
        return {label: val for label, val in zip(self.emotion_labels, predicted)}

    def save_to_json(self):
        """Save emotion sequence and prediction to JSON."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        predicted = self.predict_next_emotion()

        # Convert NumPy floats to Python floats
        new_entry = {
            "timestamp": timestamp,
            "aggregated_emotions": {label: float(val) for label, val in zip(self.emotion_labels, self.emotion_records[-1])},
            "predicted_next": {label: float(val) for label, val in predicted.items()} if predicted else {}
        }

        # Append new entry to JSON file
        try:
            with open(self.save_path, "r") as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(new_entry)

        with open(self.save_path, "w") as file:
            json.dump(data, file, indent=4)

        print(f"Predicted Next Emotions: {new_entry['predicted_next']}")
