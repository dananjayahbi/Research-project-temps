# src/aggregator.py

import time

class EmotionAggregator:
    def __init__(self, window_seconds=15 * 60, callback=None):
        """
        window_seconds: Aggregation window duration (default 15 minutes; use a shorter window for testing).
        callback: A function to call with the aggregated results once the window is over.
        """
        self.window_seconds = window_seconds
        self.start_time = time.time()
        self.emotion_records = []
        self.emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.callback = callback

    def add_emotion(self, emotion_dict):
        """Add a new prediction (a dictionary of emotion probabilities) to the records."""
        self.emotion_records.append(emotion_dict)
        if time.time() - self.start_time >= self.window_seconds:
            aggregated = self.compute_average()
            if self.callback:
                self.callback(aggregated)
            else:
                print("\n=== Aggregated Emotion Confidence (Last Window) ===")
                for label, value in aggregated.items():
                    print(f"{label}: {value * 100:.2f}%")
                print("====================================================\n")
            self.start_time = time.time()
            self.emotion_records = []

    def compute_average(self):
        """Calculate and return the average emotion probabilities over the window."""
        avg_emotions = {label: 0 for label in self.emotion_labels}
        count = len(self.emotion_records)
        if count == 0:
            return avg_emotions
        for record in self.emotion_records:
            for label in self.emotion_labels:
                avg_emotions[label] += record[label]
        for label in avg_emotions:
            avg_emotions[label] /= count
        return avg_emotions
