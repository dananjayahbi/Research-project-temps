#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Classification labels and corresponding F1 Score values
labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
f1_score = [0.8180, 0.8656, 0.7980, 0.9864, 0.9228, 0.7823, 0.8628]

# Number of labels
num_labels = len(labels)

# Compute angles for the radar chart (in radians)
angles = np.linspace(0, 2 * np.pi, num_labels, endpoint=False).tolist()
angles += angles[:1]  # complete the loop

# Append the first F1 score value to close the radar chart
f1_score += f1_score[:1]

# Create radar chart for F1 Score
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(angles, f1_score, color='green', linewidth=2, label='F1 Score')
ax.fill(angles, f1_score, color='green', alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
ax.set_ylim(0, 1)
ax.set_title("F1 Score per Label", size=16, y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('f1score_radar_chart.png', dpi=120)
plt.show()
