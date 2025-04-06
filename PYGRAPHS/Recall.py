#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Classification labels and corresponding Recall values
labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
recall = [0.8267, 0.8556, 0.7969, 0.9851, 0.9351, 0.7631, 0.8755]

# Number of labels
num_labels = len(labels)

# Compute angles for each label (in radians)
angles = np.linspace(0, 2 * np.pi, num_labels, endpoint=False).tolist()
angles += angles[:1]  # close the circle

# Append the first recall value to the end
recall += recall[:1]

# Create radar chart for Recall
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(angles, recall, color='orange', linewidth=2, label='Recall')
ax.fill(angles, recall, color='orange', alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
ax.set_ylim(0, 1)
ax.set_title("Recall per Label", size=16, y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('recall_radar_chart.png', dpi=120)
plt.show()
