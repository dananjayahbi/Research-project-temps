#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Classification labels and corresponding Precision values
labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
precision = [0.8095, 0.8758, 0.7991, 0.9878, 0.9108, 0.8026, 0.8505]

# Number of labels
num_labels = len(labels)

# Compute angle for each axis in the plot (in radians)
angles = np.linspace(0, 2 * np.pi, num_labels, endpoint=False).tolist()
angles += angles[:1]  # complete the loop

# Append the first precision value to close the radar chart
precision += precision[:1]

# Create radar chart for Precision
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(angles, precision, color='blue', linewidth=2, label='Precision')
ax.fill(angles, precision, color='blue', alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
ax.set_ylim(0, 1)
ax.set_title("Precision per Label", size=16, y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('precision_radar_chart.png', dpi=120)
plt.show()
