#!/usr/bin/env python3
import matplotlib.pyplot as plt

# Data splits: counts for each split
sizes = [124012, 15512, 15498]
labels = ['Train (80.00%)', 'Test (10.01%)', 'Validation (10.00%)']

# Define custom colors (optional)
colors = ['#66b3ff', '#ff9999', '#99ff99']

# Explode the first slice a little for emphasis (optional)
explode = (0.1, 0, 0)

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.2f%%', startangle=90,
        colors=colors, explode=explode)
plt.title('Dataset Splits', fontsize=16)
plt.axis('equal')  # Ensures the pie is drawn as a circle
plt.savefig('dataset_splits_pie_chart.png', dpi=120)
plt.show()
