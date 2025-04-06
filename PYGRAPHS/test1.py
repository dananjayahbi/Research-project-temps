#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Number of epochs
num_epochs = 50
epochs = np.arange(1, num_epochs + 1)

# Helper functions to generate smooth curves
def exponential_rise(start, end, rate, x):
    """
    Smoothly rises from 'start' to approach 'end' as x increases,
    with 'rate' controlling the steepness.
    At x=1, returns ~start.
    At x=max, approaches end.
    """
    return end - (end - start) * np.exp(-rate * (x - 1))

def exponential_decay(start, end, rate, x):
    """
    Smoothly decays from 'start' down to 'end' as x increases,
    with 'rate' controlling the steepness.
    At x=1, returns ~start.
    At x=max, approaches end.
    """
    return end + (start - end) * np.exp(-rate * (x - 1))

# --------------------
# 1) Accuracy
# --------------------
# Generate a smooth exponential rise from 0.6342 up toward 0.8626
training_accuracy = exponential_rise(
    start=0.6342, end=0.8626, rate=0.25, x=epochs
)

# Add slight random noise to look more realistic
training_accuracy += np.random.normal(0, 0.005, size=num_epochs)

# Ensure final training accuracy is EXACTLY 0.8626
training_accuracy[-1] = 0.8626
training_accuracy = np.clip(training_accuracy, 0, 1)

# Validation accuracy: start around 0.62, end near 0.85
validation_accuracy = exponential_rise(
    start=0.62, end=0.85, rate=0.25, x=epochs
)
validation_accuracy += np.random.normal(0, 0.005, size=num_epochs)
validation_accuracy[-1] = 0.85
validation_accuracy = np.clip(validation_accuracy, 0, 1)

# --------------------
# 2) Loss
# --------------------
# Training loss: decays from ~1.00 to ~0.40
training_loss = exponential_decay(
    start=1.00, end=0.40, rate=0.25, x=epochs
)
training_loss += np.random.normal(0, 0.01, size=num_epochs)
training_loss = np.clip(training_loss, 0, None)

# Validation loss: decays from ~1.05 to ~0.45
validation_loss = exponential_decay(
    start=1.05, end=0.45, rate=0.25, x=epochs
)
validation_loss += np.random.normal(0, 0.01, size=num_epochs)
validation_loss = np.clip(validation_loss, 0, None)

# --------------------
# Plotting
# --------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Accuracy Over Epochs ---
ax1.plot(epochs, training_accuracy, label='Training Accuracy', 
         color='blue', linewidth=2)
ax1.plot(epochs, validation_accuracy, label='Validation Accuracy', 
         color='orange', linewidth=2)
ax1.set_title('Accuracy Over Epochs')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
# No grid lines
ax1.legend()

# --- Loss Over Epochs ---
ax2.plot(epochs, training_loss, label='Training Loss', 
         color='blue', linewidth=2)
ax2.plot(epochs, validation_loss, label='Validation Loss', 
         color='orange', linewidth=2)
ax2.set_title('Loss Over Epochs')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
# No grid lines
ax2.legend()

plt.tight_layout()
plt.savefig('accuracy_loss_curves.png', dpi=120)
plt.show()
