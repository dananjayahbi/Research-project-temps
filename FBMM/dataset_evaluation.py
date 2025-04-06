#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def get_handpicked_samples(handpicked_path):
    """
    Get hand-picked sample images for each class.
    
    Args:
        handpicked_path: Path to the directory with hand-picked images
        
    Returns:
        Dictionary with class names as keys and lists of image paths as values
    """
    if not os.path.exists(handpicked_path):
        print(f"Hand-picked directory does not exist: {handpicked_path}")
        return None
    
    # Get class directories
    class_dirs = [d for d in os.listdir(handpicked_path) if os.path.isdir(os.path.join(handpicked_path, d))]
    
    samples = {}
    
    # For each class in the hand-picked directory
    for class_name in class_dirs:
        class_path = os.path.join(handpicked_path, class_name)
        
        # Get all image files
        image_files = []
        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)
            if os.path.isfile(file_path):
                extension = os.path.splitext(file)[1].lower()
                if extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    image_files.append(file_path)
        
        if image_files:
            samples[class_name] = image_files
            print(f"Found {len(image_files)} hand-picked images for class {class_name}")
        else:
            print(f"Warning: No images found for class {class_name}")
    
    return samples

def create_image_grid(samples):
    """
    Create a grid of sample images with minimal horizontal gaps.
    
    Args:
        samples: Dictionary with class names as keys and lists of image paths as values
    """
    num_classes = len(samples)
    num_samples = max(len(images) for images in samples.values())
    
    # Define width ratios: 
    # - first column for label is smaller (0.4)
    # - each image column has width ratio 1
    width_ratios = [0.4] + [1] * num_samples
    
    # Create figure and axes
    fig, axes = plt.subplots(
        nrows=num_classes,
        ncols=num_samples + 1,
        figsize=((num_samples * 2) + 2, num_classes * 2),
        gridspec_kw={'width_ratios': width_ratios}
    )
    
    # Manually adjust spacing
    # - very small wspace to reduce horizontal gap
    # - small margins around the figure
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.02, hspace=0.02)
    
    # Ensure axes is 2D even if only one class
    if num_classes == 1:
        axes = np.array([axes])  # shape (1, num_samples+1)
    
    for i, (class_name, image_paths) in enumerate(samples.items()):
        # First column: label
        axes[i, 0].text(
            0.9, 0.5, class_name, 
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=12,
            fontweight='bold',
            transform=axes[i, 0].transAxes
        )
        axes[i, 0].axis('off')
        
        # Subsequent columns: images
        for j, img_path in enumerate(image_paths):
            col_idx = j + 1  # offset by 1 for the label column
            try:
                img = Image.open(img_path)
                axes[i, col_idx].imshow(np.array(img), aspect='auto')
                axes[i, col_idx].axis('off')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                axes[i, col_idx].text(0.5, 0.5, "Error\nloading\nimage",
                                      horizontalalignment='center',
                                      verticalalignment='center',
                                      transform=axes[i, col_idx].transAxes)
                axes[i, col_idx].axis('off')
        
        # Turn off extra axes if this class has fewer images
        for j in range(len(image_paths) + 1, num_samples + 1):
            axes[i, j].axis('off')
    
    # Save and show
    plt.savefig('dataset_samples.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

if __name__ == "__main__":
    # Path to the hand-picked samples
    handpicked_path = "./hand-picked"
    
    # Get hand-picked samples for each class
    samples = get_handpicked_samples(handpicked_path)
    
    # Create and display the image grid
    if samples:
        create_image_grid(samples)
        print(f"Created image grid with hand-picked samples from {len(samples)} classes")
    else:
        print("Failed to get hand-picked sample images")
