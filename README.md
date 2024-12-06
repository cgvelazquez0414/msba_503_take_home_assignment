from ultralytics import YOLO
from PIL import Image
import os

pip install ultralytics

model = YOLO("yolov8m.pt")

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Define the folder path
folder_path = '/Users/carlosvelazquez/Desktop/MSBA/Analytical Programming II/images'

# Initialize a list to store results for the DataFrame
data = []

# Iterate through all images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Ensure it's an image file
        image_path = os.path.join(folder_path, filename)
        
        # Predict the objects in the image
        results = model.predict(image_path)
        result = results[0]

        # Display the image with predictions
        annotated_image = result.plot()
        plt.imshow(annotated_image)
        plt.axis('off')  # Turn off the axes for better visualization
        plt.title(f"Predictions for {filename}")
        plt.show()


        # Extract information for the DataFrame
        if result.boxes:  # Ensure there are detected boxes
            for box in result.boxes:
                class_id = int(box.cls.cpu().numpy())  # Convert class index to integer
                class_name = result.names[class_id]  # Map class ID to class name
                coordinates = box.xyxy.cpu().numpy().tolist()  # Bounding box coordinates
                probability = float(box.conf.cpu().numpy())  # Confidence score

                data.append({
                    "Filename": filename,
                    "Object Type": class_name,
                    "Coordinates": coordinates,
                    "Probability": probability
                })

# Create the DataFrame
yolo_results = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_save_path = os.path.join(folder_path, "predictions_results.csv")
yolo_results.to_csv(csv_save_path, index=False)

# Print the DataFrame to console
print("Predictions Results:")
yolo_results


import os
from PIL import Image
import torch
import pandas as pd
from torchvision import transforms
import torchvision.models.detection as detection
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F

Load the pre-trained Faster R-CNN model
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# Dynamically fetch COCO class names
COCO_CLASSES = weights.meta["categories"]

# Path to the folder containing the images
folder_path = '/Users/carlosvelazquez/Desktop/MSBA/Analytical Programming II/images'

# Ensure the input folder exists
if not os.path.exists(folder_path):
    print(f"Folder '{folder_path}' does not exist!")
    exit()

# Confidence threshold for filtering predictions
confidence_threshold = 0.5

# List to store results for the DataFrame
results_list = []

for filename in os.listdir(folder_path):
    # Construct the full path to the image
    image_path = os.path.join(folder_path, filename)

    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    print(f"Processing image: {image_path}")

    # Step 2: Load the image
    try:
        image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format
    except Exception as e:
        print(f"Error loading image {filename}: {e}")
        continue

    # Convert the image to a tensor
    image_tensor = F.to_tensor(image)

    # Step 3: Perform object detection
    with torch.no_grad():
        predictions = model([image_tensor])

    # Step 4: Extract and filter predictions
    predicted_boxes = predictions[0]['boxes']
    predicted_scores = predictions[0]['scores']
    predicted_labels = predictions[0]['labels']

    for box, score, label in zip(predicted_boxes, predicted_scores, predicted_labels):
        if score >= confidence_threshold:
            # Convert box to a list of coordinates
            coordinates = box.tolist()
            class_name = COCO_CLASSES[label.item()]
            probability = score.item()

            # Append result to the list
            results_list.append({
                "Filename": filename,
                "Object Type": class_name,
                "Coordinates": coordinates,
                "Probability": probability
            })

    # Optional: Display the image with annotations
    fig, ax = plt.subplots(1, figsize=(12, 8), dpi=72)
    ax.imshow(image)

    for box, score, label in zip(predicted_boxes, predicted_scores, predicted_labels):
        if score >= confidence_threshold:
            x_min, y_min, x_max, y_max = box.tolist()
            class_name = COCO_CLASSES[label.item()]

            # Draw the bounding box
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

            # Add the class name and score as text
            ax.text(
                x_min,
                y_min - 5,
                f"{class_name}: {score:.2f}",
                color='white',
                fontsize=10,
                bbox=dict(facecolor='red', alpha=0.5)
            )

    plt.axis("off")  # Hide axes for better visualization
    plt.show()

# Step 5: Create a DataFrame from the results list
rnn_results = pd.DataFrame(results_list)

rnn_results



import cv2
import os
import pandas as pd

# Define the folder path containing the images
folder_path = '/Users/carlosvelazquez/Desktop/MSBA/Analytical Programming II/images'

# Initialize a list to hold the metadata for all images
keypoint_data = []

# Iterate through all image files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
        # Construct the full path to the image
        image_path = os.path.join(folder_path, filename)
        
        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # ORB works with grayscale images
        
        # Initialize the ORB detector with a higher nfeatures limit
        orb = cv2.ORB_create(nfeatures=500)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(image, None)
        
                # Draw keypoints on the image
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Display the image with keypoints
        plt.figure(figsize=(8, 8))
        plt.imshow(image_with_keypoints, cmap='gray')
        plt.title(f"ORB Features - {filename}")
        plt.axis('off')
        plt.show()
        
        # Collect metadata for each keypoint
        for i, keypoint in enumerate(keypoints):
            keypoint_data.append({
                "Image": filename,
                "Keypoint #": i + 1,
                "Location (x, y)": keypoint.pt,  # Keypoint coordinates
                "Scale (size)": keypoint.size,  # Keypoint region size
                "Orientation (angle)": keypoint.angle,  # Dominant gradient direction

            })

# Convert the metadata to a DataFrame
df_keypoints = pd.DataFrame(keypoint_data)

# Display the DataFrame
print(df_keypoints)



yolo_results["above_90_percent"] = (yolo_results["Probability"] > 0.90).astype(int)
yolo_results["below_90_percent"] = (yolo_results["Probability"] <= 0.90).astype(int)

yolo_counts = yolo_results.groupby(["Filename", "Object Type"]).size().reset_index(name="Count")

# Calculate the sum of the 'above_90_percent' column
sum_above_90 = yolo_results['above_90_percent'].sum()
sum_below_90 = yolo_results['below_90_percent'].sum()
sum_count_yolo = yolo_counts['Count'].sum()
unique_object_types = yolo_counts['Object Type'].unique()
num_unique_objects = len(unique_object_types)

# Print the results
print(f"Sum of 'above_90_percent': {sum_above_90}")
print(f"Sum of 'below_90_percent': {sum_below_90}")
print(f"Sum of 'Objects Noticed': {sum_count_yolo}")
print(f"Sum of 'Unqiue Objects Noticed': {num_unique_objects}")

data = {
    "Metric": ["Sum of 'above_90_percent'", "Sum of 'below_90_percent'", "Sum of 'Objects Noticed'", "Sum of 'Unique Objects Noticed'"],
    "Value": [sum_above_90, sum_below_90, sum_count_yolo, num_unique_objects]
}

yolo_summary = pd.DataFrame(data)
yolo_summary

rnn_results["above_90_percent"] = (rnn_results["Probability"] > 0.90).astype(int)
rnn_results["below_90_percent"] = (rnn_results["Probability"] <= 0.90).astype(int)

rnn_counts = rnn_results.groupby(["Filename", "Object Type"]).size().reset_index(name="Count")

# Calculate the sum of the 'above_90_percent' column
sum_above_90 = rnn_results['above_90_percent'].sum()
sum_below_90 = rnn_results['below_90_percent'].sum()
sum_count_yolo = rnn_counts['Count'].sum()
unique_object_types = rnn_counts['Object Type'].unique()
num_unique_objects = len(unique_object_types)

# Print the results
print(f"Sum of 'above_90_percent': {sum_above_90}")
print(f"Sum of 'below_90_percent': {sum_below_90}")
print(f"Sum of 'Objects Noticed': {sum_count_yolo}")
print(f"Sum of 'Unqiue Objects Noticed': {num_unique_objects}")

data = {
    "Metric": ["Sum of 'above_90_percent'", "Sum of 'below_90_percent'", "Sum of 'Objects Noticed'", "Sum of 'Unique Objects Noticed'"],
    "Value": [sum_above_90, sum_below_90, sum_count_yolo, num_unique_objects]
}

rnn_summary = pd.DataFrame(data)
rnn_summary

both_summary = {
    "Model": ["YOLO", "RNN"],
    "Detect Above 90%": [8, 59],
    "Detect Below 90%": [89, 55],
    "Objects Noticed": [97, 114],
    "Unique Objects Noticed": [12, 16]
}

both_summary = pd.DataFrame(both_summary)

both_summary

import matplotlib.pyplot as plt

# Loop through all columns in the DataFrame except the first one (e.g., 'Model')
for column in both_summary.columns[1:]:
    plt.figure(figsize=(8, 5))
    
    # Adjust bar width and align positions for closer bars
    bar_positions = range(len(both_summary['Model']))
    bar_width = .9  # Adjust bar width for closer spacing
    
    # Create the bar chart with specified colors
    bars = plt.bar(bar_positions, both_summary[column], color=['orange', 'blue'], width=bar_width)
    
    # Add labels to each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            height, 
            f'{height:.0f}', 
            ha='center', 
            va='bottom', 
            fontsize=10
        )
    
    # Customize x-axis labels and ticks
    plt.xticks(bar_positions, both_summary['Model'])
    
    plt.title(f"Bar Chart of {column}", fontsize=14)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.tight_layout()
    plt.show()

    





