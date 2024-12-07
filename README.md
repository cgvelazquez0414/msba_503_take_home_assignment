# Project Description:
I utilized two deep learning image detection algorithms, YOLO (You Only Look Once) and R-CNN (Region-based Convolutional Neural Network), along with one non-deep learning algorithm, ORB (Oriented FAST and Rotated BRIEF), to detect objects and keypoints in images. YOLO’s main advantage lies in its speed and ability to perform real-time object detection, but it has limitations in handling small objects or overlapping detections in complex scenes. R-CNN, on the other hand, produced more accurate results due to its ability to focus on regions of interest and perform finer-grained detections, though it requires significantly more computational resources and time for processing.

The non-deep learning algorithm, ORB, is effective at finding keypoints and descriptors, which can be used for tasks like object recognition or image matching. Additionally, ORB provides valuable metadata, including location, scale, and orientation, to help analyze image features. To synthesize the findings, I summarized and compared the performance of the YOLO and R-CNN models through visualizations, offering clearer insights into their strengths, weaknesses, and overall effectiveness.

## Libraries imported
from ultralytics import YOLO
from PIL import Image
import os

pip install ultralytics

## model used
model = YOLO("yolov8m.pt")

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

#folder path
folder_path = '/Users/carlosvelazquez/Desktop/MSBA/Analytical Programming II/images'

data = []


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

yolo_results = pd.DataFrame(data)

csv_save_path = os.path.join(folder_path, "predictions_results.csv")
yolo_results.to_csv(csv_save_path, index=False)

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

COCO_CLASSES = weights.meta["categories"]

folder_path = '/Users/carlosvelazquez/Desktop/MSBA/Analytical Programming II/images'

if not os.path.exists(folder_path):
    print(f"Folder '{folder_path}' does not exist!")
    exit()

confidence_threshold = 0.5

results_list = []

for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)

    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    print(f"Processing image: {image_path}")

    try:
        image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format
    except Exception as e:
        print(f"Error loading image {filename}: {e}")
        continue

    image_tensor = F.to_tensor(image)

    with torch.no_grad():
        predictions = model([image_tensor])

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

rnn_results = pd.DataFrame(results_list)

rnn_results



import cv2
import os
import pandas as pd

folder_path = '/Users/carlosvelazquez/Desktop/MSBA/Analytical Programming II/images'

keypoint_data = []

for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
        # Construct the full path to the image
        image_path = os.path.join(folder_path, filename)
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # ORB works with grayscale images
        
        orb = cv2.ORB_create(nfeatures=500)
        
        keypoints, descriptors = orb.detectAndCompute(image, None)
        
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(image_with_keypoints, cmap='gray')
        plt.title(f"ORB Features - {filename}")
        plt.axis('off')
        plt.show()
        
        for i, keypoint in enumerate(keypoints):
            keypoint_data.append({
                "Image": filename,
                "Keypoint #": i + 1,
                "Location (x, y)": keypoint.pt,  # Keypoint coordinates
                "Scale (size)": keypoint.size,  # Keypoint region size
                "Orientation (angle)": keypoint.angle,  # Dominant gradient direction

            })

df_keypoints = pd.DataFrame(keypoint_data)

print(df_keypoints)



yolo_results["above_90_percent"] = (yolo_results["Probability"] > 0.90).astype(int)
yolo_results["below_90_percent"] = (yolo_results["Probability"] <= 0.90).astype(int)

yolo_counts = yolo_results.groupby(["Filename", "Object Type"]).size().reset_index(name="Count")

sum_above_90 = yolo_results['above_90_percent'].sum()
sum_below_90 = yolo_results['below_90_percent'].sum()
sum_count_yolo = yolo_counts['Count'].sum()
unique_object_types = yolo_counts['Object Type'].unique()
num_unique_objects = len(unique_object_types)

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

sum_above_90 = rnn_results['above_90_percent'].sum()
sum_below_90 = rnn_results['below_90_percent'].sum()
sum_count_yolo = rnn_counts['Count'].sum()
unique_object_types = rnn_counts['Object Type'].unique()
num_unique_objects = len(unique_object_types)

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

    





