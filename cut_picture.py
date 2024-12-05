import cv2
import numpy as np
import os

def cut_image(input_image_path):
    image = cv2.imread(input_image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use morphological operations to clean the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Apply contours on the cleaned image
    contours, hierarchy = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an output directory for segmented objects
    output_dir = "segmented_objects_precise"
    os.makedirs(output_dir, exist_ok=True)

    # Set a minimum size threshold for filtering
    min_area = 3000  # Adjust this value based on your objects

    # Loop through contours and save only valid objects
    object_count = 0
    segmented_files = []
    for contour in contours:
        # Filter by contour area
        area = cv2.contourArea(contour)
        if area >= min_area:
            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract the object with some padding
            padding = 10  # Add padding to the bounding box
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(image.shape[1], x + w + padding), min(image.shape[0], y + h + padding)
            object_image = image[y1:y2, x1:x2]
            
            # Save the segmented object
            object_path = os.path.join(output_dir, f"object_{object_count}.png")
            cv2.imwrite(object_path, object_image)
            segmented_files.append(object_path)
            object_count += 1

    print(f"Segmented and saved {object_count} objects. Saved in {output_dir}.")
    
