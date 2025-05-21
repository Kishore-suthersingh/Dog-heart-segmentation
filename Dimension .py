import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops

# Function to convert pixels to cm using PPI
def pixels_to_cm(pixels, ppi=300):
    cm_per_pixel = 2.54 / ppi  # Corrected conversion factor: 1 inch = 2.54 cm
    return pixels * cm_per_pixel

# Function to process X-ray and mask to find heart dimensions with center lines
def find_heart_dimensions_center_lines(xray_path, mask_path, ppi=300):
    # Load images
    xray = cv2.imread(xray_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the mask to binary
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours of the segmented heart
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour (assuming it's the heart)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the rotated bounding box of the contour
        rect = cv2.minAreaRect(largest_contour)
        (center, (width, height), angle) = rect
        center_point = (int(center[0]), int(center[1]))
        width_px = int(width)
        height_px = int(height)

        # Determine length and breadth based on the dimensions of the rotated bounding box
        length_px = max(width_px, height_px)
        breadth_px = min(width_px, height_px)
        length_cm = pixels_to_cm(length_px, ppi)
        breadth_cm = pixels_to_cm(breadth_px, ppi)

        # Calculate endpoints for length line
        length_angle_rad = np.deg2rad(angle) if height_px > width_px else np.deg2rad(angle + 90)
        length_endpoint1 = (int(center_point[0] - (length_px / 2) * np.cos(length_angle_rad)),
                            int(center_point[1] - (length_px / 2) * np.sin(length_angle_rad)))
        length_endpoint2 = (int(center_point[0] + (length_px / 2) * np.cos(length_angle_rad)),
                            int(center_point[1] + (length_px / 2) * np.sin(length_angle_rad)))

        # Calculate endpoints for breadth line
        breadth_angle_rad = np.deg2rad(angle + 90) if height_px > width_px else np.deg2rad(angle)
        breadth_endpoint1 = (int(center_point[0] - (breadth_px / 2) * np.cos(breadth_angle_rad)),
                             int(center_point[1] - (breadth_px / 2) * np.sin(breadth_angle_rad)))
        breadth_endpoint2 = (int(center_point[0] + (breadth_px / 2) * np.cos(breadth_angle_rad)),
                             int(center_point[1] + (breadth_px / 2) * np.sin(breadth_angle_rad)))

        # Draw the length and breadth lines on the X-ray image
        cv2.line(xray, length_endpoint1, length_endpoint2, (0, 255, 0), 2)  # Green for length
        cv2.line(xray, breadth_endpoint1, breadth_endpoint2, (255, 0, 0), 2) # Blue for breadth

        # Annotate measurements near the lines
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        thickness = 2
        line_offset = 10

        # Position length annotation
        length_midpoint_x = (length_endpoint1[0] + length_endpoint2[0]) // 2
        length_midpoint_y = (length_endpoint1[1] + length_endpoint2[1]) // 2
        cv2.putText(xray, f"Length: {length_cm:.2f} cm", (length_midpoint_x + line_offset, length_midpoint_y - line_offset), font, font_scale, font_color, thickness, cv2.LINE_AA)

        # Position breadth annotation
        breadth_midpoint_x = (breadth_endpoint1[0] + breadth_endpoint2[0]) // 2
        breadth_midpoint_y = (breadth_endpoint1[1] + breadth_endpoint2[1]) // 2
        cv2.putText(xray, f"Breadth: {breadth_cm:.2f} cm", (breadth_midpoint_x + line_offset, breadth_midpoint_y + 2 * line_offset), font, font_scale, font_color, thickness, cv2.LINE_AA)


        # Save and display result
        cv2.imwrite("annotated_xray_lines.png", xray)

        # Show image
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(xray, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("X-ray with Center Lines for Length and Breadth")
        plt.show()

        print(f"Annotated image with center lines saved as 'annotated_xray_lines.png'")
        print(f"Detected Heart Length (center lines): {length_cm:.2f} cm")
        print(f"Detected Heart Breadth (center lines): {breadth_cm:.2f} cm")

        return length_cm, breadth_cm

    else:
        print("No heart detected in the segmentation mask.")
        return None, None

xray_path = "Enter the input image path" 
mask_path = "Enter the correct mask image path"  
length, breadth = find_heart_dimensions_center_lines(xray_path, mask_path)