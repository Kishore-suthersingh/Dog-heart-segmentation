🐶 Dog Heart Dimension Detector

This project aims to automatically determine the length and breadth of a dog's heart from X-ray images using deep learning and computer vision techniques.

📌 Objective

The primary goal is to build a robust system that can:

* Annotate the dog’s heart from X-ray images.
* Train a segmentation model to detect the heart.
* Predict heart regions in unseen images.
* Accurately calculate the length and breadth of the detected heart.

🧰 Workflow Overview

1. Annotation

   * Annotate dog heart regions using the CVAT (Computer Vision Annotation Tool).
   * Export the annotations in Segmentation Mask 1.1 format.

2. Data Preprocessing

   * Convert the segmentation masks to contours, then to polygon format.
   * These polygons are formatted for training a segmentation model.

3. Model Training

   * Utilize YOLOv8n-seg.pt (a lightweight pretrained segmentation model).
   * Train the model using annotated polygon data for better localization and segmentation.

4. Prediction

   * Use the trained model to predict heart segmentation on test X-ray images.
   * Run the prediction script: `predict.py`.

5. Measurement Extraction

   * Calculate length and breadth of the predicted heart region using `dimension.py`.
   * The calculations are based on the segmented polygon contour.

 🚀 Getting Started

   1. Setup Environment

        Make sure `ultralytics` is installed for YOLOv8:

        pip install ultralytics

   2. Annotate Using CVAT

        * Open CVAT.
        * Annotate heart regions.
        * Export in Segmentation Mask 1.1 format.

   3. Convert to Polygon Format

        Run your custom script to:

        * Read segmentation masks.
        * Extract contours.
        * Convert them into polygons usable for YOLOv8 training.

   4. Train the Model

         python train.py

            Make sure to adjust the config for dataset paths and epochs.

   5. Predict on New Images

        python predict.py --source path_to_image_or_folder --weights yolov8n-seg.pt


   6. Calculate Length and Breadth

        python dimension.py --input path_to_predicted_mask


        The script outputs the **length and breadth in centimeters**, assuming a fixed pixel-to-cm ratio or calibration method is included.


📊 Example Output


    Image: xray_001.png
    Predicted Heart Length: 6.4 cm
    Predicted Heart Breadth: 4.1 cm

🛠️ Technologies Used

    * CVAT – Annotation tool
    * YOLOv8 (Ultralytics) – Segmentation model
    * OpenCV, NumPy – Image processing
    * Python – Programming language


