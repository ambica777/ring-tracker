
fashion_yolo_prism - v5 2023-08-25 1:27am
==============================

This dataset was exported via roboflow.com on January 11, 2024 at 6:53 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 14562 images.
Fashion are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Random brigthness adjustment of between -50 and +50 percent
* Random exposure adjustment of between -50 and +50 percent
* Random Gaussian blur of between 0 and 3 pixels
* Salt and pepper noise was applied to 15 percent of pixels


