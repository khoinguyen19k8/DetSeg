# About
Osteolytic lesions are small, heterogeneous and can grow outside of bones. Hence, it is difficult to train an object segmentation model directly on CT scans.
Moreau et al. (2020) proposed segmenting bone tissue first to improve lesion segmentation performance. However, it only imrpoves performance on PET/CT instead
of only CT scans. An idea that could solve this problem is to localize potential lesions prior to segmentation. The traditional method that follows this idea is Mask R-CNN, which utilizes
a Region Proposal Network (RPN) to localise potential regions before segmenting. In this project I propose DetSeg, an architecture
that utilizes an object detection model to localize lesions before segmenting them with an object segmentation model. Specifically, Yolov5 + 2D U-Net are 
used in this project. Mask R-CNN is also implemented for comparison. The data is provided by Elisabeth-TweeSteden Ziekenhuis (ETZ) hospital in Tilburg, the Netherlands.
This project is a part of the "Implementation of an optimized AI model for the detection and monitoring of osteolytic bone lesions" - a WeCare collaboration project between Tilburg University and ETZ hospital <br>

# Dataset
The CT scans used in this thesis belong to Elizabeth-TweeSteden Hospital (ETZ) in Tilburg, Netherlands. The patients in each scan are 18 years or
older. There are 96 full-body CT scans from 79 patients acquired by various scanners, each with a maximum of 20 lesions. The scans are DICOM
images with a resolution of either 768 x 768 or 512 x 512 pixels, which are then combined to make 3D axial slices. <br>

<p align="center">
  <img src="https://github.com/khoinguyen19k8/DetSeg/blob/main/figures/data/CTP10_001_Slices.png">
</p>

<p align="center">
  <img src="https://github.com/khoinguyen19k8/DetSeg/blob/main/figures/data/big-lesions-and-ground-truth.png">
</p>

<p align="center">
  <img src="https://github.com/khoinguyen19k8/DetSeg/blob/main/figures/data/lesion_dist.png">
</p>

# Pre-processing and Data Transformation

<p align="center">
  <img src="https://github.com/khoinguyen19k8/DetSeg/blob/main/figures/pipeline/data-augmentation-pipeline.jpg">
</p>

# Models
Yolov5 is trained on the original dataset, then 192x192 patches are cropped around detected lesions. U-Net is then trained on those cropped images.

## DetSeg
<br>
<p align="center">
  <img src="https://github.com/khoinguyen19k8/DetSeg/blob/main/figures/models/DSeg.png">
</p>

## Mask R-CNN
<br>
<p align="center">
  <img src="https://github.com/khoinguyen19k8/DetSeg/blob/main/figures/models/mask-rcnn-architecture.jpg">
</p>

# Results

## Lesion detection
<br>
<p align="center">
  <img src="https://github.com/khoinguyen19k8/DetSeg/blob/main/figures/results/lesion_detection_results.jpg">
</p>

## Lesion segmentation
<br>
<p align="center">
  <img src="https://github.com/khoinguyen19k8/DetSeg/blob/main/figures/results/lesion_segmentation_results.jpg">
</p>

## Speed benchmark
The first figure is Yolov5 speed benchmark, the second is 2D U-Net speed benchmark.

<p align="center">
  <img src="https://github.com/khoinguyen19k8/DetSeg/blob/main/figures/results/benchmark.png">
</p>

<p align="center">
  <img src="https://github.com/khoinguyen19k8/DetSeg/blob/main/figures/results/segmentation_benchmark.png">
</p>

# Visualization

<p align="center">
  <img src="https://github.com/khoinguyen19k8/DetSeg/blob/main/figures/results/yolov5_vis.png">
</p>

<p align="center">
  <img src="https://github.com/khoinguyen19k8/DetSeg/blob/main/figures/results/preds_categorized_01.png">
</p>

# Demo

https://github.com/khoinguyen19k8/DetSeg/blob/main/videos/detection_side_by_side.wav

https://github.com/khoinguyen19k8/DetSeg/blob/main/videos/segmentation_side_by_side.wav


# References
1. Nguyen, K.Q. (2022). DetSeg - Segmentation based on Osteolytic Lesions localisation in Multiple Myeloma patients.
2. Moreau, N., Rousseau, C., Fourcade, C., Santini, G., Ferrer, L., Lacombe, M., . . . Normand, N. (2020). Deep learning approaches for bone and bone lesion segmentation on 18 FDG PET/CT imaging in the context of metastatic breast cancer*.
3. Hoff, W. (2021). Automated segmentation of osteolytic lesions in whole-body CT imaging of multiple myeloma patients using deep learning models.

# License

# Contact
More details of the project are detailed in "thesis_final_v7.pdf". If you want to get access to the code please contact me by email because there is a confidential policy where I developed the project.
Nguyen Quang Khoi

k.q.nguyen@tilburguniversity.edu
khoinguyen19k8@gmail.com
