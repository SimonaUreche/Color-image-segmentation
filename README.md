# Region Growing Image Segmentation

## Project Overview
Implementation of a **region growing algorithm** for color image segmentation in L\*u\*v\* color space. The pipeline includes preprocessing (Gaussian filtering), adaptive thresholding, and postprocessing (morphological operations) to achieve accurate segmentations.

---

## Key Features
| Feature | Description |
|---------|-------------|
| Gaussian Filtering | 5×5 low-pass filter for noise reduction |
| Color Conversion | RGB → L\*u\*v\* (perceptually uniform space) |
| Region Growing | Seed-based expansion using Euclidean distance |
| Adaptive Threshold | `T = value × √(σᵤ² + σᵥ²)` |
| Postprocessing | Erosion + dilation for noise removal |
| Recoloring | Region coloring by mean RGB values |
