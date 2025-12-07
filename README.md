# GMSFNet: A Global-Local Fusion Network for Surface Water Extraction from Multi-Source Sentinel-1/2 Data under Cloudy Conditions — A Case Study in Luzhou, Sichuan

# GMSFNet
Semantic segmentation of multi-source water body data

## Program Language
The primary programming languages used in this project are:

 **Python 3.12.3**: All of the core functionalities, including data loading, model training, and evaluation, are written in Python.

## Software Required

To run this project, you will need the following software:

- **Python 3.12.3**: Ensure that Python is installed and accessible in your environment.
- **PyTorch 2.0.0+cu118**: PyTorch is used as the deep learning framework. Installation will depend on your hardware configuration (CPU or GPU). 
- **Anaconda/Miniconda (optional)**: Recommended for managing Python environments.
- **CUDA (optional, for GPU acceleration)**: If you are using GPU, you will need the correct CUDA version installed for PyTorch.

## Usage
This repository provides the implementation of the semantic segmentation model
(`GMSFNet.py`), the loss function (`loss.zip`), the evaluation/inference script
(`evaluator.py`), and a small test dataset (`test_sample.zip`) and the prediction results of the test set(`test_reslut_predict.zip`)
 
## Contents of this repository
GMSFNet.py: implementation of the proposed semantic segmentation model GMSFNet.
loss.zip: loss function implementation used in training GMSFNet.
Please unzip it to obtain loss.py (or a loss/ folder), which is imported by the model/training code.
evaluator.py: evaluation and inference script. It loads a trained GMSFNet model,
runs inference on the test dataset, computes metrics (IoU, F1, recall, precision, specificity),
and saves prediction and comparison maps.
test_sample.zip: a small subset of the test dataset used in the paper
(sample test images and corresponding ground‑truth labels).
test_result_predict.zip: prediction results of the test subset produced by the trained model in the paper.

## Data
- The data is a self-made multi-source water body dataset, which can be provided according to requirements
## License

This project is licensed under the MIT License


## Contact

For any questions or suggestions, feel free to reach out:

- **Name**: Dongchen Yao
- **Email**: yaodongchen@stu.cdut.edu.cn
