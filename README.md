# Brain Tumor Semantic Segmentation

## Overview

This project to is dedicated to Semantic Segmentation of brain tumors, which involves classifying every pixel in an image as part of a tumor or non-tumor region. 
It uses data from an external source which must be downloaded manually, then processed before running the main program.

---

## 📁 Project Structure

BrainTumorSemanticSegmentation/ 
├── Dataset/  # Folder for storing the dataset and related scripts
├── Model/    # Contains classes to define unet model and saved models with relative training logs
├── unet.py   # Define the important parts to run the training and testing processes
├── main.py   # Entry point for running the program 
└── README.md

---

## 📥 Dataset Setup

1. **Download the Dataset**  
   Download the required dataset from [**Dataset Database Name**](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation/).

2. **Add the Dataset**  
   Place the downloaded dataset folders into the `Dataset/` folder in the root of the project directory. The final structure should look like this:

   Dataset/
   ├── test/
   ├── train/
   ├── valid/
   ├── coco_to_masks.py
   └── resize_images.py

   
---

## ⚙️ Install Dependencies

Make sure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

---

## 🧹 Process the Data

Before running the main program, process the raw dataset:
1. **Create the masks**
   
   ```bash
    python main.py data --masks
    ```

3. **Crop the border of all images to make them 512*512**
   
   ```bash
   python main.py data --resize
   ```

---

## 🚀 Run the Program

You can now run the program:

1. **Training mode**
   ```bash
   python main.py train 
   ```
   This command has 2 customizable parameters:
   1. *--epochs* to set the number of training epochs to train for
   2. *--existing-data* a boolean to choose if you want to train from existing saved model *brain_tumor_model.keras*

   Example:
   ```bash
   python main.py train --epochs 50
   python main.py train --epochs 10 --existing-data true
   ```
   
3. **Testing mode**
   ```bash
   python main.py test 
   ```
   This command has a customizable parameters:
   *--n-elements* the random number of elements to train on and display when comparing it to real image and mask

   Example:
   ```bash
   python main.py train --n-elements 5
   ```

---

## 📝 Notes

Ensure the structure is preserved as expected.












