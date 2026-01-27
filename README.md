# Detecting RFI in interferometric data using machine learning

This project implements a deep learning model for segmenting Radio Frequency Interference (RFI) in radio astronomy data, specifically from the HERA telescope. It uses RNET5 for segmentation, with support for cross-validation, model saving, and evaluation.

## Repository Structure
The repo has two aspects, a simulated data (found in ```sim/```) training and a real data training (found in ```hera/```). The model is found in ```hera/src/rnet.py```.
## Requirements
This project is coded in python using tensorflow. All the requirements are listed in ```requirements.txt```. 

### Model training
If, for whatever reason, you wanted to train the nerual network I recommend using conda and running ```pip install tensorflow-gpu``` however setting up cuda and cudnn manually to meet with tensorflows requirements is a pain.
Alternatively you could train the model with your cpu but that will take forever.

## 📊 Metrics

Evaluation includes:

-  AUROC (Area Under ROC Curve)

- AUPRC (Area Under Precision-Recall Curve)

- F1 Score

## Acknowledgements 
A big thank you to the [HERA](https://reionization.org/) project for gathering and granting access to the data. A thank you to Charl Du Toits project found [here](https://github.com/CharlDuToit/RFI-NLN). Charl's methods for generating simulated Data were used for early model testings.
