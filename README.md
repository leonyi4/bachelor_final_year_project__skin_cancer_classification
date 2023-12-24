# Bachelor Final Year Project

**Hello, I am Nyi Nyi Nyein Aung (Leo),** and this repository contains my Bachelor's final year project in the field of computer vision.

## Project Overview

I aimed to explore the domain of computer vision before completing my undergraduate studies. Specifically, I focused on utilizing _Image Classification_ techniques to classify skin cancer lesions.

## Dataset and Model

- Trained a _Convolutional Neural Network (CNN)_ model using _drop-out regularization_ on the `MNIST Skin Cancer HAM10000` dataset.
- Achieved an accuracy of **75%** in classifying different skin cancer types.

## Project Contents

- The notebooks for dataset exploration, model training, and model demonstration are available in the [Notebooks Folder](./Notebooks).

## Project Report

- My partner and I compiled a comprehensive **37-page report** titled _"Drop-out Regularization: Enhancing Skin Cancer Classification with CNN"_.
- The report comprises:
  1. In-depth literature review on skin cancer diagnosis methods and CNNs in medical image analysis.
  2. Description of the CNN model architecture, data preprocessing steps, and training strategies.
  3. Analysis of results using various metrics including _ROC curve_, _confusion matrix_, and _classification report_, providing insights into the model's performance on different skin cancer lesions.

## Gradio Web Interface

- Utilized _Gradio_ to develop a **Web Interface** for the trained model, enabling users to classify any skin cancer image.
- The interface is hosted locally and can be replicated by downloading `app.py` and `best_model.keras` from this repository.
