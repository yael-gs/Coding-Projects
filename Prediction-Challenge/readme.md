# Stock Classification from Tick-by-Tick Data

## Overview
A machine learning project to classify 100-event sequences of tick-by-tick order-book data into one of 24 stocks. This project focuses on time-series data handling and deep learning for classification.

## Objectives
- Handle time-series data effectively.
- Build a classification model for stock prediction.

## Dataset
- **Rows:** ~24 million (504 days × 24 stocks × 20 observations/day × 100 events/observation).
- **Features:**
  - `venue`, `action`, `price`, `bid`, `ask`, `bid_size`, `ask_size`, etc.
  - Derived features like logarithmic transformations and embeddings.
- **Target:** Stock label (24 classes).

## Tools
- TensorFlow, Keras, NumPy, Pandas, Matplotlib, Jupyter Notebook.

## Usage
conda env create -f spec-file.yml


