# Logistic Regression from Scratch

This project is a C++ implementation of logistic regression using the Eigen library for matrix operations. It supports binary classification, CSV-based data loading, evaluation metrics, and feature normalization. The system is modular and scalable, designed with future extensibility in mind.

## Features

- **Logistic Regression Model**
  - Implements binary logistic regression from scratch
  - Configurable learning rate, number of iterations, and regularization
  - Supports optional L2 regularization

- **Data Handling**
  - Loads training and test data directly from CSV files
  - Separates feature matrices and label vectors
  - Normalizes data using standard score (z-score) scaling

- **Model Evaluation**
  - Evaluates predictions using:
    - Accuracy
    - Precision
    - Recall
    - F1 Score

- **Training Feedback**
  - Tracks cost (loss) during training
  - Configurable interval-based printing of training cost

## Technologies Used

- **C++** — Core model implementation, logic, and optimization
- **Eigen** — Matrix and vector operations for numerical computation
- **Custom CSV Parser** — Reads and parses CSV data into Eigen matrices

## Dependencies

- [Eigen 3](https://eigen.tuxfamily.org/) (Header-only C++ template library for linear algebra)

### How to Install Eigen

- **Option 1:** Install system-wide using a package manager

  **Ubuntu/Debian**
  ```bash
  sudo apt install libeigen3-dev
  ```

  **macOS (Homebrew)**
  ```bash
  brew install eigen
  ```

- **Option 2:** Download manually

  - Download from: https://eigen.tuxfamily.org/
  - Extract the folder and place it somewhere accessible
  - Include it in your build commands, e.g.:
    ```bash
    g++ -I /path/to/eigen main.cpp -o my_program
    ```

## System Architecture

- `LogisticRegression` class:
  - Encapsulates all model logic: training, predicting, evaluation, etc.
  - Includes gradient descent, forward propagation, and sigmoid activation
  - Supports regularization and cost tracking

- `DataHandler` class:
  - Responsible for reading CSV data and managing feature/label separation
  - Provides accessors for training and testing data

- `Metrics` class:
  - Calculates confusion matrix and derived evaluation metrics
  - Used after prediction for model performance measurement

## Usage Overview

1. **Prepare Your CSV Files**
   - Format: Each row = one sample; each column = one feature (except label in `y_*.csv`)
   - Ensure no malformed or non-numeric values
   - First line (header) will be skipped automatically

2. **Load the Data**
   - Use the `DataHandler` to load CSVs:

   ```cpp
   DataHandler dh;
   dh.loadCSV("data/x_train.csv", "x_train");
   dh.loadCSV("data/y_train.csv", "y_train");
   dh.loadCSV("data/x_test.csv", "x_test");
   dh.loadCSV("data/y_test.csv", "y_test");
   ```

3. **Train and Predict**
   - Create the model and train:

   ```cpp
   LogisticRegression model(&dh, 0.05, 200, true, true, 0.1, true, 10);
   model.train();
   model.predict();
   ```

4. **Evaluate Results**
   - After predictions:

   ```cpp
   std::cout << "Accuracy: " << model.accuracy() << std::endl;
   std::cout << "Precision: " << model.precision() << std::endl;
   std::cout << "Recall: " << model.recall() << std::endl;
   std::cout << "F1 Score: " << model.f1Score() << std::endl;
   ```

## Input Format

- **CSV File Types**
  - `x_train.csv` — Feature matrix for training (M × N)
  - `y_train.csv` — Labels for training (M × 1)
  - `x_test.csv` — Feature matrix for testing (K × N)
  - `y_test.csv` — Labels for testing (K × 1)

- **Note:** The program assumes rows are samples and columns are features (standard ML convention)

## Output Design

- **Training Logs:**
  - Prints cost at specified intervals during training
- **Metric Output:**
  - Evaluation metrics are printed to the console after prediction
- **Modular Design:**
  - Easily extendable for multiclass classification, new evaluation metrics, or additional input formats
