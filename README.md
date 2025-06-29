# Logistic Regression from Scratch in C++

This project is a C++ implementation of **logistic regression** using the **Eigen** library for matrix operations. It is a complete, modular system for binary classification tasks, with a command-line interface for configurable usage.

---

## Features

### Model
- Implements binary logistic regression from scratch
- Configurable learning rate, number of iterations
- Optional L2 regularization
- Fit bias/intercept toggle

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

### Data Handling
- Load CSV files directly (train/test sets)
- Save CSV output file (preds.css)
- Standardization (Z-score normalization)
- Handles header skipping automatically

### Training Feedback
- Cost (loss) printed during training
- Customizable cost display interval

---

## Technologies Used

- C++17 — Core language
- Eigen 3 — Linear algebra operations (matrix math)
- cxxopts — Command-line argument parsing

---

## Project Structure

```
/ProjectRoot
├── src/
│   ├── main.cpp
│   ├── logistic_regression.cpp
│   └── data_handler.cpp
├── include/
│   └── cxxopts.hpp
├── Eigen/
│   └── (Eigen headers)
├── Dataset/
│   ├── x_train.csv
│   ├── y_train.csv
│   ├── x_test.csv
│   └── y_test.csv
├── Build/
│   └── (generated object files)
├── Makefile
└── README.md
```

---

## Build Instructions (Windows)

### Prerequisites
- MinGW with `mingw32-make` and `g++`
- Eigen (downloaded or system-installed)

### Build the Project

Open Command Prompt in the project folder:

```bash
mingw32-make
```

This builds `log_reg.exe` in the project root.

To clean up build files:

```bash
mingw32-make clean
```

---

## How to Run the Program

Run the program from the command line like this:

```bash
log_reg.exe --x_train=Dataset/x_train.csv --y_train=Dataset/y_train.csv ^
--x_test=Dataset/x_test.csv --y_test=Dataset/y_test.csv ^
--lr=0.05 --epochs=200 --fit_intercept=true --l2=false ^
--print_cost=true --cost_interval=20 --output=predictions.csv
```

> On PowerShell, replace `^` with backticks ` for line continuation.

---

## CLI Options

| Flag              | Description                               | Default    |
|------------------|-------------------------------------------|------------|
| `--x_train`       | Path to training features CSV              | Required   |
| `--y_train`       | Path to training labels CSV                | Required   |
| `--x_test`        | Path to test features CSV                  | Required   |
| `--y_test`        | Path to test labels CSV                    | Required   |
| `--lr`            | Learning rate                             | `0.01`     |
| `--epochs`        | Number of iterations                      | `100`      |
| `--fit_intercept` | Whether to learn a bias term              | `true`     |
| `--l2`            | Enable L2 regularization                  | `false`    |
| `--lambda`        | L2 regularization strength                | `0.1`      |
| `--print_cost`    | Print cost during training                | `false`    |
| `--cost_interval` | Interval to print cost                    | `20`       |
| `--output`        | 	Save predictions to CSV file            | `20`       |

---

## Input Format

- CSV Format: Each row = 1 sample, columns = features (or label)
- Files Needed:
  - `x_train.csv` — Features for training (M × N)
  - `y_train.csv` — Labels for training (M × 1)
  - `x_test.csv`  — Features for testing (K × N)
  - `y_test.csv`  — Labels for testing (K × 1)

> The first row (headers) will be skipped automatically.

---

## Output

- Training cost is printed every N epochs (if enabled)
- Final evaluation metrics printed after prediction:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Evaluation metrics are printed after prediction.
- If --output is specified, predictions are saved to the provided CSV file.

---

## Author

**Erfan Moosavi**  
Student of Computer Engineering — passionate about AI, NLP, and philosophical thinking.

GitHub: [@ErfanMoosavi](https://github.com/ErfanMoosavi)

---

Feel free to fork, star, or contribute!