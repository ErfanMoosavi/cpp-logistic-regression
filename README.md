# Logistic Regression from Scratch in C++  

## 📌 Overview  

**Logistic Regression from Scratch in C++** is a complete, modular implementation of **binary logistic regression** using the **Eigen** library for matrix operations.  
It provides a **command-line interface** for flexible configuration and outputs detailed evaluation metrics, making it suitable for hands-on learning and small-scale classification projects.  

---

## ✨ Features  

This project brings a full logistic regression pipeline, implemented entirely from scratch:  

- **Model**  
  - Binary logistic regression  
  - Configurable learning rate & epochs  
  - Optional **L2 regularization**  
  - Toggle for intercept/bias term  

- **Evaluation Metrics**  
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  

- **Data Handling**  
  - Direct CSV loading (train/test sets)  
  - Automatic header skipping  
  - Standardization (Z-score normalization)  
  - Save predictions to CSV  

- **Training Feedback**  
  - Cost printed during training  
  - Adjustable cost display interval  

---

## ⚙️ Technologies Used  

- **C++17** — Core language  
- **Eigen 3** — Matrix and linear algebra operations  
- **cxxopts** — Command-line argument parsing  

---

## 🗂 Project Structure  

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

## 🛠️ Build Instructions (Windows)  

### Prerequisites  
- MinGW with `mingw32-make` and `g++`  
- Eigen (downloaded or system-installed)  

### Build the Project  

```bash
mingw32-make
```

This produces `log_reg.exe` in the project root.  

To clean build files:  

```bash
mingw32-make clean
```

---

## 🚀 How to Run  

Run the program from the command line:  

```bash
log_reg.exe --x_train=Dataset/x_train.csv --y_train=Dataset/y_train.csv ^
--x_test=Dataset/x_test.csv --y_test=Dataset/y_test.csv ^
--lr=0.05 --epochs=200 --fit_intercept=true --l2=false ^
--print_cost=true --cost_interval=20 --output=predictions.csv
```

> ⚠️ On PowerShell, replace `^` with backticks `` for line continuation.  

---

## ⚡ CLI Options  

| Flag              | Description                               | Default    |
|-------------------|-------------------------------------------|------------|
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
| `--output`        | Save predictions to CSV file              | None       |

---

## 📥 Input Format  

- **CSV format** → Each row = 1 sample, columns = features (or label)  
- Required files:  
  - `x_train.csv` — Training features (M × N)  
  - `y_train.csv` — Training labels (M × 1)  
  - `x_test.csv`  — Testing features (K × N)  
  - `y_test.csv`  — Testing labels (K × 1)  

> Headers are skipped automatically.  

---

## 📊 Output  

- Training cost printed every *N* epochs (if enabled)  
- Final evaluation metrics:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  
- Predictions optionally saved to CSV via `--output`  

---

## 👨‍💻 Author  

**Erfan Moosavi**  
Student of Computer Engineering — passionate about AI, NLP, and philosophy.  

GitHub → [@ErfanMoosavi](https://github.com/ErfanMoosavi)  

---

## 📜 License  

This project is licensed under the MIT License — feel free to fork, star ⭐, and contribute!  
