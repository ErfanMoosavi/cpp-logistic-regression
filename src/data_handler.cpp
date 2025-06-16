#include "data_handler.hpp"

void DataHandler::loadCSV(const string &path, const string &type)
{
    ifstream file(path);
    vector<vector<double>> values;
    string line;

    getline(file, line); // Read header line

    while (getline(file, line))
    {
        stringstream ss(line);
        string cell;
        vector<double> row;

        while (getline(ss, cell, ','))
            row.push_back(stod(cell));

        values.push_back(row);
    }

    size_t rows = values.size();
    size_t cols = values[0].size();

    Eigen::MatrixXd mat(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            mat(i, j) = values[i][j];

if (type == "x_train")
    x_train = mat;
else if (type == "y_train")
    y_train = mat.col(0);
else if (type == "x_test")
    x_test = mat;
else if (type == "y_test")
    y_test = mat.col(0);
}

const Eigen::MatrixXd& DataHandler::getXTrain() { return x_train; }
const Eigen::MatrixXd& DataHandler::getYTrain() { return y_train; }
const Eigen::MatrixXd& DataHandler::getXTest() { return x_test; }
const Eigen::MatrixXd& DataHandler::getYTest() { return y_test; }
const Eigen::MatrixXd& DataHandler::getPreds() { return preds; }
void DataHandler::setYTest(Eigen::MatrixXd y_test_) { y_test = y_test_; }
void DataHandler::setPreds(Eigen::MatrixXd preds_) { preds = preds_; }