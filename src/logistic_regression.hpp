#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include "metrics.hpp"
#include "data_handler.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

using namespace std;

const double WEIGHT_INITIALIZATION_RATE = 0.01;
const double EPSILON = 1e-15;
const string X_TRAIN = "x_train";
const string Y_TRAIN = "y_train";
const string X_TEST = "x_test";
const string Y_TEST = "y_test";

class LogisticRegression
{
public:
    LogisticRegression(DataHandler *data_handler_);
    LogisticRegression(DataHandler *data_handler_, double learning_rate_, int num_iterations_, bool fit_intercept_, bool l2_, double lambda_, bool print_cost_, int cost_print_interval_);

    void train();
    void predict() const;
    double accuracy();
    double precision();
    double recall();
    double f1Score();
    Eigen::MatrixXd getCoefficients() const;
    double getBias() const;
    Eigen::MatrixXd getPredictions();

private:
    Eigen::MatrixXd standardScaler(const Eigen::MatrixXd &x, bool is_train) const;
    void initializeParameters(const int &num_of_features);
    Eigen::MatrixXd forwardProp(const Eigen::MatrixXd &x) const;
    Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &z) const;
    double binaryCrossEntropy(const Eigen::MatrixXd &y, const Eigen::MatrixXd &a) const;
    void gradientDescent(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y);
    void printCost(int i, const Eigen::MatrixXd &y, const Eigen::MatrixXd &a) const;

    DataHandler *data_handler;
    Metrics metrics;
    double learning_rate;
    int num_iterations;
    bool fit_intercept;
    bool l2;
    double lambda;
    bool print_cost;
    int cost_print_interval;
    Eigen::MatrixXd weights;
    double bias;
mutable Eigen::RowVectorXd mean_train;
mutable Eigen::RowVectorXd stddev_train;
};

#endif