#ifndef LOGISTICREGRESSION_HPP
#define LOGISTICREGRESSION_HPP

#include "metrics.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

using namespace std;

const double WEIGHT_INITIALIZATION_RATE = 0.01;
const double EPSILON = 1e-15;

class LogisticRegression
{
public:
    // Constructors
    LogisticRegression();
    LogisticRegression(double learning_rate_, int num_iterations_, bool fit_intercept_, bool l2_, double lambda_, bool print_cost_, int cost_print_interval_);

    // Public methods
    void train(const Eigen::MatrixXd &x, const Eigen::RowVectorXd &y);
    Eigen::RowVectorXd predict(const Eigen::MatrixXd &x) const;
    double accuracy(const Eigen::RowVectorXd &y, const Eigen::RowVectorXd &y_hat);
    double precision(const Eigen::RowVectorXd &y, const Eigen::RowVectorXd &y_hat);
    double recall(const Eigen::RowVectorXd &y, const Eigen::RowVectorXd &y_hat);
    double f1Score(const Eigen::RowVectorXd &y, const Eigen::RowVectorXd &y_hat);
    Eigen::VectorXd getCoefficients() const;
    double getBias() const;

private:
    // Private methods
    Eigen::VectorXd mean(const Eigen::MatrixXd &x) const;
    Eigen::VectorXd standardDeviation(const Eigen::MatrixXd &x) const;
    Eigen::MatrixXd standardScaler(const Eigen::MatrixXd &x) const;
    void initializeParameters(const int &num_of_features);
    Eigen::RowVectorXd forwardProp(const Eigen::MatrixXd &x) const;
    Eigen::RowVectorXd sigmoid(const Eigen::RowVectorXd &z) const;
    double binaryCrossEntropy(const Eigen::RowVectorXd &y, const Eigen::RowVectorXd &a) const;
    void gradientDescent(const Eigen::MatrixXd &x, const Eigen::RowVectorXd &y);
    void printCost(int i, const Eigen::RowVectorXd &y, const Eigen::RowVectorXd &a) const;

    // Member variables
    Metrics metrics;
    Eigen::VectorXd weights;
    double bias;
    double learning_rate;
    int num_iterations;
    bool fit_intercept;
    bool l2;
    double lambda;
    bool print_cost;
    int cost_print_interval;
};
#endif