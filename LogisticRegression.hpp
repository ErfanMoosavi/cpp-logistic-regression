#ifndef LOGISTICREGRESSION_HPP
#define LOGISTICREGRESSION_HPP

#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

const double WEIGHT_INITIALIZATION_RATE = 0.01;
const double EPSILON = 1e-15;

class LogisticRegression
{
public:
    // Constructors
    LogisticRegression();
    LogisticRegression(double learning_rate_, int num_of_iterations_, bool fit_intercept_, bool l2_, double lambda_, bool print_cost_, int cost_print_interval_);

    // Public methods
    void train(const std::vector<std::vector<double>> &x, const std::vector<double> &y);
    std::vector<double> predict(const std::vector<std::vector<double>> &x);
    std::vector<double> getCoefficients() const;
    double getBias() const;

private:
    // Private methods
    std::vector<double> mean(const std::vector<std::vector<double>> &x) const;
    std::vector<double> standardDeviation(const std::vector<std::vector<double>> &x) const;
    std::vector<std::vector<double>> standardScaler(const std::vector<std::vector<double>> &x) const;
    void initializeParameters(const int &num_of_features);
    std::vector<double> forwardProp(const std::vector<std::vector<double>> &x) const;
    std::vector<double> sigmoid(const std::vector<double> &linear_output) const;
    double binaryCrossEntropy(const std::vector<double> &y, const std::vector<double> &y_hat) const;
    void gradientDescent(const std::vector<std::vector<double>> &x, const std::vector<double> &y);
    void printCost(int i, const std::vector<double> &y, const std::vector<double> &y_hat) const;

    // Member variables
    std::vector<double> weights;
    double bias;
    double learning_rate;
    int num_of_iterations;
    bool fit_intercept;
    bool l2;
    double lambda;
    bool print_cost;
    int cost_print_interval;
};
#endif