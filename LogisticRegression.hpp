#ifndef LOGISTICREGRESSION_HPP
#define LOGISTICREGRESSION_HPP

#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

class LogisticRegression
{
public:
    // Constructors
    LogisticRegression();
    LogisticRegression(double learning_rate_, int num_of_iterations_, bool fit_intercept_, bool print_cost_, int cost_print_interval_);

    // Public methods
    void train(const std::vector<std::vector<double>> &x, const std::vector<double> &y);
    std::vector<double> predict(const std::vector<std::vector<double>> &x);
    std::vector<double> getCoefficients() const;
    double getBias() const;

private:
    // Private methods
    double binaryCrossEntropy(const std::vector<double> &y, const std::vector<double> &y_hat);
    std::vector<double> sigmoid(const std::vector<double> &linear_output);
    void initializeParameters(const int &num_of_features);
    std::vector<double> forwardProp(const std::vector<std::vector<double>> &x);
    void gradientDescent(const std::vector<std::vector<double>> &x, const std::vector<double> &y);
    void printCost(int i, const std::vector<double> &y, const std::vector<double> &y_hat);

    // Member variables
    std::vector<double> weights;
    double bias;
    double learning_rate;
    int num_of_iterations;
    bool fit_intercept;
    bool print_cost;
    int cost_print_interval;
};
#endif