#include "LogisticRegression.hpp"

const double WEIGHT_INITIALIZATION_RATE = 0.01;

LogisticRegression::LogisticRegression()
    : learning_rate(0.01), num_of_iterations(100), fit_intercept(true), print_cost(false), cost_print_interval(20)
{
}

LogisticRegression::LogisticRegression(double learning_rate_, int num_of_iterations_, bool fit_intercept_, bool print_cost_, int cost_print_interval_)
    : learning_rate(learning_rate_), num_of_iterations(num_of_iterations_), fit_intercept(fit_intercept_), print_cost(print_cost_), cost_print_interval(cost_print_interval_)
{
}

void LogisticRegression::train(const std::vector<std::vector<double>> &x, const std::vector<double> &y)
{
    
    if (x.empty() || y.empty())
    {
        std::cerr << "Error: Input data (x or y) is empty." << std::endl;
        return;
    }
    
    if (x.size() != y.size())
    {
        throw std::runtime_error("Error: Input size mismatch");
    }

    initializeParameters(x[0].size());

    for (int i = 0; i <= num_of_iterations; i++)
    {
        gradientDescent(x, y);
        printCost(i, y, forwardProp(x));
    }
}

std::vector<double> LogisticRegression::predict(const std::vector<std::vector<double>> &x)
{
    std::vector<double> y_hat = forwardProp(x);
    for_each(y_hat.begin(), y_hat.end(), [](double &p)
             { p = p >= 0.5 ? 1 : 0; });
    return y_hat;
}

std::vector<double> LogisticRegression::getCoefficients() const { return weights; }

double LogisticRegression::getBias() const { return bias; }

double LogisticRegression::binaryCrossEntropy(const std::vector<double> &y, const std::vector<double> &y_hat)
{
    int num_samples = y.size();
    double sum = 0.0;
    double epsilon = 1e-9; // Small value to prevent log(0)

    for (int i = 0; i < num_samples; i++)
    {
        double y_hat_clipped = std::max(epsilon, std::min(1.0 - epsilon, y_hat[i]));
        sum += y[i] * log(y_hat_clipped) + (1 - y[i]) * log(1 - y_hat_clipped);
    }

    return -sum / num_samples;
}

std::vector<double> LogisticRegression::sigmoid(const std::vector<double> &linear_output)
{
    int num_samples = linear_output.size();
    std::vector<double> y_hat(num_samples);

    for (int i = 0; i < num_samples; i++)
    {
        y_hat[i] = 1.0 / (1.0 + exp(-linear_output[i]));
    }

    return y_hat;
}

void LogisticRegression::initializeParameters(const int &num_of_features)
{
    weights.resize(num_of_features);
    for (int j = 0; j < num_of_features; j++)
    {
        weights[j] = (rand() % 10) * WEIGHT_INITIALIZATION_RATE;
    }
    bias = 0.0;
}

std::vector<double> LogisticRegression::forwardProp(const std::vector<std::vector<double>> &x)
{
    int num_samples = x.size();
    int num_of_features = x[0].size();
    std::vector<double> linear_output(num_samples, 0.0);

    for (int i = 0; i < num_samples; i++)
    {
        for (int j = 0; j < num_of_features; j++)
        {
            linear_output[i] += weights[j] * x[i][j];
        }
        if (fit_intercept)
            linear_output[i] += bias;
    }

    return sigmoid(linear_output);
}

void LogisticRegression::gradientDescent(const std::vector<std::vector<double>> &x, const std::vector<double> &y)
{
    int num_samples = x.size();
    int num_of_features = weights.size();
    std::vector<double> y_hat = forwardProp(x);
    std::vector<double> linear_output_gradients(num_samples);
    std::vector<double> weight_gradients(num_of_features);

    for (int i = 0; i < num_samples; i++)
    {
        linear_output_gradients[i] = y_hat[i] - y[i];
    }

    for (int j = 0; j < num_of_features; j++)
    {
        for (int i = 0; i < num_samples; i++)
        {
            weight_gradients[j] += (linear_output_gradients[i] * x[i][j]);
        }
        weight_gradients[j] /= num_samples;
        weights[j] -= learning_rate * weight_gradients[j];
    }

    if (fit_intercept)
    {
        double bias_gradient = 0.0;
        bias_gradient = std::accumulate(linear_output_gradients.begin(), linear_output_gradients.end(), 0.0) / num_samples;
        bias -= learning_rate * bias_gradient;
    }
}

void LogisticRegression::printCost(int i, const std::vector<double> &y, const std::vector<double> &y_hat)
{
    if (print_cost && i % cost_print_interval == 0)
        std::cout << "Cost after " << i << "th iteration: " << binaryCrossEntropy(y, y_hat) << std::endl;
}