#include <ctime>
#include "LogisticRegression.hpp"
#include "Metrics.hpp"

int main()
{
    srand(time(0));

    std::vector<std::vector<double>> x_train = {
        {2.0, 3.0},
        {1.0, 1.0},
        {2.5, 2.5},
        {14.5, 30.0},
        {20.0, 30.0},
        {40.0, 50.0},
        {30.5, 40.5},
        {50.0, 40.0}};
    std::vector<double> y_train = {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};

    std::vector<std::vector<double>> x_test = {
        {3.0, 6.0},
        {40.0, 65.25},
        {32.12, 11},
        {10.0, 20.0},
        {4.0, 20.0},
        {20.25, 32.0}};

    std::vector<double> y_test = {0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    // learning_rate, num_of_iterations, fit_intercept, l2, lambda, print_cost, cost_print_interval
    LogisticRegression log_reg(0.04, 800, true, true, true, 0.3, 100);

    log_reg.train(x_train, y_train);

    std::vector<double> y_hat = log_reg.predict(x_test);

    std::cout << std::endl
              << "Actual Outputs: [ ";
    for (double output : y_test)
    {
        std::cout << output << " ";
    }
    std::cout << "]";

    std::cout << std::endl
              << "Predictions:    [ ";
    for (double prediction : y_hat)
    {
        std::cout << prediction << " ";
    }
    std::cout << "]" << std::endl;

    Metrics metrics;
    metrics.computeConfusionMatrix(y_test, y_hat);
    std::cout << "Accuracy: " << metrics.accuracy() << std::endl;
    std::cout << "F1 Score: " << metrics.f1Score() << std::endl
              << std::endl;

    double bias = log_reg.getBias();
    std::cout << "The bias is: " << bias << std::endl;
    std::vector<double> weights = log_reg.getCoefficients();
    std::cout << "The weights are: [ ";
    for (double w : weights)
    {
        std::cout << w << " ";
    }
    std::cout << "]";

    return 0;
}