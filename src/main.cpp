#include "logistic_regression.hpp"
#include <ctime>

using namespace std;

int main()
{
    srand(time(0));

    Eigen::MatrixXd x_train(2, 11);
    x_train << 2.0, 3.0,
        10.0, 40.0,
        2.5, 2.5,
        14.5, 30.0,
        20.0, 39.8,
        40.0, 50.0,
        30.5, 40.5,
        50.0, 40.0,
        3.4, 6.8,
        7.1, 14.0,
        2.3, 8.9;

    Eigen::RowVectorXd y_train(11);
    y_train << 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;

    Eigen::MatrixXd x_test(2, 6);
    x_test << 3.0, 6.0,
        40.0, 65.25,
        32.2, 1,
        10.0, 20.0,
        4.0, 20.0,
        20.25, 32.0;

    Eigen::RowVectorXd y_test(6);
    y_test << 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

    // learning_rate, num_of_iterations, fit_intercept, l2, lambda, print_cost, cost_print_interval
    LogisticRegression log_reg(0.01, 800, true, false, 0.3, true, 100);

    log_reg.train(x_train, y_train);

    Eigen::RowVectorXd y_hat = log_reg.predict(x_test);

    cout << "Preditions:  " << y_hat << endl;
    cout << "True Values: " << y_test << endl;
    Metrics metrics;
    metrics.computeConfusionMatrix(y_test, y_hat);
    cout << "Accuracy: " << metrics.accuracy() << endl;
    cout << "F1 Score: " << metrics.f1Score() << endl
              << endl;

    return 0;
}