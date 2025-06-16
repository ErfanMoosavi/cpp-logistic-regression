#include "logistic_regression.hpp"

using namespace std;

int main(int, char **argv)
{
    string x_train_path = argv[1];
    string y_train_path = argv[2];
    string x_test_path = argv[3];
    string y_test_path = argv[4];

    DataHandler data_handler;
    data_handler.loadCSV(x_train_path, X_TRAIN);
    data_handler.loadCSV(y_train_path, Y_TRAIN);
    data_handler.loadCSV(x_test_path, X_TEST);
    data_handler.loadCSV(y_test_path, Y_TEST);

    // learning_rate, num_of_iterations, fit_intercept, l2, lambda, print_cost, cost_print_interval
    LogisticRegression log_reg(&data_handler, 0.01, 800, true, true, 0.3, true, 80);
    
    log_reg.train();
    log_reg.predict();
    Eigen::MatrixXd predictions = log_reg.getPredictions();

    cout << "Predictions:\n" << predictions.transpose() << "\n";
    cout << "Accuracy: " << log_reg.accuracy() << "\n";
    cout << "F1 Score: " << log_reg.f1Score() << "\n";
    return 0;
}