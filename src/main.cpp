#include "logistic_regression.hpp"
#include "data_handler.hpp"
#include "cxxopts.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    cxxopts::Options options("LogisticRegression", "Train a logistic regression model from the command line");

    options.add_options()("x_train", "Path to X train CSV", cxxopts::value<string>())("y_train", "Path to Y train CSV", cxxopts::value<string>())("x_test", "Path to X test CSV", cxxopts::value<string>())("y_test", "Path to Y test CSV", cxxopts::value<string>())("lr", "Learning rate", cxxopts::value<double>()->default_value("0.01"))("epochs", "Number of iterations", cxxopts::value<int>()->default_value("100"))("fit_intercept", "Fit intercept", cxxopts::value<bool>()->default_value("true"))("l2", "Use L2 regularization", cxxopts::value<bool>()->default_value("false"))("lambda", "L2 regularization strength", cxxopts::value<double>()->default_value("0.1"))("print_cost", "Print cost per interval", cxxopts::value<bool>()->default_value("false"))("cost_interval", "Cost print interval", cxxopts::value<int>()->default_value("20"))("help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help") || !result.count("x_train") || !result.count("y_train") || !result.count("x_test") || !result.count("y_test"))
    {
        cout << options.help() << endl;
        return 0;
    }

    DataHandler data_handler;
    data_handler.loadCSV(result["x_train"].as<string>(), "x_train");
    data_handler.loadCSV(result["y_train"].as<string>(), "y_train");
    data_handler.loadCSV(result["x_test"].as<string>(), "x_test");
    data_handler.loadCSV(result["y_test"].as<string>(), "y_test");

    LogisticRegression model(
        &data_handler,
        result["lr"].as<double>(),
        result["epochs"].as<int>(),
        result["fit_intercept"].as<bool>(),
        result["l2"].as<bool>(),
        result["lambda"].as<double>(),
        result["print_cost"].as<bool>(),
        result["cost_interval"].as<int>());

    model.train();
    model.predict();

    cout << "Accuracy: " << model.accuracy() << endl;
    cout << "Precision: " << model.precision() << endl;
    cout << "Recall: " << model.recall() << endl;
    cout << "F1 Score: " << model.f1Score() << endl;

    return 0;
}