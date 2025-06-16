#include "logistic_regression.hpp"

using namespace std;

LogisticRegression::LogisticRegression()
    : metrics(Metrics()), learning_rate(0.01), num_iterations(100), fit_intercept(true), l2(false), lambda(0.1), print_cost(false), cost_print_interval(20)
{
}

LogisticRegression::LogisticRegression(double learning_rate_, int num_iterations_, bool fit_intercept_, bool l2_, double lambda_, bool print_cost_, int cost_print_interval_)
    : metrics(Metrics()), learning_rate(learning_rate_), num_iterations(num_iterations_), fit_intercept(fit_intercept_), l2(l2_), lambda(lambda_), print_cost(print_cost_), cost_print_interval(cost_print_interval_)
{
}

void LogisticRegression::train(const Eigen::MatrixXd &x, const Eigen::RowVectorXd &y)
{
    if (x.size() == 0 || y.size() == 0)
        throw runtime_error("Error: Input data is empty");
    if (x.cols() != y.size())
        throw runtime_error("Error: Input size mismatch");

    Eigen::MatrixXd scaled_x = standardScaler(x);

    initializeParameters(x.rows());
    for (int i = 0; i <= num_iterations; i++)
    {
        gradientDescent(scaled_x, y);
        printCost(i, y, forwardProp(scaled_x));
    }
}

Eigen::RowVectorXd LogisticRegression::predict(const Eigen::MatrixXd &x)
    const
{
    Eigen::RowVectorXd a = forwardProp(standardScaler(x));
    a = a.unaryExpr([](double p)
                    { return p >= 0.5 ? 1.0 : 0.0; });
    return a;
}

double LogisticRegression::accuracy(const Eigen::RowVectorXd &y, const Eigen::RowVectorXd &y_hat)
{
    metrics.computeConfusionMatrix(y, y_hat);
    return metrics.accuracy();
}

double LogisticRegression::precision(const Eigen::RowVectorXd &y, const Eigen::RowVectorXd &y_hat)
{
    metrics.computeConfusionMatrix(y, y_hat);
    return metrics.precision();
}

double LogisticRegression::recall(const Eigen::RowVectorXd &y, const Eigen::RowVectorXd &y_hat)
{
    metrics.computeConfusionMatrix(y, y_hat);
    return metrics.recall();
}

double LogisticRegression::f1Score(const Eigen::RowVectorXd &y, const Eigen::RowVectorXd &y_hat)
{
    metrics.computeConfusionMatrix(y, y_hat);
    return metrics.f1Score();
}

Eigen::VectorXd LogisticRegression::getCoefficients() const { return weights; }

double LogisticRegression::getBias()
    const
{
    if (!fit_intercept)
        throw runtime_error("Error: Intercept was not fitted");
    return bias;
}

void LogisticRegression::initializeParameters(const int &num_features)
{
    weights.resize(num_features);
    weights = Eigen::VectorXd::Random(num_features) * 0.01;
    bias = 0.0;
}

Eigen::VectorXd LogisticRegression::mean(const Eigen::MatrixXd &x)
    const
{
    return x.rowwise().mean();
}

Eigen::VectorXd LogisticRegression::standardDeviation(const Eigen::MatrixXd &x)
    const
{
    int num_samples = x.cols();
    if (num_samples <= 1)
        return Eigen::VectorXd::Zero(x.rows());

    Eigen::MatrixXd centered = x.array().colwise() - mean(x).array();
    Eigen::VectorXd variance = (centered.array().square().rowwise().sum() / (num_samples));
    return (variance.array() + EPSILON).sqrt();
}

Eigen::MatrixXd LogisticRegression::standardScaler(const Eigen::MatrixXd &x) const
{
    Eigen::MatrixXd centered = x.array().colwise() - mean(x).array();
    centered.array().colwise() /= standardDeviation(x).array();
    return centered;
}

Eigen::RowVectorXd LogisticRegression::forwardProp(const Eigen::MatrixXd &x)
    const
{
    return sigmoid((weights.transpose() * x).array() + bias);
}

Eigen::RowVectorXd LogisticRegression::sigmoid(const Eigen::RowVectorXd &z)
    const
{
    return 1.0 / (1.0 + (-z.array().exp()));
}

double LogisticRegression::binaryCrossEntropy(const Eigen::RowVectorXd &y, const Eigen::RowVectorXd &a) const
{
    Eigen::RowVectorXd a_clipped = a.unaryExpr([](double p)
                                               { return max(EPSILON, min(1.0 - EPSILON, p)); });
    double loss = -(y.array() * a_clipped.array().log() + (1 - y.array()) * (1 - a_clipped.array()).log()).mean();
    if (l2)
        loss += (lambda / (2.0 * y.size())) * weights.squaredNorm();
    return loss;
}

void LogisticRegression::gradientDescent(const Eigen::MatrixXd &x, const Eigen::RowVectorXd &y)
{
    int num_samples = x.cols();
    Eigen::RowVectorXd a = forwardProp(x);
    Eigen::RowVectorXd dz = (a - y);
    Eigen::VectorXd dw = (x * dz.transpose()) / num_samples;

    if (l2)
        dw += (lambda / num_samples) * weights;

    weights -= learning_rate * dw;

    if (fit_intercept)
    {
        double db = dz.mean();
        bias -= learning_rate * db;
    }
}

void LogisticRegression::printCost(int i, const Eigen::RowVectorXd &y, const Eigen::RowVectorXd &a)
    const
{
    if (print_cost && i % cost_print_interval == 0)
        cout << "Cost after " << i << "th iteration: " << binaryCrossEntropy(y, a) << endl;
}