#include "logistic_regression.hpp"

using namespace std;

LogisticRegression::LogisticRegression(DataHandler *data_handler_)
    : LogisticRegression(data_handler_, 0.01, 100, true, false, 0.1, false, 20) {}

LogisticRegression::LogisticRegression(DataHandler *data_handler_, double learning_rate_, int num_iterations_, bool fit_intercept_, bool l2_, double lambda_, bool print_cost_, int cost_print_interval_)
    : data_handler(data_handler_), metrics(Metrics()), learning_rate(learning_rate_), num_iterations(num_iterations_), fit_intercept(fit_intercept_), l2(l2_), lambda(lambda_), print_cost(print_cost_), cost_print_interval(cost_print_interval_) {}

void LogisticRegression::train()
{
    Eigen::MatrixXd x = data_handler->getXTrain();
    Eigen::MatrixXd y = data_handler->getYTrain();

    if (x.size() == 0 || y.rows() == 0)
        throw runtime_error("Error: Input data is empty");
    if (x.rows() != y.rows())
        throw runtime_error("Error: Input size mismatch between x (features) and y (labels)");

    Eigen::MatrixXd scaled_x = standardScaler(x, true);

    initializeParameters(x.cols());

    for (int i = 0; i <= num_iterations; i++)
    {
        gradientDescent(scaled_x, y);
        printCost(i, y, forwardProp(scaled_x));
    }
}

void LogisticRegression::predict() const
{
    Eigen::MatrixXd x_test = data_handler->getXTest();

    Eigen::MatrixXd a = forwardProp(standardScaler(x_test, false));
    a = a.unaryExpr([](double p)
                    { return p >= 0.5 ? 1.0 : 0.0; }).matrix();  // force matrix return

    data_handler->setPreds(a);
}

double LogisticRegression::accuracy()
{
    metrics.computeConfusionMatrix(data_handler->getYTest(), data_handler->getPreds());
    return metrics.accuracy();
}

double LogisticRegression::precision()
{
    metrics.computeConfusionMatrix(data_handler->getYTest(), data_handler->getPreds());
    return metrics.precision();
}

double LogisticRegression::recall()
{
    metrics.computeConfusionMatrix(data_handler->getYTest(), data_handler->getPreds());
    return metrics.recall();
}

double LogisticRegression::f1Score()
{
    metrics.computeConfusionMatrix(data_handler->getYTest(), data_handler->getPreds());
    return metrics.f1Score();
}

Eigen::MatrixXd LogisticRegression::getCoefficients() const { return weights; }

double LogisticRegression::getBias() const
{
    if (!fit_intercept)
        throw runtime_error("Error: Intercept was not fitted");
    return bias;
}

Eigen::MatrixXd LogisticRegression::getPredictions() { return data_handler->getPreds(); }

void LogisticRegression::initializeParameters(const int &num_features)
{
    weights = Eigen::MatrixXd::Random(num_features, 1) * 0.01;
    bias = 0.0;
}

Eigen::MatrixXd LogisticRegression::standardScaler(const Eigen::MatrixXd &x, bool is_train) const {
    if (is_train) {
        mean_train = x.colwise().mean();

        Eigen::MatrixXd centered = x.rowwise() - mean_train;
        Eigen::RowVectorXd var = (centered.array().square().colwise().sum()) / x.rows();
        stddev_train = (var.array() + EPSILON).sqrt();

        return centered.array().rowwise() / stddev_train.array();
    } else {
        Eigen::MatrixXd centered = x.rowwise() - mean_train;
        return centered.array().rowwise() / stddev_train.array();
    }
}


Eigen::MatrixXd LogisticRegression::forwardProp(const Eigen::MatrixXd &x) const
{
    return ((x * weights).array() + bias).matrix();
}

Eigen::MatrixXd LogisticRegression::sigmoid(const Eigen::MatrixXd &z) const
{
    return (1.0 / (1.0 + (-z.array().exp()))).matrix();
}

double LogisticRegression::binaryCrossEntropy(const Eigen::MatrixXd &y, const Eigen::MatrixXd &a) const
{
    Eigen::MatrixXd a_clipped = a.unaryExpr([](double p)
                                            { return max(EPSILON, min(1.0 - EPSILON, p)); }).matrix();
    double loss = -(y.array() * a_clipped.array().log() + (1 - y.array()) * (1 - a_clipped.array()).log()).mean();
    if (l2)
        loss += (lambda / (2.0 * y.size())) * weights.squaredNorm();
    return loss;
}

void LogisticRegression::gradientDescent(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y)
{
    int num_samples = x.rows();
    Eigen::MatrixXd a = forwardProp(x);
    Eigen::MatrixXd dz = (a.array() - y.array()).matrix();  // element-wise subtraction forced
    Eigen::MatrixXd dw = (x.transpose() * dz) / num_samples;

    if (l2)
        dw += (lambda / num_samples) * weights;

    weights -= learning_rate * dw;

    if (fit_intercept)
    {
        double db = dz.mean();
        bias -= learning_rate * db;
    }
}

void LogisticRegression::printCost(int i, const Eigen::MatrixXd &y, const Eigen::MatrixXd &a) const
{
    if (print_cost && i % cost_print_interval == 0)
        cout << "Cost after " << i << "th iteration: " << binaryCrossEntropy(y, a) << endl;
}