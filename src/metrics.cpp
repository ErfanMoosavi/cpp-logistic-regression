#include "Metrics.hpp"

using namespace std;

Metrics::Metrics() : TP(0.0), TN(0.0), FP(0.0), FN(0.0) {}

void Metrics::computeConfusionMatrix(const Eigen::MatrixXd &y, const Eigen::MatrixXd &y_hat)
{
    if (y.rows() != y_hat.rows() || y.cols() != 1 || y_hat.cols() != 1)
        throw runtime_error("Error: Labels must be column vectors of the same size");

    resetCounts();

    int num_samples = y.rows();
    for (int i = 0; i < num_samples; i++)
    {
        double yi = y(i, 0);
        double yhi = y_hat(i, 0);

        if (yi == 1.0 && yhi == 1.0)
            TP++;
        else if (yi == 0.0 && yhi == 0.0)
            TN++;
        else if (yi == 0.0 && yhi == 1.0)
            FP++;
        else if (yi == 1.0 && yhi == 0.0)
            FN++;
    }
}

double Metrics::accuracy() const
{
    isConfusionMatrixComputed();
    return (TP + TN) / (TP + TN + FP + FN);
}

double Metrics::precision() const
{
    isConfusionMatrixComputed();
    if (TP + FP == 0.0)
        return -1.0;

    return TP / (TP + FP);
}

double Metrics::recall() const
{
    isConfusionMatrixComputed();
    if (TP + FN == 0.0)
        return 0.0;

    return TP / (TP + FN);
}

double Metrics::f1Score() const
{
    isConfusionMatrixComputed();
    double prec = precision();
    double rec = recall();
    if (prec + rec == 0.0)
        return -1.0;

    return 2.0 * ((prec * rec) / (prec + rec));
}

void Metrics::resetCounts()
{
    TP = TN = FP = FN = 0.0;
}

bool Metrics::isConfusionMatrixComputed() const
{
    if (TP == 0.0 && TN == 0.0 && FP == 0.0 && FN == 0.0)
        throw runtime_error("Error: computeConfusionMatrix hasn't been called");
    return true;
}