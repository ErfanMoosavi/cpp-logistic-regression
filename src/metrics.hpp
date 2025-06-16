#ifndef METRICS_HPP
#define METRICS_HPP

#include <Eigen/Dense>
#include <stdexcept>

class Metrics
{
private:
    double TP, TN, FP, FN;

    void resetCounts();
    bool isConfusionMatrixComputed() const;

public:
    Metrics();

    void computeConfusionMatrix(const Eigen::MatrixXd &y, const Eigen::MatrixXd &y_hat);

    double accuracy() const;
    double precision() const;
    double recall() const;
    double f1Score() const;
};

#endif