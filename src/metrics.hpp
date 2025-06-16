#ifndef METRICS_HPP
#define METRICS_HPP

#include <stdexcept>
#include <Eigen/Dense>

using namespace std;

class Metrics
{
public:
    // Constructor
    Metrics();

    // Public methods
    void computeConfusionMatrix(const Eigen::RowVectorXd &y, const Eigen::RowVectorXd &y_hat);
    double accuracy() const;
    double precision() const;
    double recall() const;
    double f1Score() const;

private:
    // Private methods
    void resetCounts();
    bool isConfusionMatrixComputed() const;

    // Member variables
    double TP, TN, FP, FN;
};
#endif