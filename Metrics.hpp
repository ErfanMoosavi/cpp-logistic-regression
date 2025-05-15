#ifndef METRICS_HPP
#define METRICS_HPP

#include <stdexcept>
#include <vector>

class Metrics
{
public:
    // Constructor
    Metrics();

    // Public methods
    void computeConfusionMatrix(const std::vector<double> &y, const std::vector<double> &y_hat);
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