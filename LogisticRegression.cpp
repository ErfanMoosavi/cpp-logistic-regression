#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
using namespace std;

const double WEIGHT_INITIALIZATION_RATE = 0.01;

class Metrics
{
public:
    Metrics() : TP(-1.0), TN(-1.0), FP(-1.0), FN(-1.0) {}

    void computeConfusionMatrix(const vector<double> &y, const vector<double> &y_hat)
    {
        if (y.size() != y_hat.size())
        {
            throw runtime_error("Error: Input size mismatch");
        }
        resetCounts();

        int num_samples = y.size();
        for (int i = 0; i < num_samples; i++)
        {
            if (y[i] == 1.0 && y_hat[i] == 1.0)
                TP++;

            else if (y[i] == 0.0 && y_hat[i] == 0.0)
                TN++;

            else if (y[i] == 0.0 && y_hat[i] == 1.0)
                FP++;

            else if (y[i] == 1.0 && y_hat[i] == 0.0)
                FN++;
        }
    }

    double accuracy()
    {
        isConfusionMatrixComputed();
        return (TP + TN) / (TP + TN + FP + FN);
    }

    double precision()
    {
        isConfusionMatrixComputed();
        if (TP + FP == 0.0)
            return -1.0;

        return TP / (TP + FP);
    }

    double recall()
    {
        isConfusionMatrixComputed();
        if (TP + FN == -1.0)
            return 0.0;

        return TP / (TP + FN);
    }

    double f1Score()
    {
        isConfusionMatrixComputed();
        double prec = precision();
        double rec = recall();
        if (prec + rec == 0)
            return -1.0;

        return 2 * ((prec * rec) / (prec + rec));
    }

private:
    double TP, TN, FP, FN;

    void resetCounts()
    {
        TP = TN = FP = FN = 0.0;
    }

    bool isConfusionMatrixComputed()
    {
        if (TP == -1.0)
            throw runtime_error("Error: computeConfusionMatrix hasn't been called");
    }
};

class LogisticRegression
{
public:
    LogisticRegression()
        : learning_rate(0.01), num_of_iterations(100), fit_intercept(true), print_cost(false), cost_print_interval(20)
    {
    }

    LogisticRegression(double learning_rate_, int num_of_iterations_, bool fit_intercept_, bool print_cost_, int cost_print_interval_)
        : learning_rate(learning_rate_), num_of_iterations(num_of_iterations_), fit_intercept(fit_intercept_), print_cost(print_cost_), cost_print_interval(cost_print_interval_)
    {
    }

    void fit(const vector<vector<double>> &x, const vector<double> &y)
    {
        if (x.empty() || y.empty())
        {
            cerr << "Error: Input data (x or y) is empty." << endl;
            return;
        }

        if (x.size() != y.size())
        {
            throw runtime_error("Error: Input size mismatch");
        }

        initializeParameters(x[0].size());

        for (int i = 0; i <= num_of_iterations; i++)
        {
            gradientDescent(x, y);
            printCost(i, y, forwardProp(x));
        }
    }

    vector<double> predict(const vector<vector<double>> &x)
    {
        vector<double> y_hat = forwardProp(x);
        for_each(y_hat.begin(), y_hat.end(), [](double &p)
                 { p = p >= 0.5 ? 1 : 0; });
        return y_hat;
    }

    double binaryCrossEntropy(const vector<double> &y, const vector<double> &y_hat)
    {
        int num_samples = y.size();
        double sum = 0;
        double epsilon = 1e-9; // Small value to prevent log(0)

        for (int i = 0; i < num_samples; i++)
        {
            double y_hat_clipped = max(epsilon, min(1.0 - epsilon, y_hat[i]));
            sum += y[i] * log(y_hat_clipped) + (1 - y[i]) * log(1 - y_hat_clipped);
        }

        return -sum / num_samples;
    }

    vector<double> getCoefficients() const { return weights; }

    double getBias() const { return bias; }

private:
    vector<double> weights;
    double bias;
    double learning_rate;
    int num_of_iterations;
    bool fit_intercept;
    bool print_cost;
    int cost_print_interval;

    vector<double> sigmoid(const vector<double> &linear_output)
    {
        int num_samples = linear_output.size();
        vector<double> y_hat(num_samples);

        for (int i = 0; i < num_samples; i++)
        {
            y_hat[i] = 1.0 / (1.0 + exp(-linear_output[i]));
        }

        return y_hat;
    }

    void initializeParameters(const int &num_of_features)
    {
        weights.resize(num_of_features);

        for (int j = 0; j < num_of_features; j++)
        {
            weights[j] = (rand() % 10) * WEIGHT_INITIALIZATION_RATE;
        }
        bias = 0.0;
    }

    vector<double> forwardProp(const vector<vector<double>> &x)
    {
        int num_samples = x.size();
        int num_of_features = x[0].size();
        vector<double> linear_output(num_samples, 0.0);

        for (int i = 0; i < num_samples; i++)
        {
            for (int j = 0; j < num_of_features; j++)
            {
                linear_output[i] += weights[j] * x[i][j];
            }
            if (fit_intercept)
                linear_output[i] += bias;
        }

        return sigmoid(linear_output);
    }

    void gradientDescent(const vector<vector<double>> &x, const vector<double> &y)
    {
        int num_samples = x.size();
        int num_of_features = weights.size();
        vector<double> y_hat = forwardProp(x);
        vector<double> linear_output_gradients(num_samples);
        vector<double> weight_gradients(num_of_features);

        for (int i = 0; i < num_samples; i++)
        {
            linear_output_gradients[i] = y_hat[i] - y[i];
        }

        for (int j = 0; j < num_of_features; j++)
        {
            for (int i = 0; i < num_samples; i++)
            {
                weight_gradients[j] += (linear_output_gradients[i] * x[i][j]);
            }
            weight_gradients[j] /= num_samples;
            weights[j] -= learning_rate * weight_gradients[j];
        }

        if (fit_intercept)
        {
            double bias_gradient = 0;
            bias_gradient = accumulate(linear_output_gradients.begin(), linear_output_gradients.end(), 0.0) / num_samples;
            bias -= learning_rate * bias_gradient;
        }
    }

    void printCost(int i, const vector<double> &y, const vector<double> &y_hat)
    {
        if (i % cost_print_interval == 0 && print_cost)
            cout << "Cost after " << i << "th iteration: " << binaryCrossEntropy(y, y_hat) << endl;
    }
};

int main()
{
    srand(time(0));

    vector<vector<double>> x_train = {
        {2.0, 3.0},
        {1.0, 1.0},
        {2.5, 2.5},
        {14.5, 30.0},
        {20.0, 30.0},
        {40.0, 50.0},
        {30.5, 40.5},
        {50.0, 40.0}};
    vector<double> y_train = {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};

    vector<vector<double>> x_test = {
        {3.0, 6.0},
        {40.0, 65.25},
        {32.12, 11},
        {10.0, 20.0},
        {4.0, 20.0},
        {20.25, 32.0}};

    vector<double> y_test = {0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    // learning_rate, num_of_iterations, fit_intercept, print_cost, cost_print_interval
    LogisticRegression log_reg(0.04, 800, true, true, 100);
    log_reg.fit(x_train, y_train);
    vector<double> y_hat = log_reg.predict(x_test);

    cout << endl
         << "Actual Outputs: [ ";
    for (double output : y_test)
    {
        cout << output << " ";
    }
    cout << "]";

    cout << endl
         << "Predictions:    [ ";
    for (double prediction : y_hat)
    {
        cout << prediction << " ";
    }
    cout << "]" << endl;

    Metrics metrics;
    metrics.computeConfusionMatrix(y_test, y_hat);
    cout << "Accuracy: " << metrics.accuracy() << endl;
    cout << "F1 Score: " << metrics.f1Score() << endl
         << endl;

    double bias = log_reg.getBias();
    cout << "The bias is: " << bias << endl;
    vector<double> weights = log_reg.getCoefficients();
    cout << "The weights are: [ ";
    for (double w : weights)
    {
        cout << w << " ";
    }
    cout << "]";

    return 0;
}