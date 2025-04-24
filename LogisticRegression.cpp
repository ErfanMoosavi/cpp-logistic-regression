#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <numeric>
#include <vector>

using namespace std;

const double WEIGHT_INITIALIZATION_RATE = 0.01;

class LogisticRegression
{
public:
    LogisticRegression(double lr_ = 0.01, int num_of_iterations_ = 100)
    {
        lr = lr_;
        num_of_iterations = num_of_iterations_;
    }

    void initializeParameters(int n_x)
    {
        parameters.resize(2);
        parameters[0].resize(n_x);
        parameters[1].resize(1);

        for (int i = 0; i < n_x; i++)
        {
            parameters[0][i] = (static_cast<double>(rand() % 10)) * WEIGHT_INITIALIZATION_RATE;
        }

        parameters[1][0] = 0.0;
    }

    vector<double> sigmoid(vector<double> z)
    {
        vector<double> a;

        for (int i = 0; i < z.size(); i++)
        {
            a.push_back(1 / (1 + exp(-z[i])));
        }

        return a;
    }

    vector<double> forwardProp(vector<vector<double>> x)
    {
        int m = x.size();
        int n_x = x[0].size();
        vector<double> z(m, 0.0);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n_x; j++)
            {
                z[i] += parameters[0][j] * x[i][j];
            }
            z[i] += parameters[1][0];
        }

        return sigmoid(z);
    }

    void gradientDescent(vector<vector<double>> x, vector<double> y)
    {
        int m = x.size();
        int n_x = parameters[0].size();
        vector<double> a = forwardProp(x);
        vector<double> dz(m);
        vector<double> dw(n_x, 0);
        double db = 0;

        for (int i = 0; i < m; i++)
        {
            dz[i] = a[i] - y[i];
        }

        for (int i = 0; i < n_x; i++)
        {
            for (int j = 0; j < m; j++)
            {
                dw[i] += (dz[j] * x[j][i]);
            }
            dw[i] /= m;
            parameters[0][i] -= lr * dw[i];
        }

        db = accumulate(dz.begin(), dz.end(), 0.0) / m;
        parameters[1][0] -= lr * db;
    }

    void fit(vector<vector<double>> &x, vector<double> &y)
    {
        initializeParameters(x.size());

        for (int i = 0; i < num_of_iterations; i++)
        {
            gradientDescent(x, y);
        }
    }

private:
    vector<vector<double>> parameters;
    double lr;
    int num_of_iterations;
};

int main()
{
    srand(time(0));

    vector<vector<double>> x = {
        {2.0, 3.0},
        {1.0, 1.0},
        {2.5, 2.5},
        {40.0, 50.0},
        {30.5, 40.5},
        {50.0, 40.0}};
    vector<double> y = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};

    LogisticRegression log_reg;
    log_reg.fit(x, y);
    vector<double> y_hat = log_reg.forwardProp(x);

    cout << "Predicted probabilities:" << endl;
    for (double prediction : y_hat)
    {
        cout << prediction << endl;
    }

    return 0;
}