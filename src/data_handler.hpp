#ifndef DATA_HANDLER_HPP
#define DATA_HANDLER_HPP

#include <Eigen/Dense>
#include <fstream>
#include <istream>
#include <string>
#include <vector>

using namespace std;

class DataHandler
{
public:
    void loadCSV(const string &path, const string &type);
    const Eigen::MatrixXd& getXTrain();
    const Eigen::MatrixXd& getYTrain();
    const Eigen::MatrixXd& getXTest();
    const Eigen::MatrixXd& getYTest();
    const Eigen::MatrixXd& getPreds();
    void setYTest(Eigen::MatrixXd y_test_);
    void setPreds(Eigen::MatrixXd preds_);

private:
    Eigen::MatrixXd x_train;
    Eigen::MatrixXd y_train;
    Eigen::MatrixXd x_test;
    Eigen::MatrixXd y_test;
    Eigen::MatrixXd preds;
};

#endif