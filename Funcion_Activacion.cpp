#include "CNN.h"

double CNN::Relu(double x) {
    const double max_val = 1e9;
    const double min_val = -1e9;
    x = max(min_val, min(max_val, x));

    return (x>=0) ? x : 0;
}

VectorXd CNN::SoftMax(const Eigen::VectorXd &x) {
    double max_coeff = x.maxCoeff();
    VectorXd exp_x = VectorXd::Zero(x.size());

    for(int i=0; i<x.size(); ++i){
        exp_x[i] = exp(x[i] - max_coeff);
    }

    double sum_exp_x = exp_x.sum();

    return exp_x / ( (sum_exp_x==0) ? 1 : sum_exp_x);
}