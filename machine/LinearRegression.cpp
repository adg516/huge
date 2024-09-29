#include "LinearRegression.h"
#include <cmath>

double LinearRegression::predict(double xValue) const {
    return weight * xValue + bias;
}

void LinearRegression::setParameters(double newWeight, double newBias) {
    weight = newWeight;
    bias = newBias;
}

double LinearRegression::getWeight() const {
    return weight;
}

double LinearRegression::getBias() const {
    return bias;
}

// Cost function J
// J(w,b) = (fwb(xi) - y(i))^2/2n
// Difference between prediction and actual value squared, all over 2n
double LinearRegression::computeCost(const std::vector<double>& x, const std::vector<double>& y) const {
    int n = x.size();
    double totalError = 0;

    for(int i=0; i<n; i++) {
        double prediction = predict(x[i]);
        totalError += std::pow(prediction - y[i], 2);
    }

    return totalError / (2*n);
}


