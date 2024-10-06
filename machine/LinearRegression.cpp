#include "LinearRegression.h"
#include <cmath>

double LinearRegression::predict(const std::vector<double>& x) const {
    double prediction = bias;
    for (size_t i = 0; i < weights.size(); ++i) {
        prediction += weights[i] * x[i];
    }
    return prediction;
}

void LinearRegression::setParameters(const std::vector<double>& newWeights, double newBias) {
    weights = newWeights;
    bias = newBias;
}

std::vector<double> LinearRegression::getWeights() const {
    return weights;
}

double LinearRegression::getBias() const {
    return bias;
}

double LinearRegression::computeCost(const std::vector<std::vector<double>>& X, const std::vector<double>& y) const {
    int n = y.size();
    double totalError = 0.0;
    for (int i = 0; i < n; ++i) {
        double prediction = predict(X[i]);
        totalError += std::pow(prediction - y[i], 2);
    }
    return totalError / (2 * n);
}