#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <vector>

class LinearRegression {
private:
    std::vector<double> weights;
    double bias;

public:
    LinearRegression() : bias(0) {}
    double predict(const std::vector<double>& x) const;
    void setParameters(const std::vector<double>& weights, double bias);
    std::vector<double> getWeights() const;
    double getBias() const;
    double computeCost(const std::vector<std::vector<double>>& X, const std::vector<double>& y) const;
};

#endif // LINEARREGRESSION_H