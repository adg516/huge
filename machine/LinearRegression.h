#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <vector>

class LinearRegression {
private:
    double weight;
    double bias;

public:
    LinearRegression() : weight(0), bias(0) {}
    double predict(double xValue) const;
    void setParameters(double weight, double bias);
    double getWeight() const;
    double getBias() const;
    double computeCost(const std::vector<double>& x, const std::vector<double>& y) const;

};

#endif // LINEARREGRESSION_H

