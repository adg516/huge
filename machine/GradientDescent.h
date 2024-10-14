#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include "LinearRegression.h"
#include <vector>

class GradientDescent {
private:
    float learningRate;
    std::vector<double> costs; // To store the cost at each iteration

public:
    GradientDescent(float lr) : learningRate(lr) {}

    // Fits the LinearRegression model using gradient descent
    void fit(LinearRegression& model, const std::vector<std::vector<double>>& X, const std::vector<double>& y, int iterations);

    // Returns the history of costs for plotting
    std::vector<double> getCosts() const;
};

#endif // GRADIENTDESCENT_H
