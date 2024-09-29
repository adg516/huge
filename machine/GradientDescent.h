#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include "LinearRegression.h"
#include <vector>

class GradientDescent {
private:
    float learningRate;

// lr = learning rate = alpha
public:
    GradientDescent(float lr) : learningRate(lr) {}
    void fit(LinearRegression& model, const std::vector<double>& x, const std::vector<double>& y, int iterations);
};

#endif // GRADIENTDESCENT_H