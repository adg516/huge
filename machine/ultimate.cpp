#include <iostream>
#include "LinearRegression.h"
#include "GradientDescent.h"

int main() {
    std::vector<double> xData = {1, 2, 3, 4};
    std::vector<double> yData = {2, 3, 5, 7};

    LinearRegression model;
    GradientDescent gd(0.0001);

    gd.fit(model, xData, yData, 1000);

    std::cout << "Trained Model Parameters:" << std::endl;
    std::cout << "Weight: " << model.getWeight() << std::endl;
    std::cout << "Bias: " << model.getBias() << std::endl;

    return 0;
}