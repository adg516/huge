#include <iostream>
#include "LinearRegression.h"
#include "GradientDescent.h"

int main() {
    std::vector<std::vector<double>> XData = {{1, 2}, {2, 3}, {3, 4}, {4, 5}};
    std::vector<double> yData = {2, 3, 5, 7};

    LinearRegression model;
    GradientDescent gd(0.01);

    gd.fit(model, XData, yData, 1000);

    std::cout << "Trained Model Parameters:" << std::endl;
    auto weights = model.getWeights();
    
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << "Weight[" << i << "]: " << weights[i] << std::endl;
    }
    
    std::cout << "Bias: " << model.getBias() << std::endl;

    return 0;
}