#include "GradientDescent.h"
#include <iostream>

// lr = alpha = learning rate
/* repeat until convergence:
    w = w - alpha(dj/dw)
    b = b - alpha(dj/db)

    ∂𝐽(𝑤,𝑏)∂𝑤 = (∑𝑖=0𝑚−1(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))𝑥(𝑖))/m
    ∂𝐽(𝑤,𝑏)∂𝑏 = (∑𝑖=0𝑚−1(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖)))/m

*/
void GradientDescent::fit(LinearRegression& model, const std::vector<double>& x, const std::vector<double>& y, int iterations) {
    int n = x.size();
    double w = model.getWeight();
    double b = model.getBias();

    for(int i=0; i<iterations; i++){
        double djdw = 0.0;
        double djdb = 0.0;

        for (int j=0; j<n; j++) {
            double prediction = w * x[j] + b;
            djdw += (prediction - y[j]) * x[j];
            djdb += (prediction - y[j]);
        }

        djdw /= n;
        djdb /= n;

        w -= learningRate * djdw;
        b -= learningRate * djdb;

        if (i % 100 == 0) {
            double cost = model.computeCost(x, y);
            std::cout << "Iteration " << i << ": Cost = " << cost << std::endl;
        }
    }

    model.setParameters(w,b);
}