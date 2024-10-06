#include "GradientDescent.h"
#include <iostream>

void GradientDescent::fit(LinearRegression& model, const std::vector<std::vector<double>>& X, const std::vector<double>& y, int iterations) {
    int n = y.size();
    int features = X[0].size();
    
    std::vector<double> w(features, 0.0);
    double b = model.getBias();

    for (int i = 0; i < iterations; ++i) {
        std::vector<double> dj_dw(features, 0.0);
        double dj_db = 0.0;

        for (int j = 0; j < n; ++j) {
            double prediction = model.predict(X[j]);
            double error = prediction - y[j];
            for (int k = 0; k < features; ++k) {
                dj_dw[k] += error * X[j][k];
            }
            dj_db += error;
        }

        for (int k = 0; k < features; ++k) {
            dj_dw[k] /= n;
        }
        dj_db /= n;

        for (int k = 0; k < features; ++k) {
            w[k] -= learningRate * dj_dw[k];
        }
        b -= learningRate * dj_db;

        if (i % 100 == 0) {
            double cost = model.computeCost(X, y);
            std::cout << "Iteration " << i << ": Cost = " << cost << std::endl;
        }
    }

    model.setParameters(w, b);
}