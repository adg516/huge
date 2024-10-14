#include "ZScoreNormalization.h"
#include <cmath>

std::vector<std::vector<double>> ZScoreNormalization::normalize(const std::vector<std::vector<double>>& X) {
    int n = X.size();
    int features = X[0].size();
    std::vector<std::vector<double>> X_normalized = X;
    std::vector<double> mean(features, 0.0);
    std::vector<double> std_dev(features, 0.0);

    // Calculate mean for each feature
    for (int j = 0; j < features; ++j) {
        for (int i = 0; i < n; ++i) {
            mean[j] += X[i][j];
        }
        mean[j] /= n;
    }

    // Calculate standard deviation for each feature
    for (int j = 0; j < features; ++j) {
        for (int i = 0; i < n; ++i) {
            std_dev[j] += std::pow(X[i][j] - mean[j], 2);
        }
        std_dev[j] = std::sqrt(std_dev[j] / n);
    }

    // Normalize the dataset
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < features; ++j) {
            if (std_dev[j] != 0)
                X_normalized[i][j] = (X[i][j] - mean[j]) / std_dev[j];
        }
    }

    return X_normalized;
}
