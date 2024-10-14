#ifndef ZSCALENORMALIZATION_H
#define ZSCALENORMALIZATION_H

#include <vector>

class ZScoreNormalization {
public:
    static std::vector<std::vector<double>> normalize(const std::vector<std::vector<double>>& X);
};

#endif // ZSCALENORMALIZATION_H
