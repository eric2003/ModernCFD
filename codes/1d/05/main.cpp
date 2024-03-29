#define _USE_MATH_DEFINES // for C++
#include <cmath>
#include "matplotlibcppModified.h"

namespace plt = matplotlibcpp;

int main(int argc, char** argv)
{
    // Prepare data.
    int n = 5000;
    std::vector<double> x(n), y(n), z(n), w(n,2);
    for(int i=0; i<n; ++i) {
        x.at(i) = i*i;
        y.at(i) = sin(2*M_PI*i/360.0);
        z.at(i) = log(i);
    }

    // Set the size of output image to 1200x780 pixels
    plt::figure_size(1200, 780);
    // Plot line from given x and y data. Color is selected automatically.
    plt::plot(x, y);
    // Add graph title
    plt::title("Sample figure");
    // Save the image (file format is determined by the extension)
    plt::savefig("./basic.png");

    return 0;
}