#include "AAA.hpp"
#include "mpreal.h"
#include <complex>
#include <iostream>
#include <vector>

template<typename T>
T
Func(const T &x) {
    // return (T(1.0) / T(x - 0.1)) * std::abs(x - T(0.5)) * std::exp(std::sin(20.0 * x));
    return (exp(x)) * sqrt(x + 0.1) + abs(x - 0.1234567);
}

int
main() {
    using mpfr::mpreal;


    // Required precision of computations in decimal digits
    // Play with it to check different precisions
    const int digits = 50;

    // Setup default precision for all subsequent computations
    // MPFR accepts precision in bits - so we do the conversion
    mpreal::set_default_prec(mpfr::digits2bits(digits));

    mpreal r = 0;


    // Sample points (Z) and function values (F)


    using Scalar = mpreal;
    //    using Complex = std::complex<Scalar>;

    using Complex = Scalar;

    std::vector<Scalar> Z;
    std::vector<Scalar> F;

    // Example: Approximate the function f(z) = exp(z) on the interval [0, 1]
    size_t N = 203; // Reduced number of points for brevity
    for (size_t i = 0; i < N; ++i) {
        Scalar x = Scalar(i) / (N - 1); // Points between 0 and 1
        Scalar z = x;
        Z.push_back(z);
        F.push_back(Func(z));
    }

    // Create AAA approximant
    AAA<Scalar> approximant;
    approximant.fit(Z, F);

    // Evaluate at new points
    std::vector<Scalar> Z_eval;
    for (size_t i = 0; i < N; ++i) {
        Scalar x = Scalar(i) / (N - 1) + 0.001; // Shifted points
        Z_eval.push_back(x);
    }

    std::vector<Scalar> F_approx = approximant(Z_eval);

    // Compare with exact values
    for (size_t i = 0; i < Z_eval.size(); ++i) {
        Scalar f_exact = Func(Z_eval[i]);
        Scalar f_approx = F_approx[i];
        Scalar error = abs(f_exact - f_approx);
        std::cout << "z = " << Z_eval[i] << ", Exact = " << f_exact << ", Approx = " << f_approx
                  << ", Error = " << error << std::endl;
    }

    return 0;
}
