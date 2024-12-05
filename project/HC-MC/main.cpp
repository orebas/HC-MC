#include <boost/numeric/odeint.hpp>
#include <functional>
#include <iostream>
#include <vector>

// #include "HC-MC.hpp"
#include "ODESystem.hpp"
#include "hbtaylor.hpp"
#include "types.hpp"

using namespace boost::numeric::odeint;

#include <cppad/cppad.hpp>
#include <vector>

/*// Function to compute time derivatives of an ODE system at a given point
template <typename F, typename Scalar, typename ScalarVec>
std::vector<ScalarVec> compute_time_derivatives(
    F&& ode_function,        // ODE function representing dy/dt = F(y)
    const ScalarVec& X,      // State variables at which to compute derivatives
    size_t num_derivatives)  // Number of derivatives to compute
{
  size_t n = X.size();

  // Convert Vector<Scalar> to std::vector<Scalar> for TaylorSeries
  std::vector<Scalar> x_vec(n);
  for (size_t i = 0; i < n; ++i) {
    x_vec[i] = X[i];
  }

  // Compute Taylor series coefficients using hbtaylor
  std::vector<TaylorSeries<Scalar>> taylor_coeffs =
      computeODECoefficients(x_vec, ode_function, num_derivatives);

  // Convert TaylorSeries coefficients to time derivatives
  std::vector<ScalarVec> derivatives(num_derivatives, ScalarVec(n));
  for (size_t k = 0; k < num_derivatives; ++k) {
    for (size_t i = 0; i < n; ++i) {
      // The k-th derivative is k! times the k-th coefficient
      Scalar factorial = 1;
      for (Scalar j = 1; j <= k + 1; ++j) {
        factorial *= j;
      }
      derivatives[k][i] = taylor_coeffs[i][k + 1] * factorial;
    }
  }

  return derivatives;
}*/
template <typename T>
class ode_system {
 public:
  void operator()(const std::vector<T>& y, std::vector<T>& dydt) const {
    size_t n = y.size();
    dydt.resize(n);
    dydt[0] = T(1);  // dy_0/dt = 1
    for (size_t k = 1; k < n; ++k) {
      dydt[k] = y[k - 1];  // dy_k/dt = y_{k-1}
    }
  }
};

int main() {
  using Scalar = double;  // Ensure Scalar is double
                          // using VectorType = Vector<Scalar>;

  using VectorType = std::vector<Scalar>;
  VectorType X(5, 1.0);

  // EIGEN INITIALIATION Initial state variables
  // VectorType X(5);
  // X << 0.0, 0.0, 0.0, 0.0, 0.0;

  // Number of derivatives to compute
  size_t num_derivatives = 5;

  ode_system<TaylorSeries<Scalar>> ode_function_ts;

  // Compute Taylor series coefficients
  auto taylor_coeffs =
      computeODECoefficients<ode_system<TaylorSeries<Scalar>>, Scalar>(
          X, ode_function_ts, num_derivatives);

  // Convert coefficients to derivatives
  std::vector<std::vector<Scalar>> derivatives(num_derivatives,
                                               std::vector<Scalar>(X.size()));
  for (size_t k = 0; k < num_derivatives; ++k) {
    for (size_t i = 0; i < X.size(); ++i) {
      // k-th derivative is k! times the k-th coefficient
      Scalar factorial = 1;
      for (Scalar j = 1; j <= k + 1; ++j) {
        factorial *= j;
      }
      derivatives[k][i] = taylor_coeffs[i][k + 1] * factorial;
    }
  }

  // Output the derivatives
  std::cout << "Time derivatives at X:\n";
  for (size_t k = 0; k < derivatives.size(); ++k) {
    std::cout << "Order " << k + 1 << " derivatives:\n";
    for (size_t i = 0; i < derivatives[k].size(); ++i) {
      std::cout << "  Variable " << i << ": " << derivatives[k][i] << "\n";
    }
  }

  return 0;
}
