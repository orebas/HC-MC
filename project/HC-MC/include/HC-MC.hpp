#pragma once

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wundef"
#pragma GCC diagnostic ignored "-Wuseless-cast"
#pragma GCC diagnostic ignored "-Wunused-parameter"

// Add more GCC-specific warnings you want to ignore here
#endif

#include <Eigen/Dense>
#include <cppad/cppad.hpp>
#include <cppad/example/cppad_eigen.hpp>  // Add this line

#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <string>

#include "types.hpp"

// Configuration and result structures
struct SolveOptions {
  bool debug = false;
  double tolerance = 1e-10;
  int maxIterations = 100;
  int maxSteps = 1000;
  double initialStepSize = 0.001;
  bool verifyDerivatives = false;
};

struct SolveResult {
  ADVector solution;
  double residualNorm;
  int iterations;
  bool success;
  std::string message;
};

inline Eigen::VectorXd toDoubleVector(const ADVector& ad_vec) {
  Eigen::VectorXd vec(ad_vec.size());
  for (auto i = 0; i < ad_vec.size(); ++i) {
    vec(i) = CppAD::Value(ad_vec(i));
  }
  return vec;
}

namespace detail {
// Helper function to convert any type to double
template <typename T>
double toDouble(const T& x) {
  if constexpr (std::is_same_v<T, ADDouble>) {
    return CppAD::Value(x);
  } else {
    return x;
  }
}

// Random vector generator in [-1,1]
template <typename T>
Vector<T> randomVector(int size) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<double> dis(-1.0, 1.0);

  Vector<T> result(size);
  for (auto i = 0; i < size; ++i) {
    result(i) = T(dis(gen));
  }
  return result;
}

// Helper to evaluate vector function and get sizes
template <typename Func>
std::pair<int, int> getFunctionDimensions(const Func& F,
                                          const ADVector& x_test) {
  // Use the provided test vector to evaluate F
  auto y = F(x_test);
  return {x_test.size(), y.size()};
}

// Compute finite difference Jacobian for verification
template <typename F>
Matrix<double> finiteDiffJacobian(const F& func, const ADVector& x,
                                  double eps = 1e-8) {
  auto n = x.size();
  ADVector x_plus = x;
  auto f0 = func(x);
  auto m = f0.size();
  Matrix<double> jac(m, n);

  for (int j = 0; j < n; ++j) {
    x_plus = x;
    x_plus(j) += eps;
    auto f1 = func(x_plus);

    for (int i = 0; i < m; ++i) {
      jac(i, j) = (toDouble(f1(i)) - toDouble(f0(i))) / eps;
    }
  }
  return jac;
}

// Compare Jacobians with detailed output
inline bool compareJacobians(const Matrix<double>& J1, const Matrix<double>& J2,
                             const std::string& context, double tol = 1e-6) {
  double max_diff = 0.0;
  int max_i = 0;
  int max_j = 0;

  for (int i = 0; i < J1.rows(); ++i) {
    for (int j = 0; j < J1.cols(); ++j) {
      const double diff = std::abs(J1(i, j) - J2(i, j));
      if (diff > max_diff) {
        max_diff = diff;
        max_i = i;
        max_j = j;
      }
      if (diff > tol) {
        std::cout << context << ": Large difference at (" << i << "," << j
                  << "):\n"
                  << "AD: " << J1(i, j) << "\n"
                  << "FD: " << J2(i, j) << "\n"
                  << "Diff: " << diff << "\n";
        return false;
      }
    }
  }

  if (max_diff > 0) {
    std::cout << context << ": Maximum difference: " << max_diff << " at ("
              << max_i << "," << max_j << ")\n";
  }
  return true;
}
}  // namespace detail

template <typename FuncF, typename FuncG>
class HFunction {
 public:
  HFunction(const FuncF& F_in, const FuncG& G_in) : F(F_in), G(G_in) {}

  template <typename T>
  Vector<T> operator()(const Vector<T>& X, const T& t) const {
    auto f_val = F(X);
    auto g_val = G(X);
    return (T(1.0) - t) * g_val + t * f_val;
  }

  template <typename T>
  [[nodiscard]] Vector<T> dH_dt(const Vector<T>& X) const {
    return F(X) - G(X);
  }

 private:
  const FuncF& F;
  const FuncG& G;
};

template <typename FuncF, typename FuncG>
class PathTracker {
 public:
  PathTracker(const FuncF& f, const FuncG& g, const SolveOptions& options_pack)
      : F(f), G(g), H_func(F, G), options(options_pack) {}

  std::optional<SolveResult> track(const ADVector& X_in,
                                   const ADDouble& t_start,
                                   const ADDouble& t_end,
                                   const ADDouble& dt_init) {
    ADVector X = X_in;
    ADDouble t = t_start;
    ADDouble dt = dt_init;
    const double min_dt = detail::toDouble(dt_init) * 1e-8;  // TODO (orebas)
                                                             // magic number
    int total_iterations = 0;

    while (detail::toDouble(t) < detail::toDouble(t_end) &&
           total_iterations < options.maxSteps) {
      // Adjust step size if we're close to the end
      ADDouble dt_current = dt;
      if (detail::toDouble(t + dt) > detail::toDouble(t_end)) {
        dt_current = t_end - t;
      }

      const ADDouble t_new = t + dt_current;
      // Predictor step: Compute dX/dt
      const ADVector H_t = H_func.dH_dt(X);
      const Eigen::MatrixXd JH = computeJacobian(X, t);

      // Convert H_t to Eigen::VectorXd
      const Eigen::VectorXd H_t_double = toDoubleVector(H_t);

      // Solve for dX_dt
      const Eigen::VectorXd dX_dt = -JH.fullPivLu().solve(H_t_double);

      // Predictor: X_pred = X + dt_current * dX_dt
      const ADVector X_pred = X + (dt_current * dX_dt).cast<ADDouble>();

      // Corrector step: Refine X_pred at t_new
      auto X_corrected_opt = correctPrediction(X_pred, t_new);
      if (!X_corrected_opt.has_value()) {
        // Reduce step size and retry
        dt *= 0.5;
        if (detail::toDouble(dt) < min_dt) {
          if (options.debug) {
            std::cerr << "Step size too small. Terminating.\n";
          }
          break;
        }
        continue;
      }

      // Update X and t
      X = X_corrected_opt.value();
      t = t_new;
      total_iterations++;

      // Increase step size for next iteration
      dt = std::min(dt * 1.2, ADDouble(options.initialStepSize));

      if (options.debug) {
        std::cout << "Step " << total_iterations << ": t = " << t
                  << ", Residual norm = "
                  << detail::toDouble(H_func(X, t).norm()) << "\n";
      }
    }

    // Check if we reached the end
    if (detail::toDouble(t) < detail::toDouble(t_end)) {
      return std::nullopt;
    }

    SolveResult result;
    result.solution = X;
    result.residualNorm = detail::toDouble(F(X).norm());
    result.iterations = total_iterations;
    result.success = true;
    result.message = "Solution found";

    return result;
  }

 private:
  const FuncF& F;
  const FuncG& G;
  HFunction<FuncF, FuncG> H_func;
  SolveOptions options;

  Eigen::MatrixXd computeJacobian(const ADVector& X, const ADDouble& t) {
    using namespace CppAD;
    const long n = X.size();

    ADVector X_ad = X;  // Eigen::Matrix<ADDouble>

    // Declare independent variables
    Independent(X_ad);

    const ADVector H_X = H_func(X_ad, t);

    // Create the function object
    ADFun<double> func(X_ad, H_X);

    // Evaluate Jacobian at the current point
    std::vector<double> x_vec(n);
    for (long i = 0; i < n; ++i) {
      x_vec[i] = Value(X[i]);  // Use CppAD::Value to extract double
    }

    std::vector<double> jac_vec = func.Jacobian(x_vec);

    // Fill the EigenMatrix with the Jacobian values
    Eigen::MatrixXd jac(H_X.size(), n);
    for (long i = 0; i < H_X.size(); ++i) {
      for (long j = 0; j < n; ++j) {
        jac(i, j) = jac_vec[i * n + j];
      }
    }

    // Optional: Verify derivatives
    if (options.verifyDerivatives) {
      // Create a lambda that captures the current t value
      auto H_at_t = [this, t](const ADVector& X_) {
        return this->H_func(X_, t);
      };

      auto fd_jac = detail::finiteDiffJacobian(H_at_t, X);

      // Convert AD Jacobian to double for comparison
      const Eigen::MatrixXd jac_double = jac;

      std::stringstream context;
      context << "Jacobian verification at t=" << Value(t);
      detail::compareJacobians(jac_double, fd_jac, context.str());
    }

    return jac;
  }
  std::optional<ADVector> correctPrediction(const ADVector& X_pred,
                                            const ADDouble& t_new) {
    ADVector X = X_pred;
    for (int iter = 0; iter < options.maxIterations; ++iter) {
      const ADVector H_X = H_func(X, t_new);
      const double residual = detail::toDouble(H_X.norm());

      if (residual < options.tolerance) {
        return X;
      }

      // Compute Jacobian
      const Eigen::MatrixXd JH = computeJacobian(X, t_new);

      // Convert H_X to Eigen::VectorXd
      const Eigen::VectorXd H_X_double = toDoubleVector(H_X);

      // Solve for update
      const Eigen::VectorXd delta_X = -JH.fullPivLu().solve(H_X_double);

      // Update X
      X = X + delta_X.cast<ADDouble>();

      if (options.debug) {
        std::cout << "Corrector Iter " << iter << ": Residual = " << residual
                  << "\n";
      }

      if (delta_X.norm() < options.tolerance) {
        return X;
      }
    }

    // Did not converge
    return std::nullopt;
  }
};
// Main solve function
template <typename Func>
std::optional<SolveResult> solve(const Func& F,
                                 int input_dim,  // New parameter
                                 const SolveOptions& options = SolveOptions{}) {
  // Generate random starting point with correct dimension
  const ADVector X0 = detail::randomVector<ADDouble>(input_dim);

  // Get output dimension
  auto [n_in, n_out] = detail::getFunctionDimensions(F, X0);

  // Verify dimensions match
  if (n_in != input_dim) {
    throw std::invalid_argument("Function input dimension mismatch");
  }

  // Create G(x) = F(x) - F(X0)
  auto F_X0 = F(X0);
  auto G = [&F, &F_X0](const ADVector& x) -> ADVector { return F(x) - F_X0; };

  // Create the path tracker
  PathTracker<Func, decltype(G)> tracker(F, G, options);

  // Track from t=0 (G) to t=1 (F)
  return tracker.track(X0, ADDouble(0.0), ADDouble(1.0),
                       ADDouble(options.initialStepSize));
}

// Overload without explicit dimension - try to deduce from a test evaluation
/*template <typename Func>
std::optional<SolveResult> solve(const Func& F,
                                 const SolveOptions& options = SolveOptions{}) {
  // Try with dimension 1 first
  try {
    const ADVector x_test = detail::randomVector<ADDouble>(1);
    auto y = F(x_test);
    return solve(F, 1, options);
  } catch (...) {
    // Try with dimension 2 if that fails
    try {
      const ADVector x_test = detail::randomVector<ADDouble>(2);
      auto y = F(x_test);
      return solve(F, 2, options);
    } catch (...) {
      // If both fail, require explicit dimension
      throw std::invalid_argument(
          "Could not automatically determine function dimension. "
          "Please provide input dimension explicitly using solve(F, dimension, "
          "options).");
    }
  }
}*/

template <typename Func>
std::optional<SolveResult> solve_newton(
    const Func& F, int input_dim,
    const SolveOptions& options = SolveOptions{}) {
  // Generate random starting point with correct dimension
  const ADVector X0 = detail::randomVector<ADDouble>(input_dim);

  // Get output dimension
  auto [n_in, n_out] = detail::getFunctionDimensions(F, X0);

  // Verify dimensions match
  if (n_in != input_dim) {
    throw std::invalid_argument("Function input dimension mismatch");
  }

  ADVector X = X0;
  int iterations = 0;

  while (iterations < options.maxIterations) {
    // Evaluate function at current point
    const ADVector F_X = F(X);
    const double residual_norm = detail::toDouble(F_X.norm());

    if (residual_norm < options.tolerance) {
      SolveResult result;
      result.solution = X;
      result.residualNorm = residual_norm;
      result.iterations = iterations;
      result.success = true;
      result.message = "Converged successfully";
      return result;
    }

    // Compute Jacobian using CppAD
    using namespace CppAD;
    const long n = X.size();

    ADVector X_ad = X;
    Independent(X_ad);
    const ADVector F_X_ad = F(X_ad);
    ADFun<double> func(X_ad, F_X_ad);

    // Convert current point to std::vector<double>
    std::vector<double> x_vec(n);
    for (long i = 0; i < n; ++i) {
      x_vec[i] = Value(X[i]);
    }

    // Evaluate Jacobian
    std::vector<double> jac_vec = func.Jacobian(x_vec);

    // Convert Jacobian to Eigen matrix
    Eigen::MatrixXd J(F_X.size(), n);
    for (long i = 0; i < F_X.size(); ++i) {
      for (long j = 0; j < n; ++j) {
        J(i, j) = jac_vec[i * n + j];
      }
    }

    // Optional derivative verification
    if (options.verifyDerivatives) {
      auto fd_jac = detail::finiteDiffJacobian(F, X);
      std::stringstream context;
      context << "Newton iteration " << iterations;
      detail::compareJacobians(J, fd_jac, context.str());
    }

    // Solve linear system for Newton step
    const Eigen::VectorXd F_X_double = toDoubleVector(F_X);
    const Eigen::VectorXd delta = -J.fullPivLu().solve(F_X_double);

    // Update X
    X = X + delta.cast<ADDouble>();

    if (options.debug) {
      std::cout << "Newton iteration " << iterations
                << ": Residual = " << residual_norm << "\n";
    }

    // Check if step size is small enough to declare convergence
    if (delta.norm() < options.tolerance) {
      SolveResult result;
      result.solution = X;
      result.residualNorm = residual_norm;
      result.iterations = iterations;
      result.success = true;
      result.message = "Converged via small step size";
      return result;
    }

    ++iterations;
  }

  // Failed to converge
  SolveResult result;
  result.solution = X;
  result.residualNorm = detail::toDouble(F(X).norm());
  result.iterations = iterations;
  result.success = false;
  result.message = "Failed to converge within maximum iterations";
  return result;
}

// Overload without explicit dimension - try to deduce from a test evaluation
template <typename Func>
std::optional<SolveResult> solve_newton(
    const Func& F, const SolveOptions& options = SolveOptions{}) {
  // Try with dimension 1 first
  try {
    const ADVector x_test = detail::randomVector<ADDouble>(1);
    auto y = F(x_test);
    return solve_newton(F, 1, options);
  } catch (...) {
    // Try with dimension 2 if that fails
    try {
      const ADVector x_test = detail::randomVector<ADDouble>(2);
      auto y = F(x_test);
      return solve_newton(F, 2, options);
    } catch (...) {
      // If both fail, require explicit dimension
      throw std::invalid_argument(
          "Could not automatically determine function dimension. "
          "Please provide input dimension explicitly using solve_newton(F, "
          "dimension, options).");
    }
  }
}