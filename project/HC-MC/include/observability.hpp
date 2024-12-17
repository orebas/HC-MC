// In a new header, let's say observability.hpp:
// TODO:  The below is totally untested.

#pragma once

#include "eigenwrap.hpp"
#include "hbtaylor.hpp"
#include "types.hpp"

template <typename T>
struct ObservabilityResult {
  // Taylor series for state and observed variables
  std::vector<TaylorSeries<T>> state_series;
  std::vector<TaylorSeries<T>> obs_series;

  // Jacobians of Taylor coefficients with respect to initial conditions and
  // parameters Each matrix has rows = (num_observed_vars * num_taylor_coeffs)
  // and cols = (num_state_vars + num_params)
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> observability_matrix;
};

template <typename ODESystem>
ObservabilityResult<typename ODESystem::RealType> computeObservabilityMatrix(
    const ODESystem& system, const typename ODESystem::RealType& t0,
    int degree) {
  using T = typename ODESystem::RealType;

  // Get dimensions
  const size_t num_states = system.getStateSize();
  const size_t num_params = system.getParameterSize();
  const size_t total_inputs = num_states + num_params;

  // Create vector of independent variables (initial states and parameters)
  std::vector<ADDouble> X_ad;
  X_ad.reserve(total_inputs);

  // Add initial states
  const auto& init_state = system.getInitialState();
  for (const auto& x : init_state) {
    X_ad.push_back(ADDouble(x));
  }

  // Add parameters
  const auto& params = system.getParameters();
  for (const auto& p : params) {
    X_ad.push_back(ADDouble(p));
  }

  // Start recording
  CppAD::Independent(X_ad);

  // Create AD version of system with these parameters
  auto wrapped_system = [&system, &t0](
                            const std::vector<TaylorSeries<ADDouble>>& series,
                            std::vector<TaylorSeries<ADDouble>>& result) {
    // Extract parameters from the end of X_ad
    std::vector<TaylorSeries<ADDouble>> p_series;
    p_series.reserve(series.size());
    for (size_t i = 0; i < series.size(); ++i) {
      p_series.emplace_back(ADDouble(0), series[0].getDegree());
    }

    system.state_equations(series, result, p_series, ADDouble(t0));
  };

  // Compute Taylor series using AD types
  std::vector<ADDouble> init_state_ad(X_ad.begin(), X_ad.begin() + num_states);
  auto state_series_ad =
      computeODECoefficients(init_state_ad, wrapped_system, degree);

  // Compute observable series
  std::vector<TaylorSeries<ADDouble>> obs_series_ad;
  system.observe(state_series_ad, obs_series_ad);

  // Create dependent variables vector (flattened Taylor coefficients of
  // observables)
  std::vector<ADDouble> Y;
  const size_t num_obs = obs_series_ad.size();
  Y.reserve(num_obs * (degree + 1));

  for (const auto& series : obs_series_ad) {
    for (int i = 0; i <= degree; ++i) {
      Y.push_back(series[i]);
    }
  }

  // Create ADFun object
  CppAD::ADFun<T> func(X_ad, Y);

  // Compute Jacobian
  std::vector<T> x_vec(total_inputs);
  for (size_t i = 0; i < num_states; ++i) {
    x_vec[i] = init_state[i];
  }
  for (size_t i = 0; i < num_params; ++i) {
    x_vec[num_states + i] = params[i];
  }

  std::vector<T> jac = func.Jacobian(x_vec);

  // Convert to Eigen matrix
  const size_t num_rows = Y.size();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> obs_matrix(num_rows,
                                                              total_inputs);
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < total_inputs; ++j) {
      obs_matrix(i, j) = jac[i * total_inputs + j];
    }
  }

  // Convert AD series back to regular series
  std::vector<TaylorSeries<T>> state_series;
  std::vector<TaylorSeries<T>> obs_series;

  for (const auto& s : state_series_ad) {
    TaylorSeries<T> ts(T(0), degree);
    for (int i = 0; i <= degree; ++i) {
      ts[i] = CppAD::Value(s[i]);
    }
    state_series.push_back(ts);
  }

  for (const auto& s : obs_series_ad) {
    TaylorSeries<T> ts(T(0), degree);
    for (int i = 0; i <= degree; ++i) {
      ts[i] = CppAD::Value(s[i]);
    }
    obs_series.push_back(ts);
  }

  return {state_series, obs_series, obs_matrix};
}

struct DerivativeLevels {
  std::unordered_map<size_t, long>
      derivative_levels;  // observable index -> number of derivatives needed
  std::vector<size_t>
      unidentifiable_params;  // indices of unidentifiable parameters
  std::vector<double>
      unidentifiable_values;  // fixed values for unidentifiable parameters
};

class ObservabilityAnalyzer {
 public:
  ObservabilityAnalyzer(double rtol = 1e-12, double atol = 1e-12)
      : rtol_(rtol), atol_(atol) {}

  DerivativeLevels analyze(const Eigen::MatrixXd& full_observability_matrix,
                           size_t num_observables, size_t max_derivatives,
                           const std::vector<std::string>& param_names =
                               std::vector<std::string>()) {
    DerivativeLevels result;

    // Copy the original matrix as we'll be modifying it
    Eigen::MatrixXd working_matrix = full_observability_matrix;

    // Get initial rank
    int max_rank = computeRank(working_matrix);

    // Find unidentifiable parameters
    auto [unident_params, unident_values] =
        findUnidentifiableParameters(working_matrix);
    result.unidentifiable_params = unident_params;
    result.unidentifiable_values = unident_values;

    if (!param_names.empty()) {
      std::cout << "Unidentifiable parameters:\n";
      for (size_t i = 0; i < unident_params.size(); ++i) {
        std::cout << param_names[unident_params[i]] << " = "
                  << unident_values[i] << "\n";
      }
    }

    // Remove columns corresponding to unidentifiable parameters
    removeUnidentifiableColumns(working_matrix, unident_params);

    // For each observable, find minimum derivatives needed
    result.derivative_levels = findMinimumDerivatives(
        working_matrix, num_observables, max_derivatives, max_rank);

    return result;
  }

 private:
  double rtol_;
  double atol_;
  std::mt19937 rng_{std::random_device{}()};

  int computeRank(const Eigen::MatrixXd& matrix) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix);
    Eigen::VectorXd singular_values = svd.singularValues();

    double threshold = atol_ + rtol_ * singular_values(0);
    int rank = 0;
    for (int i = 0; i < singular_values.size(); ++i) {
      if (singular_values(i) > threshold) {
        rank++;
      }
    }
    return rank;
  }

  std::pair<std::vector<size_t>, std::vector<double>>
  findUnidentifiableParameters(const Eigen::MatrixXd& matrix) {
    // Compute nullspace
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullV);
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::VectorXd singular_values = svd.singularValues();

    double threshold = atol_ + rtol_ * singular_values(0);

    // Find nullspace columns
    std::vector<size_t> unident_params;
    std::vector<double> unident_values;
    std::uniform_real_distribution<double> dist(
        0.1, 1.0);  // Random values between 0.1 and 1.0

    // Check each column of V corresponding to small singular values
    for (int i = 0; i < singular_values.size(); ++i) {
      if (singular_values(i) <= threshold) {
        // Get nullspace vector
        Eigen::VectorXd null_vector = V.col(i);

        // Find largest component
        int max_idx = 0;
        double max_val = std::abs(null_vector(0));
        for (int j = 1; j < null_vector.size(); ++j) {
          if (std::abs(null_vector(j)) > max_val) {
            max_val = std::abs(null_vector(j));
            max_idx = j;
          }
        }

        if (max_val > rtol_) {
          unident_params.push_back(max_idx);
          unident_values.push_back(dist(rng_));
        }
      }
    }

    return {unident_params, unident_values};
  }

  void removeUnidentifiableColumns(Eigen::MatrixXd& matrix,
                                   const std::vector<size_t>& unident_params) {
    if (unident_params.empty()) return;

    // Create mask of columns to keep
    std::vector<bool> keep_col(matrix.cols(), true);
    for (size_t idx : unident_params) {
      keep_col[idx] = false;
    }

    // Count remaining columns
    long new_cols = std::count(keep_col.begin(), keep_col.end(), true);

    // Create new matrix with remaining columns
    Eigen::MatrixXd new_matrix(matrix.rows(), new_cols);
    int col_idx = 0;
    for (int i = 0; i < matrix.cols(); ++i) {
      if (keep_col[i]) {
        new_matrix.col(col_idx++) = matrix.col(i);
      }
    }

    matrix = new_matrix;
  }

  std::unordered_map<size_t, long> findMinimumDerivatives(
      const Eigen::MatrixXd& matrix, size_t num_observables,
      size_t max_derivatives, int target_rank) {
    std::unordered_map<size_t, long> deriv_levels;
    for (size_t i = 0; i < num_observables; ++i) {
      deriv_levels[i] = max_derivatives;
    }

    bool keep_reducing = true;
    while (keep_reducing) {
      bool found_reduction = false;

      // Try reducing each observable's derivatives
      for (size_t obs_idx = 0; obs_idx < num_observables; ++obs_idx) {
        if (deriv_levels[obs_idx] == 0) continue;

        // Try reducing this observable's derivatives by 1
        deriv_levels[obs_idx]--;

        // Construct reduced matrix
        Eigen::MatrixXd reduced_matrix = constructReducedMatrix(
            matrix, deriv_levels, num_observables, max_derivatives);

        // Check if rank is maintained
        if (computeRank(reduced_matrix) < target_rank) {
          // Reduction not possible, restore previous value
          deriv_levels[obs_idx]++;
        } else {
          found_reduction = true;
        }
      }

      keep_reducing = found_reduction;
    }

    return deriv_levels;
  }

  Eigen::MatrixXd constructReducedMatrix(
      const Eigen::MatrixXd& full_matrix,
      const std::unordered_map<size_t, long>& deriv_levels,
      size_t num_observables, size_t max_derivatives) {
    // Count rows in reduced matrix
    long total_rows = 0;
    for (size_t obs = 0; obs < num_observables; ++obs) {
      total_rows += deriv_levels.at(obs) + 1;
    }

    // Create reduced matrix
    Eigen::MatrixXd reduced(total_rows, full_matrix.cols());

    int row = 0;
    for (size_t obs = 0; obs < num_observables; ++obs) {
      for (int deriv = 0; deriv <= deriv_levels.at(obs); ++deriv) {
        long full_row = obs + deriv * num_observables;
        reduced.row(row++) = full_matrix.row(full_row);
      }
    }

    return reduced;
  }
};