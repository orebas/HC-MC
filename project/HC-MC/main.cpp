#include <boost/numeric/odeint.hpp>
#include <functional>
#include <iostream>
#include <vector>

#include "HC-MC.hpp"
#include "ODESystem.hpp"
#include "hbtaylor.hpp"
#include "integration_interface.hpp"
#include "observability.hpp"
#include "prob1.hpp"
#include "types.hpp"

using boost::numeric::odeint::integrate_times;
using boost::numeric::odeint::runge_kutta4;

#include "eigenwrap.hpp"

template <typename T>
struct SolverConfig {
  SolveOptions options;
  //  Vector<T> initial_guess;
};

inline SolverConfig<double> create_solver_config() {
  SolverConfig<double> config;
  config.options.debug = true;
  config.options.tolerance = 1e-6;
  config.options.maxIterations = 50;
  config.options.initialStepSize = 0.1;
  config.options.verifyDerivatives = true;

  /*config.initial_guess.resize(4);
  config.initial_guess[0] = 1.0;  // Guess for parameter p[0]
  config.initial_guess[1] = 1.0;  // Guess for parameter p[1]
  config.initial_guess[2] = 1.0;  // Guess for initial_state[0]
  config.initial_guess[3] = 0.5;  // Guess for initial_state[1]
*/
  return config;
}

// Function to print results
void print_solution_results(const std::string& method_name,
                            const std::optional<SolveResult>& result) {
  if (result) {
    std::cout << "\nSolution found!\n";
    std::cout << "Estimated parameters: ";
    for (size_t i = 0; i < 2; ++i) {
      std::cout << (result->solution[i]) << " ";
    }
    std::cout << "\nEstimated initial state: ";
    for (size_t i = 2; i < 4; ++i) {
      std::cout << (result->solution[i]) << " ";
    }
    std::cout << "\nResidual norm: " << result->residualNorm << "\n";
    std::cout << "Iterations: " << result->iterations << "\n";
  } else {
    std::cout << "Failed to find solution with " << method_name << "\n";
  }
}

// Run both solvers and print results
void run_solvers(const auto& objective, const SolverConfig<double>& config) {
  std::cout << "\n=== Solving with original method ===\n";
  auto result = solve(objective, 4, config.options);
  print_solution_results("original method", result);

  std::cout << "\n=== Solving with Newton method ===\n";
  auto newton_result = solve_newton(objective, 4, config.options);
  print_solution_results("Newton method", newton_result);
}

// Generate ground truth data from the system
template <typename System>
std::vector<observable_vector<double>> generate_ground_truth(
    const System& system, const std::vector<double>& time_values) {
  return integrate_and_observe(system, time_values);
}

// Create the parameter estimation objective function
template <typename T>
auto create_parameter_objective(
    const SystemConfig<double>& config, const std::vector<double>& time_values,
    const std::vector<observable_vector<double>>& ground_truth) {
  return [config, time_values,
          ground_truth](const ADVector& params) -> ADVector {
    const size_t param_size = 2;
    const size_t state_size = 2;
    const parameter_vector<AD<T>> p(params.data(), params.data() + param_size);
    const state_vector<AD<T>> initial_state(
        params.data() + param_size, params.data() + param_size + state_size);

    // Create the ODESystem instance directly with AD types
    ODESystem<AD<T>, MyStateEquations, MyObservationFunction> system(
        p, MyStateEquations<AD<T>>{}, MyObservationFunction<AD<T>>{}, 2, 2);
    system.setInitialState(initial_state);

    // Simulate the system
    const std::vector<AD<T>> ad_time_values(time_values.begin(),
                                            time_values.end());
    const auto simulated_observables =
        integrate_and_observe(system, ad_time_values);

    // Calculate residuals
    const auto observable_length = simulated_observables[0].size();
    ADVector residuals(observable_length * 2);

    const size_t observable_count = simulated_observables.size();
    const size_t midpoint = observable_count / 2;

    // Only use first and middle point for residuals
    size_t idx = 0;
    size_t i = 0;
    for (size_t j = 0; j < simulated_observables[i].size(); ++j) {
      residuals[idx] = simulated_observables[i][j] - T(ground_truth[i][j]);
      // std::cout << "Residual " << idx << ": " << (residuals[idx]) << " = "
      //           << (simulated_observables[i][j]) << " - " <<
      //           ground_truth[i][j]
      //           << "\n";
      idx++;
    }
    i = midpoint;
    for (size_t j = 0; j < simulated_observables[i].size(); ++j) {
      residuals[idx] = simulated_observables[i][j] - T(ground_truth[i][j]);
      // std::cout << "Residual " << idx << ": " << (residuals[idx]) << " = "
      //           << (simulated_observables[i][j]) << " - " <<
      //           ground_truth[i][j]
      //           << "\n";
      idx++;
    }

    return residuals;
  };
}

// Break down the main function into smaller pieces
void example_parameter_estimation() {
  // Setup configuration
  const auto config = create_system_config();
  const auto solver_config = create_solver_config();

  // Generate ground truth data
  const auto true_system = create_true_system(config);
  const auto ground_truth =
      generate_ground_truth(true_system, config.time_values);

  // Display ground truth data
  std::cout << "\nGround truth data:\n";
  for (size_t i = 0; i < config.time_values.size(); ++i) {
    std::cout << "t = " << config.time_values[i] << ", Observables: ";
    for (const auto& val : ground_truth[i]) {
      std::cout << val << " ";
    }
    std::cout << "\n";
  }

  // Create and run solvers
  auto objective = create_parameter_objective<double>(
      config, config.time_values, ground_truth);
  run_solvers(objective, solver_config);
}

void print_taylor_series(const std::vector<TaylorSeries<double>>& series,
                         const std::string& label) {
  std::cout << "\n" << label << " Taylor series coefficients:\n";
  for (size_t i = 0; i < series.size(); ++i) {
    std::cout << "x_" << i << "(t) = ";
    for (int j = 0; j <= series[i].getDegree(); ++j) {
      if (j > 0) std::cout << " + ";
      std::cout << series[i][j];
      if (j > 0) std::cout << "*t^" << j;
    }
    std::cout << "\n";
  }
}

void analyze_taylor_series() {
  // Create system configuration with example parameters
  const auto config = create_system_config();

  // Create the ODE system with T = double
  const auto system = create_true_system(config);

  // Compute Taylor series around t0 = 0 using double-based system
  const double t0 = 0.0;
  const int degree = 5;  // Compute up to 5th order terms

  std::cout << "\nComputing Taylor series for the ODE system:";
  std::cout << "\nParameters: p1 = " << config.true_params[0]
            << ", p2 = " << config.true_params[1];
  std::cout << "\nInitial state: x1(0) = " << config.true_initial_state[0]
            << ", x2(0) = " << config.true_initial_state[1] << "\n";

  // Compute Taylor series of the state variables
  auto state_series = system.computeTaylorSeriesOfState(t0, degree);
  print_taylor_series(state_series, "State");

  // Compute Taylor series of the observables
  auto observable_series =
      system.computeTaylorSeriesOfObservables(state_series);
  print_taylor_series(observable_series, "Observable");
}

int obs_example() {
  // Example matrix dimensions
  const size_t num_rows = 20;   // Total rows in observability matrix
  const size_t num_cols = 5;    // Number of parameters
  const size_t num_obs = 2;     // Number of observable variables
  const size_t max_derivs = 4;  // Maximum derivatives per observable

  // Create sample observability matrix
  Eigen::MatrixXd obs_matrix = Eigen::MatrixXd::Random(num_rows, num_cols);

  // Optional: Parameter names for better output
  std::vector<std::string> param_names = {"p1", "p2", "p3", "p4", "p5"};

  // Create analyzer with default tolerances
  ObservabilityAnalyzer analyzer;

  // Analyze the matrix
  DerivativeLevels result =
      analyzer.analyze(obs_matrix, num_obs, max_derivs, param_names);

  // Print results
  std::cout << "Derivative levels needed:\n";
  for (const auto& [obs_idx, deriv_count] : result.derivative_levels) {
    std::cout << "Observable " << obs_idx << ": " << deriv_count
              << " derivatives\n";
  }

  return 0;
}

int main() {
  try {
    analyze_taylor_series();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
  }

  example_parameter_estimation();
  obs_example();

  return 0;
}
