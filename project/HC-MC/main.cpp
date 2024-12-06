#include <boost/numeric/odeint.hpp>
#include <functional>
#include <iostream>
#include <vector>

#include "HC-MC.hpp"
#include "ODESystem.hpp"
#include "integration_interface.hpp"
#include "types.hpp"

using namespace boost::numeric::odeint;

#include <cppad/cppad.hpp>
#include <vector>

template <typename T>
struct SystemConfig {
  std::vector<T> time_values;
  std::vector<T> true_params;
  std::vector<T> true_initial_state;
};

template <typename T>
struct SolverConfig {
  SolveOptions options;
  Vector<T> initial_guess;
};

SolverConfig<double> create_solver_config() {
  SolverConfig<double> config;
  config.options.debug = true;
  config.options.tolerance = 1e-6;
  config.options.maxIterations = 50;
  config.options.initialStepSize = 0.1;
  config.options.verifyDerivatives = true;

  config.initial_guess.resize(4);
  config.initial_guess[0] = 1.0;  // Guess for parameter p[0]
  config.initial_guess[1] = 1.0;  // Guess for parameter p[1]
  config.initial_guess[2] = 1.0;  // Guess for initial_state[0]
  config.initial_guess[3] = 0.5;  // Guess for initial_state[1]

  return config;
}

// Helper functions
SystemConfig<double> create_system_config() {
  return {
      {0.0, 0.5, 1.0, 1.5, 2.0},  // time_values
      {1.3, 1.8},                 // true_params
      {1.0, 0.5}                  // true_initial_state
  };
}

// State equation template
template <typename T>
struct MyStateEquations {
  void operator()(const state_vector<T>& x, state_vector<T>& dxdt,
                  const parameter_vector<T>& p, const T& /* t */) const {
    dxdt.resize(2);
    dxdt[0] = p[0] * x[0];
    dxdt[1] = p[1] * x[1];
  }
};

// Observation function template
template <typename T>
struct MyObservationFunction {
  void operator()(const state_vector<T>& x, observable_vector<T>& y) const {
    y.resize(2);
    y[0] = x[0];
    y[1] = x[1];
  }
};

// Create the true system with state equations and observation functions
auto create_true_system(const SystemConfig<double>& config) {
  /*  auto state_equations =
        [](const state_vector<double>& x, state_vector<double>& dxdt,
           const parameter_vector<double>& p, const double&  t ) {
          dxdt.resize(2);
          dxdt[0] = p[0] * x[0];
          dxdt[1] = p[1] * x[1];
        };

    auto observation_function = [](const state_vector<double>& x,
                                   observable_vector<double>& y) {
      y.resize(2);
      y[0] = x[0];
      y[1] = x[1];
    };*/

  ODESystem<double, MyStateEquations, MyObservationFunction> system(
      config.true_params, MyStateEquations<double>{},
      MyObservationFunction<double>{});
  system.setInitialState(config.true_initial_state);
  return system;
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

template <typename VectorType>
VectorType ode_system(const VectorType& y) {
  size_t n = y.size();
  VectorType dy_dt(n);
  dy_dt[0] = 1;  // Example: dy_0/dt = 1
  for (size_t k = 1; k < n; ++k) {
    dy_dt[k] = y[k - 1];  // dy_k/dt = y_{k-1}
  }
  return dy_dt;
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
    /*std::cout << "\n=== Parameter Objective Called ===\n";*/

    // Debug input parameters
    /*std::cout << "Input params: ";
    for (const auto& p : params) {
      std::cout << (p) << " ";
    }
    std::cout << "\n";*/

    // Extract parameters and initial state
    size_t param_size = 2;
    size_t state_size = 2;
    parameter_vector<AD<T>> p(params.data(), params.data() + param_size);
    state_vector<AD<T>> initial_state(params.data() + param_size,
                                      params.data() + param_size + state_size);

    /*std::cout << "Creating system with:\n";
    std::cout << "Parameters: " << (p[0]) << ", " << (p[1]) << "\n";
    std::cout << "Initial state: " << (initial_state[0]) << ", "
              << (initial_state[1]) << "\n";*/

    // Create the ODESystem instance directly with AD types
    ODESystem<AD<T>, MyStateEquations, MyObservationFunction> system(
        p, MyStateEquations<AD<T>>{}, MyObservationFunction<AD<T>>{});
    system.setInitialState(initial_state);

    // Simulate the system
    std::vector<AD<T>> ad_time_values(time_values.begin(), time_values.end());
    auto simulated_observables = integrate_and_observe(system, ad_time_values);

    // Debug simulated results
    /*std::cout << "\nSimulated observables:\n";
    for (size_t i = 0; i < simulated_observables.size(); ++i) {
      std::cout << "t = " << time_values[i] << ": ";
      for (const auto& val : simulated_observables[i]) {
        std::cout << (val) << " ";
      }
      std::cout << "\n";
    }*/

    // Calculate residuals
    auto observable_length = simulated_observables[0].size();
    ADVector residuals(observable_length * 2);

    size_t observable_count = simulated_observables.size();
    size_t midpoint = observable_count / 2;

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
  auto config = create_system_config();
  auto solver_config = create_solver_config();

  // Generate ground truth data
  auto true_system = create_true_system(config);
  auto ground_truth = generate_ground_truth(true_system, config.time_values);

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

int main() {
  example_parameter_estimation();

  return 0;
}
