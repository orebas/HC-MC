#include <boost/numeric/odeint.hpp>
#include <iostream>
#include <vector>

#include "HC-MC.hpp"

using namespace boost::numeric::odeint;

// Define AD types as used in HC-MC.hpp
using ADDouble = CppAD::AD<double>;
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using ADVector = Vector<ADDouble>;

// Template the state type and system for AD compatibility
template <typename T>
using state_vector = std::vector<T>;

// Harmonic oscillator system templated for AD types
template <typename T>
struct harmonic_oscillator {
  T m, k;

  harmonic_oscillator(const T& mass, const T& spring_constant)
      : m(mass), k(spring_constant) {}

  void operator()(const state_vector<T>& x, state_vector<T>& dxdt,
                  const double /* t */) {
    dxdt[0] = x[1];
    dxdt[1] = -k / m * x[0];
  }
};

// Helper function templated for AD types
template <typename T>
state_vector<T> integrate_to_time(const T& mass, const T& k,
                                  const state_vector<T>& initial_state,
                                  const double target_time) {
  const harmonic_oscillator<T> system(mass, k);
  state_vector<T> state = initial_state;

  typedef runge_kutta4<state_vector<T>> stepper_type;
  integrate_const(stepper_type(), system, state, 0.0, target_time, 0.01);

  return state;
}

// Function to generate ground truth data (using regular doubles)
std::vector<double> generate_ground_truth(double target_time) {
  const double true_mass = 1.3;
  const double true_k = 1.8;
  const std::vector<double> initial_state = {1.0, 0.0};

  return integrate_to_time<double>(true_mass, true_k, initial_state,
                                   target_time);
}

int main() {
  const double target_time = 2.0;
  std::vector<double> ground_truth = generate_ground_truth(target_time);

  std::cout << "Ground truth state at t=" << target_time << ":\n";
  std::cout << "Position: " << ground_truth[0] << "\n";
  std::cout << "Velocity: " << ground_truth[1] << "\n";

  // Create objective function for the solver
  // Parameters are [mass, spring_constant]
  auto parameter_objective =
      [target_time, &ground_truth](const ADVector& params) -> ADVector {
    // Extract parameters (keeping AD type)
    const ADDouble& mass = params(0);
    const ADDouble& k = params(1);

    // Initial conditions (convert to AD type)
    const state_vector<ADDouble> initial_state = {ADDouble(1.0), ADDouble(0.0)};

    // Integrate system with AD types
    state_vector<ADDouble> final_state =
        integrate_to_time<ADDouble>(mass, k, initial_state, target_time);

    // Create residual vector
    ADVector residual(2);
    residual(0) = mass * (final_state[0] - ADDouble(ground_truth[0]));
    residual(1) = k * (final_state[1] - ADDouble(ground_truth[1]));

    return residual;
  };

  // Set solver options
  SolveOptions options;
  options.debug = true;
  options.tolerance = 1e-6;
  options.maxIterations = 50;
  options.initialStepSize = 0.1;
  options.verifyDerivatives = true;

  // Solve the system
  auto result = solve(parameter_objective, 2, options);

  if (result) {
    std::cout << "\nSolution found!\n";
    std::cout << "Estimated mass: " << CppAD::Value(result->solution(0))
              << "\n";
    std::cout << "Estimated k: " << CppAD::Value(result->solution(1)) << "\n";
    std::cout << "Residual norm: " << result->residualNorm << "\n";
    std::cout << "Iterations: " << result->iterations << "\n";
  } else {
    std::cout << "Failed to find solution\n";
  }

  return 0;
}