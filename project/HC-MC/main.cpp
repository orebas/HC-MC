#include <boost/numeric/odeint.hpp>
#include <functional>
#include <iostream>
#include <vector>

#include "HC-MC.hpp"

using namespace boost::numeric::odeint;

template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

// Define AD types as used in HC-MC.hpp
using ADDouble = CppAD::AD<double>;
using ADVector = Vector<ADDouble>;

// Template the state type and system for AD compatibility
template <typename T>
using state_vector = std::vector<T>;
template <typename T>
using parameter_vector = std::vector<T>;
template <typename T>
using observable_vector = std::vector<T>;

// ODE system class
template <typename T, typename StateFunctionType,
          typename ObservationFunctionType>
class ODESystem {
 public:
  // using StateFunction =
  //     std::function<void(const state_vector<T>&, state_vector<T>&, const
  //     T&)>;
  // using ObservationFunction =
  //     std::function<void(const state_vector<T>&, observable_vector<T>&)>;

  // Constructors
  ODESystem(const parameter_vector<T>& params, StateFunctionType& state_func,
            ObservationFunctionType& obs_func)
      : parameters(params),
        state_equations(state_func),
        observation_function(obs_func),
        parameter_size(params.size()),
        state_size(0),
        observable_size(0)  // Will be set when observation function is called
  {}

  // Accessors
  const parameter_vector<T>& getParameters() const { return parameters; }
  size_t getParameterSize() const { return parameter_size; }
  size_t getStateSize() const { return state_size; }
  size_t getObservableSize() const { return observable_size; }

  // Setters
  void setInitialState(const state_vector<T>& initial_state_param) {
    this->initial_state = initial_state_param;
    state_size = initial_state_param.size();
  }

  // State equations and observation function
  void operator()(const state_vector<T>& x, state_vector<T>& dxdt,
                  const T& t) const {
    state_equations(x, dxdt, parameters, t);
  }

  void operator()(const state_vector<T>& x, state_vector<T>& dxdt,
                  const parameter_vector<T>& p, const T& t) const {
    parameters = p;
    state_equations(x, dxdt, parameters, t);
  }

  void observe(const state_vector<T>& x, observable_vector<T>& y) const {
    observation_function(x, y);
    if (observable_size == 0) {
      observable_size = y.size();  // Set observable size on first call
    }
  }

  // Initial state
  const state_vector<T>& getInitialState() const { return initial_state; }

 private:
  parameter_vector<T> parameters;
  StateFunctionType state_equations;
  ObservationFunctionType observation_function;

  state_vector<T> initial_state;
  size_t parameter_size;
  size_t state_size;
  mutable size_t observable_size;
};

// Function to integrate and observe
template <typename T, typename F1, typename F2>
std::vector<observable_vector<T>> integrate_and_observe(
    const ODESystem<T, F1, F2>& system, const std::vector<T>& time_values) {
  state_vector<T> state = system.getInitialState();
  std::vector<observable_vector<T>> observables;

  auto observer = [&](const state_vector<T>& x, const T& /* t */) {
    observable_vector<T> y;
    system.observe(x, y);
    observables.push_back(y);
  };

  typedef runge_kutta4<state_vector<T>, T, state_vector<T>, T> stepper_type;
  integrate_times(stepper_type(), system, state, time_values.begin(),
                  time_values.end(), T(0.01), observer);
  return observables;
}

int main() {
  // Define the time points at which we observe the system
  std::vector<double> time_values = {0.0, 0.5, 1.0, 1.5, 2.0};

  // True parameters and initial state
  parameter_vector<double> true_params = {1.3, 1.8};
  state_vector<double> true_initial_state = {1.0, 0.5};

  // Define the state equations function
  auto state_equations =
      [](const state_vector<double>& x, state_vector<double>& dxdt,
         const parameter_vector<double>& p, const double& /* t */) {
        dxdt.resize(2);
        dxdt[0] = p[0] * x[0];
        dxdt[1] = p[1] * x[1];
      };

  // Define the observation function
  auto observation_function = [](const state_vector<double>& x,
                                 observable_vector<double>& y) {
    y.resize(2);
    y[0] = x[0];
    y[1] = x[1];
  };

  // Create the ODESystem instance
  ODESystem<double, decltype(state_equations), decltype(observation_function)>
      true_system(true_params, state_equations, observation_function);
  true_system.setInitialState(true_initial_state);

  // Generate ground truth observables
  auto ground_truth = integrate_and_observe(true_system, time_values);

  // Display ground truth data
  for (size_t i = 0; i < time_values.size(); ++i) {
    std::cout << "t = " << time_values[i] << ", Observables: ";
    for (const auto& val : ground_truth[i]) {
      std::cout << val << " ";
    }
    std::cout << "\n";
  }

  using T = ADDouble;

  // Objective function for parameter estimation
  auto parameter_objective =
      [time_values, &ground_truth](const ADVector& params) -> ADVector {
    // Extract parameters and initial state
    size_t param_size = 2;
    size_t state_size = 2;
    parameter_vector<T> p(params.data(), params.data() + param_size);
    state_vector<T> initial_state(params.data() + param_size,
                                  params.data() + param_size + state_size);

    // Define the state equations function
    auto state_equations = [](const state_vector<T>& x, state_vector<T>& dxdt,
                              const parameter_vector<T>& lp, const T& /* t */) {
      dxdt.resize(2);
      dxdt[0] = lp[0] * x[0];
      dxdt[1] = lp[1] * x[1];  //+ x[0] * T(0.1);
    };

    // Define the observation function
    auto observation_function = [](const state_vector<T>& x,
                                   observable_vector<T>& y) {
      y.resize(2);
      y[0] = x[0];
      y[1] = x[1];
    };

    // Create the ODESystem instance
    ODESystem<T, decltype(state_equations), decltype(observation_function)>
        system(p, state_equations, observation_function);
    system.setInitialState(initial_state);

    // Simulate the system
    std::vector<T> ad_time_values(time_values.begin(), time_values.end());
    auto simulated_observables = integrate_and_observe(system, ad_time_values);

    // Display current interation observations

    for (size_t i = 0; i < time_values.size(); ++i) {
      std::cout << "t = " << time_values[i] << ", Observables: ";
      for (const auto& val : simulated_observables[i]) {
        std::cout << val << " ";
      }
      std::cout << "\n";
    }

    // Calculate residuals

    auto observable_length = simulated_observables[0].size();
    // size_t total_residuals =
    //     simulated_observables.size() * 2;  // observable_size = 2
    ADVector residuals(observable_length * 2);

    size_t observable_count = simulated_observables.size();
    size_t midpoint = observable_count / 2;

    size_t idx = 0;
    size_t i = 0;
    for (size_t j = 0; j < simulated_observables[i].size(); ++j) {
      residuals[idx++] = simulated_observables[i][j] - T(ground_truth[i][j]);
    }
    i = midpoint;
    for (size_t j = 0; j < simulated_observables[i].size(); ++j) {
      residuals[idx++] = simulated_observables[i][j] - T(ground_truth[i][j]);
    }

    return residuals;
  };

  // Set solver options
  SolveOptions options;
  options.debug = true;
  options.tolerance = 1e-6;
  options.maxIterations = 50;
  options.initialStepSize = 0.1;
  options.verifyDerivatives = true;

  // Initial guess for parameters and initial state
  ADVector initial_params(4);
  initial_params[0] = 1.0;  // Guess for parameter p[0]
  initial_params[1] = 1.0;  // Guess for parameter p[1]
  initial_params[2] = 1.0;  // Guess for initial_state[0]
  initial_params[3] = 0.5;  // Guess for initial_state[1]

  // Solve the parameter estimation problem
  auto result = solve(parameter_objective, 4,  // initial_params,
                      options);

  if (result) {
    std::cout << "\nSolution found!\n";
    std::cout << "Estimated parameters: ";
    for (size_t i = 0; i < 2; ++i) {  // parameter_size = 2
      std::cout << CppAD::Value(result->solution[i]) << " ";
    }
    std::cout << "\nEstimated initial state: ";
    for (size_t i = 2; i < 4; ++i) {  // state_size = 2
      std::cout << CppAD::Value(result->solution[i]) << " ";
    }
    std::cout << "\nResidual norm: " << result->residualNorm << "\n";
    std::cout << "Iterations: " << result->iterations << "\n";
  } else {
    std::cout << "Failed to find solution\n";
  }

  return 0;
}
