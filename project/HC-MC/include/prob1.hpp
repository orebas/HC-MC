#pragma once

#include <vector>

#include "ODESystem.hpp"
#include "types.hpp"

// State equation template for Problem 1
template <typename T>
struct MyStateEquations {
  void operator()(const state_vector<T>& x, state_vector<T>& dxdt,
                  const parameter_vector<T>& p,
                  [[maybe_unused]] const T& t) const {
    dxdt.resize(2);
    dxdt[0] = p[0] * x[0];
    dxdt[1] = p[1] * x[1];
  }
};

// Observation function template for Problem 1
template <typename T>
struct MyObservationFunction {
  void operator()(const state_vector<T>& x, observable_vector<T>& y) const {
    y.resize(2);
    y[0] = x[0];
    y[1] = x[1];
  }
};

// Problem-specific configuration
template <typename T>
struct SystemConfig {
  std::vector<T> time_values;
  std::vector<T> true_params;
  std::vector<T> true_initial_state;
};

// Helper functions for Problem 1
inline SystemConfig<double> create_system_config() {
  return {
      {0.0, 0.5, 1.0, 1.5, 2.0},  // time_values
      {1.3, 1.8},                 // true_params
      {1.0, 0.5}                  // true_initial_state
  };
}

// Create the true system with state equations and observation functions
template <typename T>
auto create_true_system(const SystemConfig<T>& config) {
  ODESystem<T, MyStateEquations, MyObservationFunction> system(
      config.true_params, MyStateEquations<T>{}, MyObservationFunction<T>{}, 2,
      2);
  system.setInitialState(config.true_initial_state);
  return system;
}
