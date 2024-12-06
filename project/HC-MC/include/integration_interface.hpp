#pragma once

#include <boost/numeric/odeint.hpp>

// Function to integrate and observe
template <typename T, typename System>
std::vector<observable_vector<T>> integrate_and_observe(
    const System& system, const std::vector<T>& time_values) {
  using namespace boost::numeric::odeint;

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
