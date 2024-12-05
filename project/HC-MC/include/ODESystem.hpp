#pragma once

#include <boost/numeric/odeint.hpp>
#include <functional>
#include <vector>

#include "types.hpp"

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
