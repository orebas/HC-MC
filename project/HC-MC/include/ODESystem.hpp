#pragma once

#include <algorithm>
#include <boost/numeric/odeint.hpp>
#include <functional>
#include <stdexcept>
#include <vector>

#include "hbtaylor.hpp"
#include "types.hpp"

// ODE system class
// Requirements:
//  - F and G should be templated functors, i.e. they should work with T or
//  TaylorSeries<T>.
//  - The user provides something like:
//    struct F {
//      template <typename RealType>
//      void operator()(const std::vector<RealType>& x,
//                      std::vector<RealType>& dxdt,
//                      const std::vector<RealType>& p,
//                      const RealType& t);
//    };
//    struct G {
//      template <typename RealType>
//      void operator()(const std::vector<RealType>& x,
//                      std::vector<RealType>& obs);
//    };
//
// This ensures we can handle both normal floats/doubles and TaylorSeries
// expansions.

template <typename T, template <typename> class StateFunctionTemplate,
          template <typename> class ObservationFunctionTemplate>
class ODESystem {
 public:
  using RealType = T;
  using StateType = std::vector<T>;
  using ParamType = std::vector<T>;
  using ObsType = std::vector<T>;

  // The state and observation functions are template templates
  // We instantiate them with T:
  StateFunctionTemplate<T> state_equations;
  ObservationFunctionTemplate<T> observation_function;

  ODESystem(const ParamType& params, const StateFunctionTemplate<T>& state_func,
            const ObservationFunctionTemplate<T>& obs_func, int state_size_in,
            int observable_size_in)
      : state_equations(state_func),
        observation_function(obs_func),
        parameters(params),
        parameter_size(params.size()),
        state_size(state_size_in),
        observable_size(observable_size_in) {}

  // Accessors
  const ParamType& getParameters() const { return parameters; }
  size_t getParameterSize() const { return parameter_size; }
  size_t getStateSize() const { return state_size; }
  size_t getObservableSize() const { return observable_size; }

  // Setters
  void setInitialState(const StateType& initial_state_param) {
    this->initial_state = initial_state_param;
    state_size = initial_state_param.size();
  }

  const StateType& getInitialState() const { return initial_state; }

  // Combine parameters and initial conditions into one vector
  std::vector<T> getFullParameterVector() const {
    std::vector<T> full(parameter_size + state_size);
    std::copy(parameters.begin(), parameters.end(), full.begin());
    std::copy(initial_state.begin(), initial_state.end(),
              full.begin() + parameter_size);
    return full;
  }

  // Set parameters and initial conditions from a single vector
  void setFullParameterVector(const std::vector<T>& full) {
    if (full.size() != parameter_size + state_size) {
      throw std::runtime_error(
          "Full vector size does not match parameter+state size.");
    }
    std::copy(full.begin(), full.begin() + parameter_size, parameters.begin());
    std::copy(full.begin() + parameter_size, full.end(), initial_state.begin());
  }

  // ODE function operator
  // For time stepping with fixed parameters
  void operator()(const StateType& x, StateType& dxdt, const T& t) const {
    state_equations(x, dxdt, parameters, t);
  }

  // ODE function operator if we provide parameters explicitly
  void operator()(const StateType& x, StateType& dxdt, const ParamType& p,
                  const T& t) const {
    state_equations(x, dxdt, p, t);
  }

  // Observation function
  void observe(const StateType& x, ObsType& y) const {
    observation_function(x, y);
  }

  // Shrinking the ODE system by fixing parameters:
  // This returns a new ODESystem with certain parameters fixed.
  // fixedIndices: which parameters to fix
  // fixedValues: values of those parameters
  // The resulting system will have fewer parameters.
  ODESystem<T, StateFunctionTemplate, ObservationFunctionTemplate>
  shrinkParameters(const std::vector<size_t>& fixedIndices,
                   const std::vector<T>& fixedValues) const {
    if (fixedIndices.size() != fixedValues.size()) {
      throw std::runtime_error(
          "Mismatched fixedIndices and fixedValues sizes.");
    }

    // Create a new parameter vector with those indices removed
    std::vector<bool> isFixed(parameter_size, false);
    for (auto idx : fixedIndices) {
      if (idx >= parameter_size) {
        throw std::runtime_error("Parameter index out of range");
      }
      isFixed[idx] = true;
    }

    ParamType newParams;
    newParams.reserve(parameter_size - fixedIndices.size());
    for (size_t i = 0; i < parameter_size; ++i) {
      if (!isFixed[i]) {
        newParams.push_back(parameters[i]);
      }
    }

    // Capture the fixed parameters in a wrapper for state_equations
    // We'll create a lambda that takes (x, dxdt, reducedParams, t),
    // reconstructs the full parameter set, and calls the original
    // state_equations.
    auto original_equations = state_equations;
    ParamType fixedParams = fixedValues;  // copy for lambda capture
    auto new_state_equations = [original_equations, isFixed, fixedParams](
                                   const std::vector<T>& x,
                                   std::vector<T>& dxdt,
                                   const std::vector<T>& reducedP, const T& t) {
      // Rebuild full parameter vector
      size_t reduced_idx = 0;
      std::vector<T> fullP;
      fullP.reserve(isFixed.size());
      for (size_t i = 0; i < isFixed.size(); ++i) {
        if (isFixed[i]) {
          fullP.push_back(fixedParams[std::distance(
              isFixed.begin(),
              std::find(isFixed.begin(), isFixed.end(), true))]);
        } else {
          fullP.push_back(reducedP[reduced_idx++]);
        }
      }
      original_equations(x, dxdt, fullP, t);
    };

    // Observation function does not change because it does not depend on
    // parameters explicitly (assuming linear in x).
    auto new_obs = observation_function;  // same observation function

    ODESystem<T, StateFunctionTemplate, ObservationFunctionTemplate>
        reducedSystem(
            newParams, new_state_equations,
            new_obs);  // TODO(orebas): fix state size and observable size
    reducedSystem.setInitialState(this->initial_state);
    return reducedSystem;
  }

  // Similarly, shrink initial conditions by fixing some of them:
  ODESystem<T, StateFunctionTemplate, ObservationFunctionTemplate>
  shrinkInitialConditions(const std::vector<size_t>& fixedStateIndices,
                          const std::vector<T>& fixedValues) const {
    if (fixedStateIndices.size() != fixedValues.size()) {
      throw std::runtime_error("Mismatched fixedStateIndices and fixedValues.");
    }

    std::vector<bool> isFixedState(state_size, false);
    for (auto idx : fixedStateIndices) {
      if (idx >= state_size) {
        throw std::runtime_error("State index out of range");
      }
      isFixedState[idx] = true;
    }

    // Create a reduced initial state
    StateType newInit;
    newInit.reserve(state_size - fixedStateIndices.size());
    for (size_t i = 0; i < state_size; ++i) {
      if (!isFixedState[i]) {
        newInit.push_back(initial_state[i]);
      }
    }

    // Create a wrapper around the state_equations that inserts the fixed state
    // variables
    auto original_equations = state_equations;
    StateType fixedStates = fixedValues;
    auto new_state_equations = [original_equations, isFixedState, fixedStates](
                                   const std::vector<T>& x,
                                   std::vector<T>& dxdt,
                                   const std::vector<T>& p, const T& t) {
      // Rebuild full state vector
      std::vector<T> fullX;
      fullX.reserve(isFixedState.size());
      size_t reduced_idx = 0;
      size_t fixed_count = 0;
      for (size_t i = 0; i < isFixedState.size(); ++i) {
        if (isFixedState[i]) {
          fullX.push_back(fixedStates[fixed_count++]);
        } else {
          fullX.push_back(x[reduced_idx++]);
        }
      }

      std::vector<T> fullDxdt(fullX.size(), T(0));
      original_equations(fullX, fullDxdt, p, t);

      // Now extract the reduced dxdt
      dxdt.resize(x.size());
      reduced_idx = 0;
      for (size_t i = 0; i < isFixedState.size(); ++i) {
        if (!isFixedState[i]) {
          dxdt[reduced_idx++] = fullDxdt[i];
        }
      }
    };

    auto new_obs = [this, isFixedState](const std::vector<T>& x,
                                        std::vector<T>& y) {
      // Rebuild full state vector for observation
      std::vector<T> fullX;
      fullX.reserve(isFixedState.size());
      size_t reduced_idx = 0;
      for (size_t i = 0; i < isFixedState.size(); ++i) {
        if (isFixedState[i]) {
          // For observation, use the fixed initial state value
          // (assuming it doesn't change - this might need refinement)
          fullX.push_back(
              T(0));  // or store the fixed states from initial condition?
        } else {
          fullX.push_back(x[reduced_idx++]);
        }
      }
      std::vector<T> fullObs;
      this->observation_function(fullX, fullObs);

      // The observation might still be full dimension since G is linear in x.
      // If G expects original dimension, we must return that dimension.
      // If we truly shrank the system dimension, G would need adjusting.
      // Here we assume G matches original dimension. If needed,
      // we could store a reduced G. For now, we just return fullObs.
      y = fullObs;
    };

    ODESystem<T, StateFunctionTemplate, ObservationFunctionTemplate>
        reducedSystem(this->parameters, new_state_equations, new_obs);
    reducedSystem.setInitialState(newInit);
    return reducedSystem;
  }

  /*
    // Compute Taylor series expansion of the state around time t0
    // The idea:
    //  X(t) = X(t0) + (t - t0)*X'(t0) + (t - t0)^2/2!*X''(t0) + ...
    // We use the provided TaylorSeries class to store coefficients.
    // This requires that state_equations can operate on TaylorSeries<T> as
    input. std::vector<TaylorSeries<T>> computeTaylorSeriesOfState(const T& t0,
                                                            int degree) const {
      // Initialize TaylorSeries for each state variable
      std::vector<TaylorSeries<T>> series;
      series.reserve(state_size);
      for (size_t i = 0; i < state_size; ++i) {
        TaylorSeries<T> ts(initial_state[i], degree);
        ts[0] = initial_state[i];  // constant term
        series.push_back(ts);
      }

      // We'll do a recursive approach:
      // For i=1 to degree:
      //   Evaluate F at t0 with the current expansions.
      //   The linear terms in t give first derivative, etc.
      //
      // However, a direct approach:
      // If we let the input be TaylorSeries(t0 + h), we can call F with x
      // represented as expansions. Then we solve for coefficients by matching
      // terms of equal powers of h.
      //
      // A simpler (though less general) approach:
      //   For the first derivative: dxdt at t0 is just F(X(t0), p, t0).
      //   For the second derivative: we differentiate again by applying chain
      //   rule, etc.
      // But we haven't implemented a general chain rule here. We assume that
      // `state_equations` can handle TaylorSeries to produce all higher
      // derivatives automatically.
      //
      // Let's represent time as a TaylorSeries around t0: T(t) = t0 + h, with h
      // as the variable. We'll create a TaylorSeries<T> for time: t_series = t0
    +
      // h where h is the expansion variable.
      TaylorSeries<T> t_series(t0, degree);
      // h = t - t0, so t_series[0] = t0, and t_series[1] = 1, and others 0
      // Actually, since evaluate methods rely on expansions:
      // Let's set up: t_series[0] = t0, t_series[1] = 1.0;
      t_series[0] = t0;
      if (degree > 0) {
        t_series[1] = T(1);
      }

      // At each step i, we will compute dx/dt = F(X, p, t_series)
      // X is currently known as a TaylorSeries expansion to order i-1.
      // We can call state_equations with these series. It should produce dxdt
    as
      // a series. Then we match coefficients to fill in series[*][i].

      // Convert parameters to TaylorSeries as well (constant expansions)
      std::vector<TaylorSeries<T>> p_series(parameters.size(),
                                            TaylorSeries<T>(T(0), degree));
      for (size_t i = 0; i < parameters.size(); ++i) {
        p_series[i][0] = parameters[i];  // parameter is constant w.r.t time
      }

      for (int i = 1; i <= degree; ++i) {
        // Prepare x_series for input
        std::vector<TaylorSeries<T>> x_series =
            series;  // copy current expansions
        std::vector<TaylorSeries<T>> dxdt_series(state_size,
                                                 TaylorSeries<T>(T(0), degree));

        // Call state_equations but templated on TaylorSeries<T>
        // We'll create a small adapter that calls the template:
        {
          // Wrap parameters in a lambda that uses TaylorSeries:
          auto eq = [this](const std::vector<TaylorSeries<T>>& x,
                           std::vector<TaylorSeries<T>>& dxdt,
                           const std::vector<TaylorSeries<T>>& p,
                           const TaylorSeries<T>& tt) {
            // Extract plain arrays for calling original functor:
            // Here we rely on the fact that `state_equations` is already
            // templated. state_equations(x, dxdt, p, t)
            this->state_equations(x, dxdt, p, tt);
          };

          eq(x_series, dxdt_series, p_series, t_series);
        }

        // Now, dxdt_series[j][n] corresponds to nth derivative terms. By
        // definition:
        //   series[j][i] = dxdt[j][i-1] / i   (to get ith coefficient)
        // Because coefficient for order i in Taylor series = f^(i)(t0)/i!
        // dxdt_series gives the derivative expansions. The linear term in
        // dxdt_series is dx/dt at t0, etc. We must ensure that we properly
        // extract the i-th coefficient from dxdt_series. dxdt_series[j][i-1]
        // should give us the i-th derivative * i!. But we've constructed
        // dxdt_series so that [k] is the k-th coefficient i.e.,
        // dxdt_series[j][i-1] is actually the (i-1)-th derivative / (i-1)! ?
    For
        // simplicity, assume dxdt_series is computed similarly so that:
        //   series[j][i] = dxdt_series[j][i-1] / i
        // This matches the pattern in the provided `computeODECoefficients`
        // snippet.

        for (size_t j = 0; j < state_size; ++j) {
          series[j][i] = dxdt_series[j][i - 1] / T(i);
        }
      }

      return series;
    }*/

  std::vector<TaylorSeries<T>> computeTaylorSeriesOfState(const T& t0,
                                                          int degree) const {
    // Create a wrapper that adapts our state_equations to the interface
    // expected by computeODECoefficients
    auto wrapped_system = [this, &t0](
                              const std::vector<TaylorSeries<T>>& series,
                              std::vector<TaylorSeries<T>>& result) {
      // Convert parameters to TaylorSeries (as constants)
      std::vector<TaylorSeries<T>> p_series;
      p_series.reserve(parameters.size());
      for (const auto& p : parameters) {
        p_series.emplace_back(p, series[0].getDegree());
      }
      // Call state_equations with the wrapped parameters
      StateFunctionTemplate<TaylorSeries<T>>()(series, result, p_series, t0);
    };

    return computeODECoefficients(this->initial_state, wrapped_system, degree);
  }

  // Compute Taylor series of the observables:
  std::vector<TaylorSeries<T>> computeTaylorSeriesOfObservables(
      const std::vector<TaylorSeries<T>>& x_series) const {
    // We assume G is linear, so G(X(t)) = A * X(t) for some matrix A.
    // If G is truly linear, applying it to a TaylorSeries vector is
    // straightforward: Just apply G to each coefficient independently.

    // We'll just call observation_function with the TaylorSeries expansions and
    // let it handle them. If G is linear: G(ax + b) = aG(x) + G(b), linear
    // means: G(x) = Mx for some matrix M. We'll call observation_function once
    // with TaylorSeries input.
    std::vector<TaylorSeries<T>> y_series(
        observable_size, TaylorSeries<T>(T(0), x_series[0].getDegree()));

    ObservationFunctionTemplate<TaylorSeries<T>>()(x_series, y_series);
    return y_series;
  }

 private:
  ParamType parameters;
  StateType initial_state;
  size_t parameter_size;
  size_t state_size;
  mutable size_t observable_size;
};
