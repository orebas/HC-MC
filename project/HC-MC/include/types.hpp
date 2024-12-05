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

// Type aliases to improve readability
template <typename Base>
using AD = CppAD::AD<Base>;
using ADDouble = AD<double>;

template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

using DoubleVector = Vector<double>;

// Helper type trait to get base type
template <typename T>
struct BaseType {
  using type = T;
};

template <typename Base>
struct BaseType<AD<Base>> {
  using type = Base;
};