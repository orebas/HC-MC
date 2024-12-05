#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

template <typename T>
class TaylorSeries {
 private:
  int degree;                   // Degree of the Taylor series
  std::vector<T> coefficients;  // coefficients[i] represents the i-th
                                // derivative divided by i!
  static constexpr T epsilon = std::numeric_limits<T>::epsilon();

 public:
  // Constructor initializes all coefficients to zero
  TaylorSeries(const T &constant, int degree_in = 0)
      : degree(degree_in), coefficients(degree_in + 1, T(0)) {
    coefficients[0] = constant;
  }

  TaylorSeries()
      : TaylorSeries(T(0), 0) {
  }  // Default constructor delegates to existing constructor

  // Access operators
  T &operator[](int idx) { return coefficients[idx]; }
  const T &operator[](int idx) const { return coefficients[idx]; }

  // Basic arithmetic operators
  TaylorSeries operator+(const TaylorSeries &other) const {
    int max_degree = std::max(degree, other.degree);
    TaylorSeries result(T(0), max_degree);
    for (int i = 0; i <= max_degree; ++i) {
      result[i] = (i <= degree ? coefficients[i] : T(0)) +
                  (i <= other.degree ? other[i] : T(0));
    }
    return result;
  }

  TaylorSeries operator-(const TaylorSeries &other) const {
    int max_degree = std::max(degree, other.degree);
    TaylorSeries result(T(0), max_degree);
    for (int i = 0; i <= max_degree; ++i) {
      result[i] = (i <= degree ? coefficients[i] : T(0)) -
                  (i <= other.degree ? other[i] : T(0));
    }
    return result;
  }

  TaylorSeries operator*(const TaylorSeries &other) const {
    int result_degree = degree + other.degree;
    TaylorSeries result(T(0), result_degree);
    for (int n = 0; n <= result_degree; ++n) {
      for (int k = 0; k <= n; ++k) {
        if (k <= degree && (n - k) <= other.degree) {
          result[n] += coefficients[k] * other[n - k];
        }
      }
    }
    return result;
  }

  TaylorSeries operator/(const TaylorSeries &other) const {
    if (std::abs(other[0]) < epsilon) {
      throw std::runtime_error("Division by near-zero in Taylor series");
    }

    int result_degree = degree;
    TaylorSeries result(T(0), result_degree);
    result[0] = coefficients[0] / other[0];

    for (int n = 1; n <= result_degree; ++n) {
      T sum = coefficients[n];
      for (int k = 1; k <= n; ++k) {
        if (k <= other.degree) {
          sum -= other[k] * result[n - k];
        }
      }
      result[n] = sum / other[0];
    }
    return result;
  }

  // Scalar multiplication
  TaylorSeries operator*(const T &scalar) const {
    TaylorSeries result(T(0), degree);
    for (int i = 0; i <= degree; ++i) {
      result[i] = coefficients[i] * scalar;
    }
    return result;
  }

  // Scalar division
  TaylorSeries operator/(const T &scalar) const {
    if (std::abs(scalar) < epsilon) {
      throw std::runtime_error("Division by near-zero scalar in Taylor series");
    }
    TaylorSeries result(T(0), degree);
    for (int i = 0; i <= degree; ++i) {
      result[i] = coefficients[i] / scalar;
    }
    return result;
  }

  TaylorSeries &operator*=(const T &scalar) {
    for (int i = 0; i <= degree; ++i) {
      coefficients[i] *= scalar;
    }
    return *this;
  }

  TaylorSeries &operator/=(const T &scalar) {
    if (std::abs(scalar) < epsilon) {
      throw std::runtime_error("Division by near-zero scalar in Taylor series");
    }
    for (int i = 0; i <= degree; ++i) {
      coefficients[i] /= scalar;
    }
    return *this;
  }

  // Compound assignment operators
  TaylorSeries &operator+=(const TaylorSeries &other) {
    *this = *this + other;
    return *this;
  }

  TaylorSeries &operator-=(const TaylorSeries &other) {
    *this = *this - other;
    return *this;
  }

  // Evaluate series at a point
  // Evaluate series at a point x
  T evaluate(T x_value) const {
    T x0 = coefficients[0];  // Center of the series
    T h = x_value - x0;      // Deviation from the center
    T result = coefficients[degree];
    for (int i = degree - 1; i >= 0; --i) {
      result = result * h + coefficients[i];
    }
    return result;
  }

  // Get derivative
  TaylorSeries derivative() const {
    if (degree == 0) {
      throw std::runtime_error(
          "Cannot take derivative of degree 0 Taylor series");
    }
    TaylorSeries result(T(0), degree - 1);
    for (int i = 0; i < degree; ++i) {
      result[i] = coefficients[i + 1] * T(i + 1);
    }
    return result;
  }

  // Friend functions for commutative operations
  friend TaylorSeries operator*(T scalar, const TaylorSeries &ts) {
    return ts * scalar;
  }

  // Stream output
  friend std::ostream &operator<<(std::ostream &os, const TaylorSeries &ts) {
    os << ts[0];
    for (int i = 1; i <= ts.degree; ++i) {
      if (ts[i] != T(0)) {
        os << " + " << ts[i] << "x^" << i;
      }
    }
    return os;
  }

  // Unary minus operator
  TaylorSeries operator-() const {
    TaylorSeries result(T(0), degree);
    for (int i = 0; i <= degree; ++i) {
      result[i] = -coefficients[i];
    }
    return result;
  }

  // Updated transcendental functions using recursive formulas
  static TaylorSeries exp(const TaylorSeries &x) {
    TaylorSeries result(std::exp(x[0]), x.degree);
    for (int n = 1; n <= x.degree; ++n) {
      T sum = T(0);
      for (int k = 1; k <= n; ++k) {
        sum += T(k) * x[k] * result[n - k];
      }
      result[n] = sum / T(n);
    }
    return result;
  }

  static TaylorSeries log(const TaylorSeries &x) {
    if (x[0] <= T(0)) {
      throw std::runtime_error("Log of non-positive number");
    }

    TaylorSeries result(std::log(x[0]), x.degree);

    for (int n = 1; n <= x.degree; ++n) {
      T sum = x[n];
      for (int k = 1; k <= n - 1; ++k) {
        sum -= (T(k) * x[n - k] * result[k]) / T(n);
      }
      result[n] = sum / x[0];
    }
    return result;
  }

  static void sincos(const TaylorSeries &x, TaylorSeries &s, TaylorSeries &c) {
    s = TaylorSeries(std::sin(x[0]), x.degree);
    c = TaylorSeries(std::cos(x[0]), x.degree);
    for (int n = 1; n <= x.degree; ++n) {
      T sum_s = T(0);
      T sum_c = T(0);
      for (int k = 1; k <= n; ++k) {
        sum_s += T(k) * x[k] * c[n - k];
        sum_c -= T(k) * x[k] * s[n - k];
      }
      s[n] = sum_s / T(n);
      c[n] = sum_c / T(n);
    }
  }

  static TaylorSeries sin(const TaylorSeries &x) {
    TaylorSeries s(T(0), x.degree);
    TaylorSeries c(T(0), x.degree);
    sincos(x, s, c);
    return s;
  }

  static TaylorSeries cos(const TaylorSeries &x) {
    TaylorSeries s(T(0), x.degree);
    TaylorSeries c(T(0), x.degree);
    sincos(x, s, c);
    return c;
  }

  static TaylorSeries sqrt(const TaylorSeries &x) {
    if (x[0] <= T(0)) {
      throw std::runtime_error("Sqrt of non-positive number");
    }

    TaylorSeries result(std::sqrt(x[0]), x.degree);

    for (int n = 1; n <= x.degree; ++n) {
      T sum = T(0);
      for (int k = 1; k <= n - 1; ++k) {
        sum += result[k] * result[n - k];
      }
      result[n] = (x[n] - sum / T(2)) / (T(2) * result[0]);
    }
    return result;
  }
  template <typename S>
  static TaylorSeries pow(const TaylorSeries &x, S exponent) {
    // Using the identity pow(x, a) = exp(a * log(x)) for non-integer exponents
    if (std::is_floating_point<S>::value || exponent < 0) {
      return exp(log(x) * exponent);
    }

    // For non-negative integer exponents, use repeated squaring
    if (exponent == 0) {
      return TaylorSeries(T(1), x.degree);
    }
    if (exponent == 1) {
      return x;
    }

    // Handle even vs odd exponents
    TaylorSeries half = pow(x, exponent / 2);
    TaylorSeries result = half * half;
    if (static_cast<int>(exponent) & 1) {
      result = result * x;
    }  // Use bitwise AND to check if odd
    return result;
  }

  static TaylorSeries tan(const TaylorSeries &x) { return sin(x) / cos(x); }
};

// Global overloads for transcendental functions
template <typename T>
TaylorSeries<T> sin(const TaylorSeries<T> &x) {
  return TaylorSeries<T>::sin(x);
}

template <typename T>
TaylorSeries<T> cos(const TaylorSeries<T> &x) {
  return TaylorSeries<T>::cos(x);
}

template <typename T>
TaylorSeries<T> exp(const TaylorSeries<T> &x) {
  return TaylorSeries<T>::exp(x);
}

template <typename T>
TaylorSeries<T> log(const TaylorSeries<T> &x) {
  return TaylorSeries<T>::log(x);
}

template <typename T>
TaylorSeries<T> tan(const TaylorSeries<T> &x) {
  return TaylorSeries<T>::tan(x);
}

template <typename T>
TaylorSeries<T> sqrt(const TaylorSeries<T> &x) {
  return TaylorSeries<T>::sqrt(x);
}

template <typename T, typename S>
TaylorSeries<T> pow(const TaylorSeries<T> &x, S exponent) {
  return TaylorSeries<T>::pow(x, exponent);
}

template <typename T>
std::ostream &operator<<(std::ostream &os,
                         const std::vector<TaylorSeries<T>> &v) {
  os << "[";
  for (int i = 0; i < v.size(); ++i) {
    os << v[i];
    if (i < v.size() - 1) {
      os << ",\n ";
    }
  }
  os << "]";
  return os;
}

template <typename ODESystemType, typename T>

std::vector<TaylorSeries<T>> computeODECoefficients(const std::vector<T> &X,
                                                    const ODESystemType &O,
                                                    int degree) {
  int n = X.size();
  std::vector<TaylorSeries<T>> coefficients;
  coefficients.reserve(X.size());

  // Initialize each TaylorSeries with the given degree
  for (int i = 0; i < X.size(); i++) {
    TaylorSeries<T> ts(X[i], degree + 7);
    ts[0] = X[i];  // Set constant term to initial value
    coefficients.push_back(ts);
  }
  auto Y = coefficients;
  for (int i = 1; i < degree + 1; i++) {
    O(coefficients, Y);
    // std::cout << Y << coefficients << "\n";
    for (int j = 0; j < n; j++) {
      coefficients[j][i] = Y[j][i - 1] / T(i);
    }
  }
  return coefficients;
}
