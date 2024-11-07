#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <string>

#include "HC-MC.hpp"

// Helper function to print vectors nicely
void print_vector(const ADVector& v, const std::string& name) {
  std::cout << name << " = [";
  for (int i = 0; i < v.size(); ++i) {
    std::cout << std::setprecision(15) << CppAD::Value(v(i));
    if (i < v.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << "\n";
}

TEST_CASE("Circle-Line Intersection") {
  auto F = [](const ADVector& X) {
    ADVector result(2);
    result(0) = X(0) * X(0) + X(1) * X(1) - 4.0;  // Circle of radius 2
    result(1) = X(0) + X(1) - 1.0;                // Line x + y = 1
    return result;
  };

  SolveOptions options;
  options.debug = true;
  options.tolerance = 1e-10;

  auto result = solve(F, options);
  REQUIRE(result.has_value());

  const auto& solve_result = *result;
  REQUIRE(solve_result.success);
  REQUIRE(solve_result.residualNorm < 1e-8);

  // Verify the solution satisfies the system
  const auto& X = solve_result.solution;
  auto F_X = F(X);
  print_vector(X, "Solution");
  print_vector(F_X, "F(solution)");

  // Check both equations are satisfied
  REQUIRE(std::abs(CppAD::Value(F_X(0))) < 1e-8);  // Circle equation
  REQUIRE(std::abs(CppAD::Value(F_X(1))) < 1e-8);  // Line equation

  // Verify solution matches one of the two expected points
  double x = CppAD::Value(X(0));
  double y = CppAD::Value(X(1));
  bool matches_solution =
      (std::abs(x + y - 1.0) < 1e-8) &&        // Line equation
      (std::abs(x * x + y * y - 4.0) < 1e-8);  // Circle equation

  REQUIRE(matches_solution);
}

TEST_CASE("Parabola-Line Intersection") {
  auto F = [](const ADVector& X) {
    ADVector result(2);
    result(0) = X(0) * X(0) - X(1);  // Parabola y = x^2
    result(1) = X(1) - 4.0;          // Line y = 4
    return result;
  };

  SolveOptions options;
  options.debug = false;

  auto result = solve(F);
  REQUIRE(result.has_value());

  const auto& solve_result = *result;
  REQUIRE(solve_result.success);
  REQUIRE(solve_result.residualNorm < 1e-8);

  // Verify solution satisfies both equations
  const auto& X = solve_result.solution;
  double x = CppAD::Value(X(0));
  double y = CppAD::Value(X(1));

  REQUIRE(std::abs(x * x - y) < 1e-8);  // Parabola equation
  REQUIRE(std::abs(y - 4.0) < 1e-8);    // Line equation

  // The solution should be either (2,4) or (-2,4)
  bool is_valid_solution =
      (std::abs(std::abs(x) - 2.0) < 1e-8) && (std::abs(y - 4.0) < 1e-8);
  REQUIRE(is_valid_solution);
}

TEST_CASE("3D Sphere-Plane-Line Intersection") {
  auto F = [](const ADVector& X) {
    ADVector result(3);
    result(0) =
        X(0) * X(0) + X(1) * X(1) + X(2) * X(2) - 4.0;  // Sphere radius 2
    result(1) = X(0) + X(1) + X(2) - 1.0;               // Plane x+y+z=1
    result(2) = X(0) - X(1);                            // Line x=y
    return result;
  };

  SolveOptions options;
  options.debug = false;
  options.tolerance = 1e-10;

  auto result = solve(F);
  REQUIRE(result.has_value());

  const auto& solve_result = *result;
  REQUIRE(solve_result.success);
  REQUIRE(solve_result.residualNorm < 1e-8);

  // Verify solution satisfies all equations
  const auto& X = solve_result.solution;
  const double x = CppAD::Value(X(0));
  const double y = CppAD::Value(X(1));
  const double z = CppAD::Value(X(2));

  REQUIRE(std::abs(x * x + y * y + z * z - 4.0) < 1e-8);  // Sphere
  REQUIRE(std::abs(x + y + z - 1.0) < 1e-8);              // Plane
  REQUIRE(std::abs(x - y) < 1e-8);                        // Line
}

TEST_CASE("System with Transcendental Functions") {
  auto F = [](const ADVector& X) {
    ADVector result(2);
    result(0) = CppAD::cos(X(0)) - X(1);  // cos(x) = y
    result(1) = X(0) * X(1) - 0.5;        // xy = 0.5
    return result;
  };

  SolveOptions options;
  options.debug = false;
  options.verifyDerivatives = true;  // Test derivative computation

  auto result = solve(F);
  REQUIRE(result.has_value());

  const auto& solve_result = *result;
  REQUIRE(solve_result.success);
  REQUIRE(solve_result.residualNorm < 1e-8);

  // Verify solution satisfies both equations
  const auto& X = solve_result.solution;
  const double x = CppAD::Value(X(0));
  const double y = CppAD::Value(X(1));

  REQUIRE(std::abs(std::cos(x) - y) < 1e-8);
  REQUIRE(std::abs(x * y - 0.5) < 1e-8);
}

TEST_CASE("Higher Dimensional System") {
  auto F = [](const ADVector& X) {
    ADVector result(4);
    result(0) = X(0) * X(0) * X(0) + CppAD::sin(X(1)) - 1.0;
    result(1) = X(0) * X(0) + X(1) * X(1) + X(2) * X(2) + X(3) * X(3) - 5.0;
    result(2) = X(0) + X(1) + X(2) + CppAD::pow(X(3), 5) + 2.5;
    result(3) = CppAD::sin(X(0)) + CppAD::exp(X(1)) +
                CppAD::cos(X(1)) * X(2) * X(2) * X(2) + 2.0;
    return result;
  };

  SolveOptions options;
  options.debug = true;
  options.maxIterations = 200;  // This system might need more iterations

  auto result = solve(F);
  REQUIRE(result.has_value());

  const auto& solve_result = *result;
  REQUIRE(solve_result.success);
  REQUIRE(solve_result.residualNorm < 1e-8);

  // Verify each equation is satisfied
  const auto& X = solve_result.solution;
  auto F_X = F(X);
  for (int i = 0; i < F_X.size(); ++i) {
    REQUIRE(std::abs(CppAD::Value(F_X(i))) < 1e-8);
  }
}

int unused_main(int argc, char** argv) {
  doctest::Context context;
  context.applyCommandLine(argc, argv);

  const int result = context.run();

  if (context.shouldExit()) {
    return result;
  }

  return result;
}