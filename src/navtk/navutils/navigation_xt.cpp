#include <navtk/navutils/navigation.hpp>

#include <algorithm>
#include <cmath>

#include <navtk/errors.hpp>
#include <navtk/inspect.hpp>
#include <navtk/linear_algebra.hpp>
#include <navtk/navutils/math.hpp>
#include <navtk/navutils/wgs84.hpp>
#include <navtk/utils/ValidationContext.hpp>

using navtk::utils::ValidationContext;
using xt::transpose;
using xt::view;
using xt::xrange;

namespace navtk {
namespace navutils {

Matrix3 axis_angle_to_dcm(const Vector3& axis, double angle) {
	Vector ext = axis;
	auto n_ext = navtk::norm(ext);
	if (n_ext != 1) ext = (axis / n_ext);
	double x       = ext[0];
	double y       = ext[1];
	double z       = ext[2];
	double cos_ang = cos(angle);
	double sin_ang = sin(angle);

	return {{cos_ang + pow(x, 2) * (1 - cos_ang),
	         x * y * (1 - cos_ang) - z * sin_ang,
	         x * z * (1 - cos_ang) + y * sin_ang},
	        {y * x * (1 - cos_ang) + z * sin_ang,
	         cos_ang + pow(y, 2) * (1 - cos_ang),
	         y * z * (1 - cos_ang) - x * sin_ang},
	        {z * x * (1 - cos_ang) - y * sin_ang,
	         z * y * (1 - cos_ang) + x * sin_ang,
	         cos_ang + pow(z, 2) * (1 - cos_ang)}};
}

/*
 * A simplified version of the van_loan algorithm for 1 element matrices.
 *
 * The standard versions of calc_van_loan depend upon the matrix
 * B = expm(Matrix{{-Fdt, GQG^{T}dt}, {0, F^{T}dt}})
 *
 * Where Qd is calculated from the product of the transpose of the lower right and upper right
 * blocks
 *
 * Qd = B(n..2n, n..2n)^T  B(0..n, n..2n)
 *
 * The generic matrix exponential for a 2x2 matrix (which B is in this case) has a
 * solution (from https://mathworld.wolfram.com/MatrixExponential.html)
 *
 * A = Matrix{{a, b}, {c, d}}
 * K = sqrt((a - d)^2 + 4bc)
 * exp(A) = 1/K * Matrix{{m11, m12}, {m21, m22}}
 *
 * with
 *
 * m11 = exp((a + d)/2)) * (K * cosh(K/2) + (a - d) * sinh(K/2))
 * m12 = 2 * b * exp((a + d)/2) * sinh(K/2)
 * m21 = 2 * c * exp((a + d)/2) * sinh(K/2)
 * m22 = exp((a + d)/2)) * (K * cosh(K/2) + (d - a) * sinh(K/2))
 *
 * However, as we only need m12 and m22 of the result, our c element is 0, and a = -d, the
 * calculations can be simplified. exp((a + d)/2) simplifies to 1, and K = 2d = d - a...
 */
Matrix calc_van_loan(const double f, const double g, const double q, const double dt) {
	if (f == 0) return Matrix{{q * dt}};
	auto b   = g * g * q * dt;
	auto k   = 2 * f * dt;
	auto snh = std::sinh(0.5 * k);
	// k == d-a and factors out, and cosh(a) + sinh(a) = e^a
	return Matrix{{2 * b * snh * std::exp(0.5 * k) / k}};
}

Matrix calc_van_loan(const Matrix& F, const Matrix& G, const Matrix& Q, const double dt) {
	auto NS = num_rows(F);
	if (NS == 1) {
		Matrix g_eval = dot(dot(G, Q), transpose(G));
		return calc_van_loan(F(0, 0), 1, g_eval(0, 0), dt);
	}

	Matrix a_star                                    = zeros(2 * NS, 2 * NS);
	view(a_star, xt::range(_, NS), xt::range(_, NS)) = F * -dt;
	if (is_identity(G)) {
		view(a_star, xt::range(_, NS), xt::range(NS, _)) = Q * dt;
	} else {
		view(a_star, xt::range(_, NS), xt::range(NS, _)) = dot(dot(G, Q), transpose(G)) * dt;
	}
	view(a_star, xt::range(NS, _), xt::range(NS, _)) = transpose(F) * dt;
	auto b_star                                      = expm(a_star);
	Matrix B22T(transpose(view(b_star, xt::range(NS, _), xt::range(NS, _))));
	Matrix B12(view(b_star, xt::range(_, NS), xt::range(NS, _)));
	return dot(B22T, B12);
}

Matrix3 correct_dcm_with_tilt(const Matrix3& dcm, const Vector3& tilt) {
	double sum_squares = pow(tilt[0], 2) + pow(tilt[1], 2) + pow(tilt[2], 2);
	if (sum_squares > 0) {
		double m  = sqrt(sum_squares);
		auto I    = eye(3);
		Matrix3 s = skew(tilt);
		// Below is equivalent to I - sin(m) * s / m + dot((1 - cos(m)) * s, s) / pow(m, 2);
		Matrix3 B = xt::fma(
		    (1 - cos(m)) / sum_squares, xt::linalg::matrix_power(s, 2), xt::fma(-sin(m) / m, s, I));
		return dot(B, dcm);
	}
	return dcm;
}

namespace {
void validate_discretized_inputs(const Matrix& f, const Matrix& q, double dt) {

	if (ValidationContext validation{}) {
		validation.add_matrix(f, "f").dim('N', 'N').add_matrix(q, "q").dim('N', 'N').validate();
		if (dt < 0.0) {
			log_or_throw<std::invalid_argument>("`dt` should be >= 0; is {}", dt);
		}
	}
}
}  // namespace

std::pair<Matrix, Matrix> discretize_first_order(const Matrix& f, const Matrix& q, double dt) {
	validate_discretized_inputs(f, q, dt);
	return {xt::fma(f, dt, eye(num_cols(f))), q * dt};
}

std::pair<Matrix, Matrix> discretize_second_order(const Matrix& f, const Matrix& q, double dt) {
	validate_discretized_inputs(f, q, dt);
	Matrix Phi    = xt::fma(0.5 * dt * dt, matrix_power(f, 2), xt::fma(f, dt, eye(num_cols(f))));
	Matrix q_prop = dot(dot(Phi, q), transpose(Phi));
	Matrix Qd     = (q_prop + q) * (0.5 * dt);
	return {std::move(Phi), std::move(Qd)};
}

std::pair<Matrix, Matrix> discretize_van_loan(const Matrix& f, const Matrix& q, double dt) {
	validate_discretized_inputs(f, q, dt);
	return {expm(f * dt), calc_van_loan(f, eye(num_cols(f)), q, dt)};
}

Matrix3 ecef_to_cen(const Vector3& Pe) { return llh_to_cen(ecef_to_llh(Pe)); }

Vector3 ecef_to_llh(const Vector3& Pe) {
	// WGS-84 Constants
	auto a                   = SEMI_MAJOR_RADIUS;     // Semi-major radius (m)
	auto e2                  = ECCENTRICITY_SQUARED;  // Eccentricity squared (.)
	double pm0               = sqrt(pow(Pe[0], 2) + pow(Pe[1], 2));
	double pm1               = Pe[2];
	double phi0              = atan2(pm1, pm0);
	double h0                = 0;
	double dp0               = a;
	double dp1               = a;
	int count                = 0;
	const int max_iterations = 5;

	while ((std::abs(dp0) > 7e-6 || std::abs(dp1) > 1e-6) && count <= max_iterations) {
		double slat  = sin(phi0);
		double clat  = cos(phi0);
		double s2lat = slat * slat;
		double Nden  = 1 - e2 * s2lat;
		double N     = a / sqrt(Nden);

		// Calculate residual by subtracting initial position in meridianal plane (meters)
		dp0 = pm0 - (N + h0) * clat;
		dp1 = pm1 - (N * (1 - e2) + h0) * slat;
		// Calculate inverse Jacobian (transformation from residual to lat and alt)
		double k1 = 1 - e2 * s2lat;
		double k2 = sqrt(k1);

		double A11  = slat * (e2 * a * clat * clat / k1 / k2 - a / k2 - h0);
		double A12  = clat;
		double A21  = clat * (a * (1 - e2) / k2 + h0 + a * e2 * (1 - e2) * s2lat / k1 / k2);
		double A22  = slat;
		double Adet = A11 * A22 - A21 * A12;
		double dHa  = (A22 * dp0 - A12 * dp1) / Adet;
		double dHb  = (-A21 * dp0 + A11 * dp1) / Adet;

		phi0 += dHa;
		h0 += dHb;

		++count;
	}

	double lam = atan2(Pe[1], Pe[0]);
	return {phi0, lam, h0};
}

Vector3 ecef_to_local_level(const Vector3& P0e, const Vector3& Pe) {
	return dot(transpose(ecef_to_cen(P0e)), Pe - P0e);
}

Matrix3 llh_to_cen(const Vector3& Pwgs) {
	double clat = cos(Pwgs[0]);
	double slat = sin(Pwgs[0]);
	double clon = cos(Pwgs[1]);
	double slon = sin(Pwgs[1]);
	return Matrix{
	    {-slat * clon, -slon, -clat * clon}, {-slat * slon, clon, -clat * slon}, {clat, 0, -slat}};
}

Vector3 llh_to_ecef(const Vector3& Pwgs) {
	// WGS-84 Constants
	auto a  = SEMI_MAJOR_RADIUS;     // Semi-major radius (m)
	auto e2 = ECCENTRICITY_SQUARED;  // Eccentricity squared (.)

	auto lam    = Pwgs[1];
	auto phi    = Pwgs[0];
	auto h      = Pwgs[2];
	auto cosphi = cos(phi);
	auto sinphi = sin(phi);

	auto N = a / sqrt(1 - e2 * pow(sinphi, 2));

	return {(N + h) * cosphi * cos(lam), (N + h) * cosphi * sin(lam), (N * (1 - e2) + h) * sinphi};
}

Vector3 local_level_to_ecef(const Vector3& P0e, const Vector3& Pn) {
	return dot(ecef_to_cen(P0e), Pn) + P0e;
}

}  // namespace navutils
}  // namespace navtk
