#pragma once

#include <navtk/aspn.hpp>
#include <navtk/factory.hpp>
#include <navtk/inertial/AidingAltData.hpp>
#include <navtk/inspect.hpp>
#include <navtk/navutils/wgs84.hpp>
#include <navtk/tensors.hpp>

namespace navtk {
namespace inertial {
/**
 * Converts a delta velocity measurement in the inertial sensor frame
 * into a specific force measurement in the NED frame. Applies the
 * correction for a rotating frame using delta rotation measurements:
 *
 * \f$ f^n_{corr, k + 1} = C_b^n \left( \delta v_{k + 1} + \frac{1}{2}\delta\theta_{k + 1}\times
 * \delta v_{k + 1} + \frac{1}{2}\int^{t_{k + 1}}_{t_k} \delta\theta \times \ f^b - \omega^b \times
 * \delta v \,\mathrm{d}t \right)\f$
 *
 * Which is eq 10.64 (11.64 in 2nd ed) of Titterton and Weston, with adjusted notation.
 * The last term in the parentheses is a 'high update rate' term to correct for sculling motion,
 * which is 0 when \f$ f^b \f$ and \f$ \omega^b \f$ are constant over the interval. We make the
 * assumption that this is true for each measurement, and therefore this term is not implemented.
 *
 * Obtaining the above equation from the 'parent' equation
 *
 * \f$ f^n_{corr, k + 1} = C_b^n \left( \int^{t_{k + 1}}_{t_k}\! f^b \,\mathrm{d}t + \int^{t_{k +
 * 1}}_{t_k}\!\delta\theta\times f^b \,\mathrm{d}t\right)\f$
 *
 * has raised questions, so that bit is included here for posterity. Following Savage
 * (Strapdown Analytics, 7.2.2.2-6 through 7.2.2.2.-22), first solve this derivative and re-arrange:
 *
 * \f$ \frac{\delta}{dt}(\delta\theta \times \delta v) = \omega \times \delta v + \delta\theta
 * \times f \f$
 *
 * \f$ \delta\theta \times f = \frac{\delta}{dt}(\delta\theta \times\delta v) - \omega \times \delta
 * v \f$
 *
 * Also,
 *
 * \f$ \delta\theta \times f = \frac{1}{2}\delta\theta \times f + \frac{1}{2} \delta\theta \times f
 * \f$
 *
 * Then substitute the first into the second and get
 *
 * \f$ \delta\theta \times f = \frac{1}{2}\delta\theta \times f + \frac{1}{2}(
 * \frac{\delta}{dt}(\delta\theta \times\delta v) - \omega \times f )\f$
 *
 * which can then be substituted into the integral in the 'parent' equation
 *
 * \f$ \int^{t_{k + 1}}_{t_k}\!\delta\theta \times f \,\mathrm{d}t = \frac{1}{2}\int^{t_{k +
 * 1}}_{t_k}\!\delta\theta \times f\,\mathrm{d}t + \frac{1}{2}\int^{t_{k + 1}}_{t_k}\!(
 * \frac{\delta}{dt}(\delta\theta \times\delta v) - \omega \times f)\,\mathrm{d}t\f$
 *
 * and simplified to
 *
 * \f$ \int^{t_{k + 1}}_{t_k}\!\delta\theta \times f \,\mathrm{d}t = \frac{1}{2}\delta\theta_{k + 1}
 * \times\delta v_{k + 1} + \frac{1}{2}\int^{t_{k + 1}}_{t_k}\!\delta\theta \times f - \omega
 * \times f\,\mathrm{d}t\f$
 *
 * @param C_s_to_n DCM that would rotate dv from the inertial sensor frame
 * to the NED frame.
 * @param dt Delta time over which dv is valid (s).
 * @param dth Vector3 of delta rotation values over the dt period (rad),
 * in the inertial sensor frame.
 * @param dv Vector3 of delta velocity values over the dt period (m/s),
 * in the inertial sensor frame.
 *
 * @return Vector3 of specific force measurements in the navigation
 * frame (m/s^2).
 */
Vector3 calc_force_ned(const Matrix3& C_s_to_n, double dt, const Vector3& dth, const Vector3& dv);

/**
 * Calculates the average specific forces over an interval from the delta velocity and delta time
 * between two sets of PVA.
 *
 * @param pva1 The first PVA, whose timestamp denotes the start of the interval.
 * @param pva2 The second PVA, whose timestamp denotes the end of the interval.
 *
 * @return The average specific forces over the period.
 */
Vector3 calc_force_ned(const aspn_xtensor::MeasurementPositionVelocityAttitude& pva1,
                       const aspn_xtensor::MeasurementPositionVelocityAttitude& pva2);

/**
 * Converts a measured delta rotation vector (sensor with respect to inertial frame) to a rotation
 * rate vector with respect to the navigation (NED) frame by removing the net effects of the earths
 * rotation with respect to the inertial frame \f$ \omega^{n}_{ie} \f$eq and the rotation of the
 * navigation frame with respect to the earth \f$ \omega^{n}_{en} \f$.
 * That is
 *
 * \f$ \omega^{s}_{ns} = \omega^{s}_{is} - C^{s}_{n}[\omega^n_{ie} + \omega^n_{en}] \f$ (eq 3.29)
 *
 * where
 *
 * \f$ \omega^n_{ie} = \begin{bmatrix} \Omega \cos{lat} & 0 & -\Omega\sin{lat} \end{bmatrix} \f$
 * (eq 3.72)
 *
 * and \f$ \omega^n_{en} = \begin{bmatrix} \frac{v_e}{R_e + h} & \frac{-v_n}{R_n + h} &
 * \frac{-v_e\tan{lat}}{R_e + h} \end{bmatrix}\f$ (eq 3.87, with correction on the last term)
 *
 * Equation numbers reference Titterton and Weston, 2nd ed.
 *
 * @param C_s_to_n0 Current inertial sensor-to-nav DCM.
 * @param r_e See return value of navtk::navutils::transverse_radius.
 * @param r_n See return value of navtk::navutils::meridian_radius.
 * @param alt0 The inertial's current ellipsoidal altitude, m.
 * @param cos_l Cosine of inertial latitude.
 * @param dt Delta time that `dth` was measured over.
 * @param dth Vector3 of delta rotations over dt in the inertial sensor
 * frame, \f$ \Theta^{s}_{is} \f$ rad.
 * @param sin_l Sine of inertial latitude.
 * @param tan_l Tangent of inertial latitude.
 * @param v_ned0 Vector3 of inertial velocity at the start of
 * integration in the NED frame, m/s.
 * @param omega Earth rotation rate, rad/s. Defaults to navtk::navutils::ROTATION_RATE.
 *
 * @return Inertial sensor rotation rate with respect to the navigation frame coordinatized in the
 * sensor frame, \f$ \omega^{s}_{ns}\f$.
 */
Vector3 calc_rot_rate(const Matrix3& C_s_to_n0,
                      double r_e,
                      double r_n,
                      double alt0,
                      double cos_l,
                      double dt,
                      const Vector3& dth,
                      double sin_l,
                      double tan_l,
                      const Vector3& v_ned0,
                      double omega = navutils::ROTATION_RATE);

/**
 * Converts a measured delta rotation vector (sensor with respect to inertial frame) to a rotation
 * rate vector with respect to the navigation (NED) frame by removing the net effects of the earths
 * rotation with respect to the inertial frame \f$ \omega^{n}_{ie} \f$ and the rotation of the
 * navigation frame with respect to the earth \f$ \omega^{n}_{en} \f$. Uses constants defined by
 * WGS84.hpp; for finer control over earth model parameters use overload.
 *
 * @param pva Position, velocity and attitude at beginning of inertial measurement period.
 * @param dt Delta time that `dth` was measured over.
 * @param dth Vector3 of delta rotations over dt in the inertial sensor
 * frame, \f$ \Theta^{s}_{is} \f$ rad.
 *
 * @return Inertial sensor rotation rate with respect to the navigation frame coordinatized in the
 * sensor frame, \f$ \omega^{s}_{ns}\f$.
 */
Vector3 calc_rot_rate(const aspn_xtensor::MeasurementPositionVelocityAttitude& pva,
                      double dt,
                      const Vector3& dth);


// TODO: #690 Implement a more robust alternative which doesn't make a small angle assumption.
/**
 * Calculates the average rotation rates over an interval from the change in attitude and delta time
 * between two sets of PVA.
 *
 * Note that since the attitude contained in MeasurementPositionVelocityAttitude is stored as a DCM,
 * the largest observable difference between two attitudes is 2PI. Any rotation rates larger than
 * 2PI will be unobservable. Additionally, this implementation assumes that rotation rates are small
 * (like those experienced by an inertial system sampling several times per second). If the
 * difference between the two attitudes is large then the calculated rate will have significant
 * inaccuracies.
 *
 * @param pva1 The first PVA, whose timestamp denotes the start of the interval.
 * @param pva2 The second PVA, whose timestamp denotes the end of the interval.
 *
 * @return The average rotation rates over the period.
 */
Vector3 calc_rot_rate(const aspn_xtensor::MeasurementPositionVelocityAttitude& pva1,
                      const aspn_xtensor::MeasurementPositionVelocityAttitude& pva2);

/**
 * Helps mitigate vertical drift by correcting down acceleration/velocity prior to integration.
 * The correction is based upon the difference between the inertial altitude and the provided
 * aiding altitude. Note that \p accel_vector is not returned but is modified in place, and will
 * likely have changed after calling this function.
 *
 * If using altitude aiding, ensure the INS is not reset, as that is incompatible with the aiding
 * algorithm and will produce invalid results. The aiding algorithm is a third-order control loop,
 * taking into account the difference between the INS altitude and that of an external sensor, and
 * adjusting the acceleration, velocity, and position of the INS altitude solution. Note that the
 * algorithm applies its corrections to accelerations with the effects of gravity already removed.
 * Applying our altitude aiding will require conversion from dv's and dth's to accelerations with
 * the corresponding dt.
 *
 * The altitude aiding algorithm is taken from \f$ \textit{Applied Mathematics in Integrated
 * Navigation Systems, Third Edition} \f$, by Robert M. Rogers (p 118 - p 120). Vertical channel
 * dynamics are given by the following equations:
 *
 * \f$ \dot{h} = v_z^n - C_1 (h-h_B) \f$     (eq 5.70)
 *
 * \f$ \dot{v}_z^n = f_z + g_z + (\rho+2\Omega)_x v_y^n - (\rho+2\Omega)_y x_y^n - C_2(h-h_B)
 * - C_3 \int (h-h_B)dt \f$     (eq 5.71)
 *
 * with the coefficients defined as:
 *
 * \f$ C_1 = 3 \lambda \f$
 * \f$ C_2 = 4 \lambda^2 + \frac{2g}{R} \f$
 * \f$ C_3 = 2 \lambda^3 \f$
 * (eq 5.83)
 *
 * When choosing a value for \f$ \lambda \f$, it is imperative to ensure it is sufficiently small
 * enough for the dynamics of the tracking loop.
 *
 * @param r_zero Ellipsoidal Earth radius, meters.
 * @param accel_vector 3x1 vector of acceleration in the NED frame, m/s^2.
 * @param aiding_alt_data Container holding aiding altitude, error, and method.
 * @param alt0 Current inertial altitude, m HAE.
 * @param dt Delta time that \p accel_vector is to be integrated over, sec.
 * @param g Gravity estimate, m/s^2.
 *
 * @return A scaling term, C1, to be used in correcting the altitude.
 *
 */
double apply_aiding_alt_accel(double r_zero,
                              Vector3* accel_vector,
                              AidingAltData* aiding_alt_data,
                              double alt0,
                              double dt,
                              const Vector3& g);

/**
 * Calculates an offset that can be used to convert between NED forces and accelerations.
 *
 * @tparam S1 Type of r_e.
 * @tparam S2 Type of r_n.
 * @tparam S3 Type of alt0.
 * @tparam S4 Type of cos_l.
 * @tparam S5 Type of g.
 * @tparam S6 Type of sec_l.
 * @tparam S7 Type of sin_l.
 * @tparam S8 Type of v_ned0.
 * @tparam std::enable_if_t<> Constrains S1, S2, S3, S4, S6, and S7 to be 0 dimensional and S5 and
 * S8 to be 1 dimensional.
 *
 * @param r_e east_distances.
 * @param r_n north_distances.
 * @param alt0 The inertial's current ellipsoidal altitude, m.
 * @param cos_l Cosine of inertial latitude.
 * @param g Estimated local gravity along North, East and Down axes, m/s^2.
 * @param sec_l Secant of inertial latitude.
 * @param sin_l Sine of inertial latitude.
 * @param v_ned0 Vector3 of inertial velocity in the NED frame, m/s.
 * @param omega Earth rotation rate, rad/s.
 *
 * @return The offset between NED forces and accelerations such that `forces + offset =
 * accelerations`.
 */
template <
    typename S1,
    typename S2,
    typename S3,
    typename S4,
    typename S5,
    typename S6,
    typename S7,
    typename S8,
    IfAllConditions<TensorsAreDim<0, S1, S2, S3, S4, S6, S7>, TensorsAreDim<1, S5, S8>>* = nullptr>
Vector3 calc_force_and_acceleration_offset(const S1& r_e,
                                           const S2& r_n,
                                           const S3& alt0,
                                           const S4& cos_l,
                                           const S5& g,
                                           const S6& sec_l,
                                           const S7& sin_l,
                                           const S8& v_ned0,
                                           double omega = navutils::ROTATION_RATE) {
	auto v_ned0_vec = to_vec(v_ned0);
	auto g_vec      = to_vec(g);
	auto l_dot      = v_ned0_vec(0) / (r_n + alt0);
	auto lambda_dot = v_ned0_vec(1) * sec_l / (r_e + alt0);

	return {-v_ned0_vec(1) * (2 * omega + lambda_dot) * sin_l + v_ned0_vec(2) * l_dot + g_vec(0),
	        // Equation 3.77, pp 52, Titterton text
	        v_ned0_vec(0) * (2 * omega + lambda_dot) * sin_l +
	            v_ned0_vec(2) * (2 * omega + lambda_dot) * cos_l + g_vec(1),
	        // Equation 3.78, pp 52, Titterton text
	        -v_ned0_vec(1) * (2 * omega + lambda_dot) * cos_l - v_ned0_vec(0) * l_dot + g_vec(2)};
}

/**
 * Batched version of `calc_force_and_acceleration_offset`.
 *
 * @overload
 *
 * @see calc_force_and_acceleration_offset
 *
 * @tparam B1 Type of r_e.
 * @tparam B2 Type of r_n.
 * @tparam B3 Type of alt0.
 * @tparam B4 Type of cos_l.
 * @tparam B5 Type of g.
 * @tparam B6 Type of sec_l.
 * @tparam B7 Type of sin_l.
 * @tparam B8 Type of v_ned0.
 * @tparam std::enable_if_t<> Constrains the max dimension of B1, B2, B3, B4, B6, and B7 to be 1
 * _or_ the max dimension of B5 and B8 to be 2.
 *
 * @param r_e east_distances, shape (N).  Can accept initializer lists or a
 * `double` as well.
 * @param r_n north_distances, shape (N).  Can accept initializer lists or a
 * `double` as well.
 * @param alt0 The inertial's current ellipsoidal altitudes, m, shape (N).  Can accept initializer
 * lists or a `double` as well.
 * @param cos_l Cosines of inertial latitudes, shape (N).  Can accept initializer lists or a
 * `double` as well.
 * @param g Estimated local gravities along North, East and Down axes, m/s^2, shape (N, 3).  Can
 * accept initializer lists or a `Vector` as well.
 * @param sec_l Secants of inertial latitudes, shape (N).  Can accept initializer lists or a
 * `double` as well.
 * @param sin_l Sines of inertial latitudes, shape (N).  Can accept initializer lists or a
 * `double` as well.
 * @param v_ned0 Inertial velocities in the NED frame, m/s, shape (N, 3).  Can accept initializer
 * lists or a `Vector` as well.
 * @param omega Earth rotation rate, rad/s.
 *
 * @return The offsets between NED forces and accelerations such that `forces + offset =
 * accelerations`, shape (N, 3)
 */
template <typename B1                                    = Vector,
          typename B2                                    = Vector,
          typename B3                                    = Vector,
          typename B4                                    = Vector,
          typename B5                                    = Matrix,
          typename B6                                    = Vector,
          typename B7                                    = Vector,
          typename B8                                    = Matrix,
          IfAnyConditions<TensorsHaveMaxDim<1, B1, B2, B3, B4, B6, B7>,
                          TensorsHaveMaxDim<2, B5, B8>>* = nullptr>
Matrix calc_force_and_acceleration_offset(const B1& r_e,
                                          const B2& r_n,
                                          const B3& alt0,
                                          const B4& cos_l,
                                          const B5& g,
                                          const B6& sec_l,
                                          const B7& sin_l,
                                          const B8& v_ned0,
                                          double omega = navutils::ROTATION_RATE) {
	auto r_e_vec    = to_vec(r_e);
	auto r_n_vec    = to_vec(r_n);
	auto alt0_vec   = to_vec(alt0);
	auto cos_l_vec  = to_vec(cos_l);
	auto g_mat      = to_matrix(g);
	auto sec_l_vec  = to_vec(sec_l);
	auto sin_l_vec  = to_vec(sin_l);
	auto v_ned0_mat = to_matrix(v_ned0);

	const size_t N = std::max({r_e_vec.shape()[0],
	                           r_n_vec.shape()[0],
	                           alt0_vec.shape()[0],
	                           cos_l_vec.shape()[0],
	                           g_mat.shape()[0],
	                           sec_l_vec.shape()[0],
	                           sin_l_vec.shape()[0],
	                           v_ned0_mat.shape()[0]});

	Matrix offset = empty(N, 3);

	Scalar l_dot, lambda_dot;

	for (size_t i = 0; i < N; i++) {
		l_dot      = v_ned0_mat(i, 0) / (r_n_vec(i) + alt0_vec(i));
		lambda_dot = v_ned0_mat(i, 1) * sec_l_vec(i) / (r_e_vec(i) + alt0_vec(i));

		offset(i, 0) = -v_ned0_mat(i, 1) * (2 * omega + lambda_dot) * sin_l_vec(i) +
		               v_ned0_mat(i, 2) * l_dot + g_mat(i, 0);
		// Equation 3.77, pp 52, Titterton text
		offset(i, 1) = v_ned0_mat(i, 0) * (2 * omega + lambda_dot) * sin_l_vec(i) +
		               v_ned0_mat(i, 2) * (2 * omega + lambda_dot) * cos_l_vec(i) + g_mat(i, 1);
		// Equation 3.78, pp 52, Titterton text
		offset(i, 2) = -v_ned0_mat(i, 1) * (2 * omega + lambda_dot) * cos_l_vec(i) -
		               v_ned0_mat(i, 0) * l_dot + g_mat(i, 2);
	}

	return offset;
}

}  // namespace inertial
}  // namespace navtk
