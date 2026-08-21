#pragma once

#include <memory>
#include <utility>

#include <navtk/factory.hpp>
#include <navtk/inspect.hpp>
#include <navtk/navutils/math.hpp>
#include <navtk/navutils/wgs84.hpp>
#include <navtk/tensors.hpp>

namespace navtk {
/**
 * NavToolkit namespace for navigation utilities.
 */
namespace navutils {

/**
 * Calculate the meridian radius of curvature (termed \f$R_N\f$ in Titterton and Weston, sometimes
 * called \f$R_M\f$, where "M" stands for meridian). This is the radius of curvature in the
 * north-south direction (i.e., along a meridian), which varies according to latitude. The radius
 * of curvature can be used to relate change in latitude (\f$d\phi\f$) to change in the north
 * direction in a local-level frame expressed in meters (\f$d\text{North}\f$):
 *
 * \f$  d\text{North} = R_N d\phi \f$
 *
 * References:
 *
 * Strapdown Inertial Navigation Technology 2nd ed, Titterton and Weston, eq. 3.83
 *
 * [WGS-84 Reference System (NIMA report TR 8350.2)](https://nga-rescue.is4s.us/wgs84fin.pdf), page
 * 7-5.
 *
 * @see transverse_radius()
 *
 * @param latitude Latitude (radians)
 *
 * @return Radius of curvature in north/south direction (along a meridian) (meters).
 *
 */
template <typename S, IfTensorOfDim<S, 0>* = nullptr>
double meridian_radius(const S& latitude) {
	auto sin_lat = sin(latitude);
	return SEMI_MAJOR_RADIUS * (1 - ECCENTRICITY_SQUARED) /
	       pow(1 - ECCENTRICITY_SQUARED * sin_lat * sin_lat, 1.5);
}

/**
 * Batched version of `meridian_radius`.
 *
 * @overload
 *
 * @see meridian_radius
 *
 * @param latitude Latitudes (radians)
 *
 * @return Radii of curvature in north/south direction (along a meridian) (meters).
 */
template <typename B, IfTensorOfDim<B, 1>* = nullptr>
Vector meridian_radius(const B& latitude) {
	auto sin_lat = xt::sin(latitude);
	return SEMI_MAJOR_RADIUS * (1 - ECCENTRICITY_SQUARED) /
	       xt::pow(1 - ECCENTRICITY_SQUARED * sin_lat * sin_lat, 1.5);
}

/**
 * Calculate the transverse radius of curvature (termed \f$R_E\f$ in Titterton and Weston, sometimes
 * called \f$R_N\f$). This is the radius of curvature in the east-west direction (i.e., normal to a
 * meridian), which varies according to latitude. The radius of curvature can be used to relate
 * change in longitude (\f$d\lambda\f$) to change in the east direction in a local-level frame
 * expressed in meters (\f$d\text{East}\f$):
 *
 * \f$  d\text{East} = R_N \cos\phi d\lambda \f$
 *
 * where \f$\phi\f$ is the latitude.
 *
 * @see meridian_radius()
 *
 * @param latitude WGS-84 latitude (radians)
 * @return Radius of curvature in north/south direction (along a meridian) (meters).
 *
 * REFERENCES:
 *
 * Strapdown Inertial Navigation Technology 2nd ed, Titterton and Weston, eq. 3.84
 *
 * [WGS-84 Reference System (NIMA report TR 8350.2)](https://nga-rescue.is4s.us/wgs84fin.pdf), page
 * 7-5.
 *
 */
template <typename S, IfTensorOfDim<S, 0>* = nullptr>
double transverse_radius(const S& latitude) {
	auto sin_lat = sin(latitude);
	return SEMI_MAJOR_RADIUS / pow(1 - ECCENTRICITY_SQUARED * sin_lat * sin_lat, 0.5);
}

/**
 * Batched version of `transverse_radius`.
 *
 * @overload
 *
 * @see transverse_radius
 *
 * @param latitude WGS-84 latitudes (radians)
 * @return Radii of curvature in north/south direction (along a meridian) (meters).
 */
template <typename B, IfTensorOfDim<B, 1>* = nullptr>
Vector transverse_radius(const B& latitude) {
	auto sin_lat = xt::sin(latitude);
	return SEMI_MAJOR_RADIUS / xt::pow(1 - ECCENTRICITY_SQUARED * sin_lat * sin_lat, 0.5);
}

/**
 * Converts an axis-angle rotation to the equivalent DCM. If coordinate frame A is rotated
 * by `angle` about `axis` to form coordinate frame B, then the DCM that is returned
 * is the DCM that rotates a vector from frame B to frame A.
 *
 * For example, if an NED frame (frame A) is
 * rotated by 30 degrees about the down axis to form a platform frame (frame B) (i.e., yaw = 30
 * degrees, pitch = 0, roll = 0), then `axis = [0, 0, 1]`, `angle = 30 * PI / 180`, and the DCM that
 * is output will rotate an arbitrary vector \f$\textbf{w}\f$ from frame B (platform frame) to frame
 * A (NED):
 *
 * \f$ \textbf{w}^\text{NED} = \textbf{C}_\text{P}^\text{NED} \textbf{w}^\text{P} \f$
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for more details on axis-angle
 * and DCM expressions of attitude.
 *
 * @param axis The axis about which to rotate (unit vector)
 * @param angle The amount of right-hand rotation about the axis (radians)
 *
 * @return The equivalent direction cosine matrix (DCM).
 */
Matrix3 axis_angle_to_dcm(const Vector3& axis, double angle);

/**
 * Calculates the discrete-time system noise covariance matrix from a continuous-time system
 * description. First, define a continuous-time system as
 *
 * \f$ \dot{\textbf{x}} = \textbf{Fx} + \textbf{Gw} \f$
 *
 *
 * \f$ E[\textbf{w}(t-\tau)\textbf{w}^T(t-\tau)] = \textbf{Q}\delta(\tau) \f$
 *
 * where \f$ E[] \f$ is the expectation operator and \f$ \delta(\tau) \f$ is the Dirac delta
 * function.
 *
 * This can be converted to an equivalent discrete-time system
 *
 * \f$ \textbf{x}_{t_{k+1}} = \pmb{\Phi}\textbf{x}_{t_k} + \textbf{w}_d \f$
 *
 * where
 *
 * \f$ \pmb{\Phi}=e^{\textbf{F}\Delta t} \f$
 *
 * \f$ \Delta t = t_{k+1} - t_k \f$
 *
 * \f$ E[\textbf{w}_{t_j}\textbf{w}^T_{t_k}] = \textbf{Q}_d \delta_{kj} \f$
 *
 * and \f$ \delta_{kj} \f$ is the Kronecker delta function.
 *
 * @param F The continuous-time dynamics matrix \f$\textbf{F}\f$, NxN.
 * @param G The noise mapping matrix \f$ \textbf{G} \f$, NxM.
 * @param Q The continuous time noise covariance matrix \f$ \textbf{Q} \f$, MxM.
 * @param dt The discretization time interval \f$ \Delta t \f$
 *
 * @return The discrete-time system noise covariance matrix \f$ \textbf{Q}_d \f$.
 */
Matrix calc_van_loan(const Matrix& F, const Matrix& G, const Matrix& Q, double dt);

/**
 *
 * Corrects a DCM by applying tilt error corrections (usually generated via the Pinson Error
 * model).
 *
 * Consider a `dcm` that rotates a vector from frame B to frame A
 * (\f$\textbf{C}_\text{B}^\text{A}\f$). Next, consider `tilt` to be a rotation vector. When
 * coordinate frame A is rotated by the rotation vector defined by `tilt` it becomes coordinate
 * frame \f$\text{A}'\f$.
 *
 * The DCM that is returned by this function would then be
 *
 * \f$\textbf{C}_\text{B}^{\text{A}'} = \textbf{C}_\text{A}^{\text{A}'}
 * \textbf{C}_\text{B}^\text{A}\f$.
 *
 * For example, consider the case where the attitude is expressed as a `dcm` that rotates from a
 * platform frame \f$\text{P}\f$ to the \f$\text{NED}\f$ frame \f$\textbf{C}_\text{P}^\text{NED}\f$.
 * However, this attitude is not quite correct, and when applied doesn't rotate a vector from the
 * platform frame to the true \f$\text{NED}\f$ frame (\f$\text{NED}_{true}\f$). The `tilt`
 * corrections represent the relationship between the assumed \f$\text{NED}\f$ frame
 * (\f$\text{NED}\f$) and the true \f$\text{NED}\f$ frame (\f$\text{NED}_{true}\f$). If the
 * \f$\text{NED}\f$ coordinate frame is rotated by the rotation vector described by `tilt`, the
 * result is the  \f$\text{NED}_{true}\f$ coordinate frame. In this example, the function will
 * return the DCM corresponding to \f$\textbf{C}_\text{P}^{\text{NED}_{true}}\f$
 *
 * Note that this function is equivalent to (in pseudocode):
 *
 * `angle = magnitude(tilt)`
 *
 * `axis = tilt / angle` (axis vector is normalized rotation vector)
 *
 * `corrected_dcm = transpose(axis_angle_to_dcm(axis, angle)) * dcm`
 *
 * WARNING: The definition of the tilt rotation vector given here is opposite of that used in
 * other functions (correct_quat_with_tilt(), for example), where the tilt rotation does a frame
 * rotation from \f$\text{A}'\f$ to \f$\text{A}\f$. In other words, a sign change on the tilts is
 * required between these functions. However, the definition of tilts here is consistent with those
 * typically estimated by our navtk::filtering::Pinson15NedBlock model using the additive error
 * state formulation. In other words, when one has an estimated
 * \f$\textbf{C}^{\text{NED}}_\text{P}\f$ matrix and is using the Pinson model to estimate error
 * states, the following applications of tilts are all equivalent and will produce
 * \f$\textbf{C}^{\text{NED}_{true}}_\text{P}\f$:
 *
 * correct_dcm_with_tilt(\f$\textbf{C}^{\text{NED}}_\text{P}\f$, tilt)
 *
 * quat_to_dcm(correct_quat_with_tilt(dcm_to_quat(\f$\textbf{C}^{\text{NED}}_\text{P}\f$),
 * -tilt))
 *
 * dot(rot_vec_to_dcm(-tilt), \f$\textbf{C}^{\text{NED}}_\text{P}\f$)
 *
 * dot(transpose(rot_vec_to_dcm(tilt)), \f$\textbf{C}^{\text{NED}}_\text{P}\f$)
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for more details on
 * axis-angle and DCM expressions of attitude.
 *
 * @param dcm The DCM that rotates a vector into the estimated frame in which the tilts are defined.
 * @param tilt Tilt errors that rotate the estimated frame to yield the corrected frame
 * [x-tilt, y-tilt, z-tilt] (radians)
 *
 * @return The corrected DCM.
 */
Matrix3 correct_dcm_with_tilt(const Matrix3& dcm, const Vector3& tilt);

/**
 * Converts a DCM into the equivalent quaternion.
 *
 * If `dcm` is a direction cosine matrix which rotates a vector from frame B
 * to frame A (\f$\textbf{C}_\text{B}^\text{A}\f$), then this function calculates
 * the equivalent quaternion \f$\textbf{q}_\text{B}^\text{A}\f$.
 *
 * References: "Strapdown Inertial Technology", Titterton & Weston; Strapdown Analytics, Vol 1
 * eq. 3.2.4.3-9.
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for more details on
 * quaternion and DCM expressions of attitude.
 * @see quat_to_dcm() for the companion function that converts in the opposite direction.
 * @see dcm_to_rpy()
 *
 * @tparam S Type of input, Matrix by default.
 * @tparam std::enable_if_t<> Constrains S to be 2 dimensional.
 *
 * @param dcm Direction cosine matrix.  Can accept initializer lists.
 *
 * @return The equivalent quaternion.
 */
template <typename S = Matrix, IfTensorOfDim<S, 2>* = nullptr>
Vector4 dcm_to_quat(const S& dcm) {
	auto d0 = dcm(0, 0);
	auto d1 = dcm(1, 1);
	auto d2 = dcm(2, 2);

	// Book just says max, but alternative formulation uses squares, so probably max(abs)
	auto pa = std::fabs(1 + d0 + d1 + d2);
	auto pb = std::fabs(1 + d0 - d1 - d2);
	auto pc = std::fabs(1 - d0 + d1 - d2);
	auto pd = std::fabs(1 - d0 - d1 + d2);

	Scalar q0, q1, q2, q3;

	if (pa >= pb && pa >= pc && pa >= pd) {
		q0        = 0.5 * sqrt(pa);
		auto q0t4 = 4 * q0;
		q1        = (dcm(2, 1) - dcm(1, 2)) / (q0t4);
		q2        = (dcm(0, 2) - dcm(2, 0)) / (q0t4);
		q3        = (dcm(1, 0) - dcm(0, 1)) / (q0t4);
	} else if (pb >= pa && pb >= pc && pb >= pd) {
		q1        = 0.5 * sqrt(pb);
		auto q1t4 = 4 * q1;
		q0        = (dcm(2, 1) - dcm(1, 2)) / (q1t4);
		q2        = (dcm(1, 0) + dcm(0, 1)) / (q1t4);
		q3        = (dcm(0, 2) + dcm(2, 0)) / (q1t4);
	} else if (pc >= pa && pc >= pb && pc >= pd) {
		q2        = 0.5 * sqrt(pc);
		auto q2t4 = 4 * q2;
		q0        = (dcm(0, 2) - dcm(2, 0)) / (q2t4);
		q1        = (dcm(1, 0) + dcm(0, 1)) / (q2t4);
		q3        = (dcm(2, 1) + dcm(1, 2)) / (q2t4);
	} else {
		q3        = 0.5 * sqrt(pd);
		auto q3t4 = 4 * q3;
		q0        = (dcm(1, 0) - dcm(0, 1)) / (q3t4);
		q1        = (dcm(0, 2) + dcm(2, 0)) / (q3t4);
		q2        = (dcm(2, 1) + dcm(1, 2)) / (q3t4);
	}

	// use of sign bit is because -0.0 and 0.0 are technically both <= 0 and neither are < 0.
	if (std::signbit(q0)) {
		return {-q0, -q1, -q2, -q3};
	} else {
		return {q0, q1, q2, q3};
	}
}

/**
 * This is a batched version of `dcm_to_quat`.
 *
 * @overload
 *
 * @see dcm_to_quat
 *
 * @tparam B Type of input.
 * @tparam std::enable_if_t<> Constrains B to be 3 dimensional.
 *
 * @param dcm Direction cosine matrices, shape (N, 3, 3).
 *
 * @return The equivalent quaternions, shape (N, 4).
 */
template <typename B, IfTensorOfDim<B, 3>* = nullptr>
Matrix dcm_to_quat(const B& dcm) {
	const size_t N = dcm.shape()[0];

	Matrix quats = empty(N, 4);

	Scalar d0, d1, d2, pa, pb, pc, pd, q0, q1, q2, q3, q0t4, q1t4, q2t4, q3t4;

	for (size_t i = 0; i < N; i++) {
		d0 = dcm(i, 0, 0);
		d1 = dcm(i, 1, 1);
		d2 = dcm(i, 2, 2);

		// Book just says max, but alternative formulation uses squares, so probably max(abs)
		pa = std::fabs(1 + d0 + d1 + d2);
		pb = std::fabs(1 + d0 - d1 - d2);
		pc = std::fabs(1 - d0 + d1 - d2);
		pd = std::fabs(1 - d0 - d1 + d2);
		q0 = 0.0;
		q1 = 0.0;
		q2 = 0.0;
		q3 = 0.0;

		if (pa >= pb && pa >= pc && pa >= pd) {
			q0   = 0.5 * sqrt(pa);
			q0t4 = 4 * q0;
			q1   = (dcm(i, 2, 1) - dcm(i, 1, 2)) / (q0t4);
			q2   = (dcm(i, 0, 2) - dcm(i, 2, 0)) / (q0t4);
			q3   = (dcm(i, 1, 0) - dcm(i, 0, 1)) / (q0t4);
		} else if (pb >= pa && pb >= pc && pb >= pd) {
			q1   = 0.5 * sqrt(pb);
			q1t4 = 4 * q1;
			q0   = (dcm(i, 2, 1) - dcm(i, 1, 2)) / (q1t4);
			q2   = (dcm(i, 1, 0) + dcm(i, 0, 1)) / (q1t4);
			q3   = (dcm(i, 0, 2) + dcm(i, 2, 0)) / (q1t4);
		} else if (pc >= pa && pc >= pb && pc >= pd) {
			q2   = 0.5 * sqrt(pc);
			q2t4 = 4 * q2;
			q0   = (dcm(i, 0, 2) - dcm(i, 2, 0)) / (q2t4);
			q1   = (dcm(i, 1, 0) + dcm(i, 0, 1)) / (q2t4);
			q3   = (dcm(i, 2, 1) + dcm(i, 1, 2)) / (q2t4);
		} else {
			q3   = 0.5 * sqrt(pd);
			q3t4 = 4 * q3;
			q0   = (dcm(i, 1, 0) - dcm(i, 0, 1)) / (q3t4);
			q1   = (dcm(i, 0, 2) + dcm(i, 2, 0)) / (q3t4);
			q2   = (dcm(i, 2, 1) + dcm(i, 1, 2)) / (q3t4);
		}

		// use of sign bit is because -0.0 and 0.0 are technically both <= 0.
		if (std::signbit(q0)) {
			quats(i, 0) = -q0;
			quats(i, 1) = -q1;
			quats(i, 2) = -q2;
			quats(i, 3) = -q3;
		} else {
			quats(i, 0) = q0;
			quats(i, 1) = q1;
			quats(i, 2) = q2;
			quats(i, 3) = q3;
		}
	}
	return quats;
}

/**
 * Converts a DCM to Euler angles.
 *
 * When provided a `dcm` that rotates a vector from frame B to frame A
 * (\f$\textbf{C}_\text{B}^\text{A}\f$), this function returns the yaw, pitch, and roll that
 * correspond to a 3-2-1 frame rotation sequence from frame A to frame B as described in
 * `rpy_to_dcm`.
 *
 * REFERENCE: This function implements the transpose of Titterton and Weston eqs 3.66 through 3.68,
 * and Savage, eq 3.2.3.2-3 and 3.2.3.2-4. Note the difference in the equations for the pitch near
 * \f$ \frac{-\pi}{2} \f$ case with the Savage version having a \f$ \pi \f$ offset; this is the
 * version implemented here. When pitch is near \f$ \pm\frac{\pi}{2} \f$, roll and yaw rotations
 * appear to be about the same axis and one must be selected to be 'held' and the other solved for
 * in relation to the held angle. This function will always hold the 'roll' angle when required.
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for more details on Euler
 * angle and DCM expressions of attitude.
 * @see rpy_to_dcm() for the companion function that converts in the opposite direction.
 * @see dcm_to_quat()
 *
 * @tparam S Type of input, Matrix by default.
 * @tparam std::enble_if_t<> Constrains S to be 2 dimensional.
 *
 * @param dcm Direction cosine matrix (\f$\textbf{C}_\text{B}^\text{A}\f$), can accept initializer
 * lists.
 *
 * @return Equivalent Euler angles [roll pitch yaw] (radians) that describe a frame rotation from
 * frame A to frame B.
 */
template <typename S = Matrix, IfTensorOfDim<S, 2>* = nullptr>
Vector3 dcm_to_rpy(const S& dcm) {
	auto asin_arg = std::min(1.0, std::max(dcm(2, 0), -1.0));
	auto r        = atan2(dcm(2, 1), dcm(2, 2));
	auto p        = -asin(asin_arg);
	auto y        = atan2(dcm(1, 0), dcm(0, 0));

	if (asin_arg <= -1 + 1e-12) {
		auto y_min_r = atan2(dcm(1, 2) - dcm(0, 1), dcm(0, 2) + dcm(1, 1));
		y            = y_min_r + r;
	}
	if (asin_arg >= 1 - 1e-12) {
		auto y_pls_r = atan2(dcm(1, 2) + dcm(0, 1), dcm(0, 2) - dcm(1, 1)) + PI;
		y            = remainder((y_pls_r - r), 2.0 * PI);
	}
	return {r, p, y};
}

/**
 * This is a batched version of `dcm_to_rpy`.
 *
 * @overload
 *
 * @see dcm_to_rpy
 *
 * @tparam B Type of input.
 * @tparam std::enable_if_t<> Constrains B to be 3 dimensional.
 *
 * @param dcm Direction cosine matrices, shape (N, 3, 3).
 *
 * @return The equivalent Euler angle vectors, shape (N, 3).
 */
template <typename B, IfTensorOfDim<B, 3>* = nullptr>
Matrix dcm_to_rpy(const B& dcm) {
	const size_t N = dcm.shape()[0];

	Matrix rpys = empty(N, 3);

	Scalar asin_arg;

	for (size_t i = 0; i < N; i++) {

		asin_arg   = std::min(1.0, std::max(dcm(i, 2, 0), -1.0));
		rpys(i, 0) = atan2(dcm(i, 2, 1), dcm(i, 2, 2));
		rpys(i, 1) = -asin(asin_arg);
		rpys(i, 2) = atan2(dcm(i, 1, 0), dcm(i, 0, 0));

		if (asin_arg <= -1 + 1e-12) {
			auto y_min_r = atan2(dcm(i, 1, 2) - dcm(i, 0, 1), dcm(i, 0, 2) + dcm(i, 1, 1));
			rpys(i, 2)   = y_min_r + rpys(i, 0);
		}
		if (asin_arg >= 1 - 1e-12) {
			auto y_pls_r = atan2(dcm(i, 1, 2) + dcm(i, 0, 1), dcm(i, 0, 2) - dcm(i, 1, 1)) + PI;
			rpys(i, 2)   = remainder((y_pls_r - rpys(i, 0)), 2.0 * PI);
		}
	}

	return rpys;
}

/**
 * Converts small changes of latitude (radians) into distance in meters along the North axis of the
 * local level NED frame. Input parameters are with respect to the WGS-84 ellipsoid.
 *
 * This conversion is approximate, and the error grows as the `delta_lat` grows.
 *
 * Example:
 *
 * `delta_lat` = \f$Lat_1 - Lat_0\f$ (radians)
 *
 * `approx_lat` = \f$(Lat_1 + Lat_0)/2\f$ (or just \f$Lat_0\f$ or \f$Lat_1\f$) (radians)
 *
 * `altitude` = approximate WGS-84 altitude (meters)
 *
 * The function will return the approximate distance, in meters, between the points (\f$Lat_1\f$,
 * \f$Lon\f$, `altitude`) and (\f$Lat_0\f$, \f$Lon\f$, `altitude`) for any longitude \f$Lon\f$. The
 * sign of the output matches the sign of the `delta_lat` input.
 *
 * @see delta_lon_to_east()
 * @see north_to_delta_lat() for the companion function that converts in the opposite direction.
 * @see east_to_delta_lon()
 *
 * @tparam S1 Type of delta_lat.
 * @tparam S2 Type of approx_lat.
 * @tparam S3 Type of altitude.
 * @tparam std::enable_if_t<> Constrains S1, S2, and S3 to be 0 dimensional.
 *
 * @param delta_lat Small distance in latitude to convert (radians)
 * @param approx_lat Approximate latitude (radians)
 * @param altitude WGS-84 altitude (ellipsoidal) (meters), 0.0 by default.
 *
 * @return Equivalent distance in the north direction (meters).
 */
template <typename S1, typename S2, typename S3, IfAllTensorsOfDim<0, S1, S2, S3>* = nullptr>
double delta_lat_to_north(const S1& delta_lat, const S2& approx_lat, const S3& altitude) {
	return (meridian_radius(approx_lat) + altitude) * delta_lat;
}

/**
 * This is a batched version of `delta_lat_to_north`.
 *
 * @overload
 *
 * @see delta_lat_to_north
 *
 * @tparam B1 Type of delta_lat.
 * @tparam B2 Type of approx_lat.
 * @tparam B3 Type of altitude.
 * @tparam std::enable_if_t<> Constrains B1, B2, and B3 to have a maximum dimension of precisely 1.
 *
 * @param delta_lat Small distances in latitude to convert (radians), shape (N).  Can accept
 * initializer lists or a `double` as well.
 * @param approx_lat Approximate latitudes (radians), shape (N).  Can accept initializer lists or a
 * `double` as well.
 * @param altitude WGS-84 altitudes (ellipsoidal) (meters), shape (N).  Can accept initializer lists
 * or a `double` as well.  0.0 by default.
 *
 * @return Equivalent distances in the north direction (meters), shape (N).
 */
template <typename B1                     = Vector,
          typename B2                     = Vector,
          typename B3                     = Vector,
          IfTensorsMaxDim<1, B1, B2, B3>* = nullptr>
Vector delta_lat_to_north(const B1& delta_lat, const B2& approx_lat, const B3& altitude) {
	return (meridian_radius(approx_lat) + altitude) * delta_lat;
}

// default altitude of 0.0
#ifndef NEED_DOXYGEN_EXHALE_WORKAROUND
template <typename A = Vector, typename B = Vector>
auto delta_lat_to_north(const A& delta_lat, const B& approx_lat) {
	return delta_lat_to_north(delta_lat, approx_lat, 0.0);
}
#endif

/**
 * Converts small changes of longitude (radians) into distance in meters along the East axis of
 the
 * local level NED frame. Input parameters are with respect to the WGS-84 ellipsoid.
 *
 * This conversion is approximate, and the error grows as the `delta_lon` grows.
 *
 * Example:
 *
 * `delta_lon` = \f$Lon_1 - Lon_0\f$ (radians)
 *
 * `approx_lat` = approximate WGS-84 latitude (radians)
 *
 * `altitude` = approximate WGS-84 altitude (meters)
 *
 * The function will return the approximate distance, in meters, between the points
 (`approx_lat`,
 * \f$Lon_1\f$, `altitude`) and (`approx_lat`, \f$Lon_0\f$, `altitude`). The sign of the output
 * matches the sign of the `delta_lon` input.
 *
 * @see delta_lat_to_north()
 * @see north_to_delta_lat()
 * @see east_to_delta_lon() for the companion function that converts in the opposite direction.
 *
 * @tparam S1 Type of delta_lon.
 * @tparam S2 Type of approx_lat.
 * @tparam S3 Type of altitude.
 * @tparam std::enable_if_t<> Constrains S1, S2, and S3 to have dimension 0.
 *
 * @param delta_lon Small distance in longitude to convert (radians)
 * @param approx_lat Approximate latitude (radians)
 * @param altitude WGS-84 altitude (ellipsoidal) (meters), 0.0 by default.
 *
 * @return Equivalent distance (meters).
 */
template <typename S1, typename S2, typename S3, IfAllTensorsOfDim<0, S1, S2, S3>* = nullptr>
double delta_lon_to_east(const S1& delta_lon, const S2& approx_lat, const S3& altitude) {
	return (transverse_radius(approx_lat) + altitude) * delta_lon * cos(approx_lat);
}

/**
 * This is a batched version of `delta_lon_to_east`.
 *
 * @overload
 *
 * @see delta_lon_to_east
 *
 * @tparam B1 Type of delta_lon.
 * @tparam B2 Type of approx_lat.
 * @tparam B3 Type of altitude.
 * @tparam std::enable_if_t<> Constrains B1, B2, and B3 to have a maximum dimension of precisely 1.
 *
 * @param delta_lon Small distances in longitude to convert (radians), shape (N).  Can accept
 * initializer lists or a `double` as well.
 * @param approx_lat Approximate latitudes (radians), shape (N).  Can accept initializer lists or a
 * `double` as well.
 * @param altitude WGS-84 altitudes (ellipsoidal) (meters), shape (N).  Can accept initializer lists
 * or a `double` as well.  0.0 by default.
 *
 * @return Equivalent distances in the east direction (meters), shape (N).
 */
template <typename B1                     = Vector,
          typename B2                     = Vector,
          typename B3                     = Vector,
          IfTensorsMaxDim<1, B1, B2, B3>* = nullptr>
Vector delta_lon_to_east(const B1& delta_lon, const B2& approx_lat, const B3& altitude) {
	return (transverse_radius(approx_lat) + altitude) * delta_lon * cos(approx_lat);
}

// default altitude of 0.0
#ifndef NEED_DOXYGEN_EXHALE_WORKAROUND
template <typename A = Vector, typename B = Vector>
auto delta_lon_to_east(const A& delta_lon, const B& approx_lat) {
	return delta_lon_to_east(delta_lon, approx_lat, 0.0);
}
#endif

/**
 * Converts from continuous-time dynamic system representation to discrete-time representation
 * using first-order algorithm.
 *
 * If the continuous time system is
 *
 * \f$ \dot{\textbf{x}} = \textbf{Fx} + \textbf{w} \f$
 *
 * \f$ E[\textbf{w}(t-\tau)\textbf{w}^T(t-\tau)] = \textbf{Q}\delta(\tau) \f$
 *
 * where \f$ E[] \f$ is the expectation operator and \f$ \delta(\tau) \f$ is the Dirac delta
 * function.
 *
 * This can be converted to an equivalent discrete-time system
 *
 * \f$ \textbf{x}_{t_{k+1}} = \pmb{\Phi}\textbf{x}_{t_k} + \textbf{w}_d \f$
 *
 * where
 *
 * \f$ E[\textbf{w}_{d_j}\textbf{w}^T_{d_k}] = \textbf{Q}_d \delta_{kj} \f$
 *
 * and \f$ \delta_{kj} \f$ is the Kronecker delta function.
 *
 * When \f$\textbf{F}\f$ is continuous over the time interval between \f$t_{k}\f$ and
 * \f$t_{k+1}\f$, then \f$\pmb{\Phi}\f$ can be calculated as
 *
 * \f$ \begin{equation}
 * \pmb{\Phi}=e^{\textbf{F}\Delta t} = \textbf{I} + \sum\limits_{n=1}^\infty{1 \over
 * {n!}}\textbf{F}^n\Delta t^n\end{equation} \f$
 *
 * \f$ \Delta t = t_{k+1} - t_k \f$
 *
 * This function calculates \f$\pmb{\Phi}\f$ using the first-order calculation
 *
 * \f$\pmb{\Phi}_1 \approx \textbf{I} + \textbf{F}\Delta t\f$
 *
 * and
 *
 * \f$\textbf{Q}_{d_1} \approx \textbf{Q}\Delta t\f$
 *
 * @see discretize_van_loan()
 * @see discretize_second_order()
 *
 * @param f Continuous-time dynamics matrix \f$\textbf{F}\f$
 * @param q Continuous-time process noise \f$\textbf{Q}\f$
 * @param dt Time (seconds) over which propagation is to take place (\f$\Delta t\f$)
 *
 * @return (`Phi`, `Qd`) The discrete-time matrices as an std::pair. First is `Phi`
 * \f$=\pmb{\Phi}_1\f$, second is `Qd` \f$=\textbf{Q}_{d_1}\f$.
 */
std::pair<Matrix, Matrix> discretize_first_order(const Matrix& f, const Matrix& q, double dt);

/**
 * Converts from continuous-time dynamic system representation to discrete-time representation
 * using second order algorithm.
 *
 * See `discretize_first_order()` for variable definitions.
 *
 * This function calculates \f$\pmb{\Phi}\f$ using the second order calculation
 *
 * \f$\pmb{\Phi}_2 \approx \textbf{I} + \textbf{F}\Delta t + \frac{1}{2}\textbf{F}^2\Delta t^2\f$
 *
 * and
 *
 * \f$\textbf{Q}_{d_2} \approx \frac{\pmb{\Phi}\textbf{Q}\pmb{\Phi}^T + \textbf{Q}}{2}\Delta t\f$
 *
 * @see discretize_first_order()
 * @see discretize_van_loan()
 *
 * @param f Continuous-time dynamics matrix \f$\textbf{F}\f$
 * @param q Continuous-time process noise \f$\textbf{Q}\f$
 * @param dt Time (seconds) over which propagation is to take place (\f$\Delta t\f$)
 *
 * @return (`Phi`, `Qd`) The discrete-time matrices as an std::pair. First is `Phi`
 * \f$=\pmb{\Phi}_2\f$, second is `Qd` \f$=\textbf{Q}_{d_2}\f$.
 **/
std::pair<Matrix, Matrix> discretize_second_order(const Matrix& f, const Matrix& q, double dt);


/**
 * Converts from continuous-time dynamic system representation to discrete-time representation
 * using the full Van Loan solution.
 *
 * See `discretize_first_order()` for variable definitions.
 *
 * @see calc_van_loan() if the `Phi` output is not needed.
 * @see discretize_first_order()
 * @see discretize_second_order()
 *
 * @param f Continuous-time dynamics matrix \f$\textbf{F}\f$
 * @param q Continuous-time process noise \f$\textbf{Q}\f$
 * @param dt Time (seconds) over which propagation is to take place (\f$\Delta t\f$)
 *
 * @return (`Phi`, `Qd`) The discrete-time matrices as an std::pair. First is `Phi`
 * \f$=\pmb{\Phi}\f$, second is `Qd` \f$=\textbf{Q}_{d}\f$.
 *
 */
std::pair<Matrix, Matrix> discretize_van_loan(const Matrix& f, const Matrix& q, double dt);

/**
 * Converts distance in meters along the East axis of the local level NED frame into radians of
 * longitude with respect to the WGS-84 ellipsoid.
 *
 * Example Use: You are at a position (`lat`, `lon`, `alt`). You want to know what the longitude
 * is at a point 10m to your east (positive longitude). That can be calculated using this function
 * as:
 *
 * `new_lon = lon + east_to_delta_lon(10.0, lat, alt)`
 *
 * @see delta_lat_to_north()
 * @see delta_lon_to_east()  for the companion function that converts in the opposite direction.
 * @see north_to_delta_lat()
 *
 * @tparam S1 Type of east_distance.
 * @tparam S2 Type of approx_lat.
 * @tparam S3 Type of altitude.
 * @tparam std::enable_if_t<> Constrains S1, S2, and S3 to be 0 dimensional.
 *
 * @param east_distance Small distance to convert to radians of longitude (meters, east is positive)
 * @param approx_lat Approximate latitude (radians)
 * @param altitude WGS-84 altitude (ellipsoidal) (meters), 0.0 by default.
 *
 * @return Equivalent distance expressed as radians of latitude (radians).
 */
template <typename S1, typename S2, typename S3, IfAllTensorsOfDim<0, S1, S2, S3>* = nullptr>
double east_to_delta_lon(const S1& east_distance, const S2& approx_lat, const S3& altitude) {
	return east_distance / ((transverse_radius(approx_lat) + altitude) * cos(approx_lat));
}

/**
 * This is a batched version of `east_to_delta_lon`.
 *
 * @overload
 *
 * @see east_to_delta_lon
 *
 * @tparam B1 Type of east_distance.
 * @tparam B2 Type of approx_lat.
 * @tparam B3 Type of altitude.
 * @tparam std::enable_if_t<> Constrains B1, B2, and B3 to have a maximum dimension of precisely 1.
 *
 * @param east_distance Small distances to convert to radians of longitude (meters, east is
 * positive), shape (N).  Can accept initializer lists or a `double` as well.
 * @param approx_lat Approximate latitudes (radians), shape (N).  Can accept initializer lists or a
 * `double` as well.
 * @param altitude WGS-84 altitudes (ellipsoidal) (meters), shape (N).  Can accept initializer lists
 * or a `double` as well.  0.0 by default.
 *
 * @return Equivalent distances in the east direction (meters), shape (N).
 */
template <typename B1                     = Vector,
          typename B2                     = Vector,
          typename B3                     = Vector,
          IfTensorsMaxDim<1, B1, B2, B3>* = nullptr>
Vector east_to_delta_lon(const B1& east_distance, const B2& approx_lat, const B3& altitude) {
	return east_distance / ((transverse_radius(approx_lat) + altitude) * cos(approx_lat));
}

// default altitude of 0.0
#ifndef NEED_DOXYGEN_EXHALE_WORKAROUND
template <typename A = Vector, typename B = Vector>
auto east_to_delta_lon(const A& east_distance, const B& approx_lat) {
	return east_to_delta_lon(east_distance, approx_lat, 0.0);
}
#endif

/**
 * Computes the DCM that rotates a vector from the NED frame to the ECEF frame
 * (\f$\textbf{C}_\text{NED}^\text{ECEF}\f$). This DCM
 * varies as a function of position, which in this function is provided as an ECEF position vector.
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for definitions of
 * ECEF and NED frames.
 * @see llh_to_cen()
 *
 * @param Pe ECEF Position Vector at which \f$\textbf{C}_\text{NED}^\text{ECEF}\f$ is desired: [x y
 * z] (meters)
 *
 * @return NED-to-ECEF DCM (\f$\textbf{C}_\text{NED}^\text{ECEF}\f$).
 */
Matrix3 ecef_to_cen(const Vector3& Pe);

/**
 * Converts a position expressed in Earth-Centered
 *  Earth-Fixed (ECEF) coordinates to WGS-84 latitude, longitude, height above
 *  ellipsoid (LLH) coordinates (sometimes referred to as "Geodetic" coordinates).
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for more information on
 * ECEF and latitude, longitude, height representations of position.
 * @see llh_to_ecef()
 *
 * @param Pe ECEF Position: [x y z] (meters)
 *
 * @return Geodetic position defined relative to WGS-84 ellipsoid: length three vector consisting
 * of latitude, longitude, and ellipsoidal altitude (radians, radians, meters).
 *
 * REFERENCE: [WGS-84 Reference System (NIMA report TR
 * 8350.2)](https://nga-rescue.is4s.us/wgs84fin.pdf)
 */
Vector3 ecef_to_llh(const Vector3& Pe);

/**
 * Convert ECEF location to fixed local-level NED frame location.
 *
 * In general, an NED frame is defined as a local-level frame at a specified location. In this
 * function, the location of the origin of the NED frame is expressed in ECEF coordinates as
 * \f$\textbf{p}_0^{\text{ECEF}}\f$. Specifying \f$\textbf{p}_0^{\text{ECEF}}\f$ creates a
 * local-level NED frame at that location, which we will call the \f$\text{NED}_{\textbf{p}_0}\f$
 * frame.
 *
 * This function converts the ECEF position of an arbitrary point \f$\textbf{p}^{\text{ECEF}}\f$ to
 * a representation of that same point coordinatized in the \f$\text{NED}_{\textbf{p}_0}\f$ frame.
 * Note that this more than just a rotation from ECEF to NED. There is also a translation to
 * accommodate the different origins of the two coordinate frames (\f$\text{ECEF}\f$ and
 * \f$\text{NED}_{\textbf{p}_0}\f$).
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for definitions of ECEF and
 * NED frames.
 * @see local_level_to_ecef() for the companion function that converts in the opposite direction.
 *
 * @param P0e \f$\text{NED}_{\textbf{p}_0}\f$ frame origin expressed in ECEF coordinates (meters) =
 * \f$\textbf{p}_0^{\text{ECEF}}\f$
 * @param Pe Location of point in ECEF frame (meters) = \f$\textbf{p}^{\text{ECEF}}\f$
 *
 * @return Location of `Pe` expressed in the fixed local-level \f$\text{NED}_{\textbf{p}_0}\f$ frame
 * (meters).
 */
Vector3 ecef_to_local_level(const Vector3& P0e, const Vector3& Pe);

/**
 * Computes the DCM that rotates a vector from the NED frame to the ECEF frame
 * (\f$\textbf{C}_\text{NED}^\text{ECEF}\f$). This DCM varies as a function of position, which in
 * this function is provided as a (lat, lon, height) vector.
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for definitions of NED and
 * ECEF frames.
 * @see ecef_to_cen()
 *
 * @param Pwgs Geodetic position defined relative to WGS-84 ellipsoid: length three vector
 * consisting of latitude, longitude, and ellipsoidal height (radians, radians, meters)
 *
 * @return NED-to-ECEF DCM (\f$\textbf{C}_\text{NED}^\text{ECEF}\f$).
 */
Matrix3 llh_to_cen(const Vector3& Pwgs);

/**
 * Converts a position expressed in WGS-84 latitude, longitude, height above
 * ellipsoid (LLH) coordinates (sometimes referred to as "Geodetic" coordinates) to Earth-Centered
 * Earth-Fixed (ECEF) coordinates.
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for more information about
 * ECEF and latitude, longitude, and height expressions of position.
 * @see ecef_to_llh() for the companion function that converts in the opposite direction.
 *
 * @param Pwgs Geodetic position defined relative to WGS-84 ellipsoid, expressed as a length three
 * vector of latitude, longitude, and ellipsoidal altitude (radians, radians, meters)
 *
 * @return ECEF Position: [x y z] (meters).
 *
 * REFERENCE: [WGS-84 Reference System (NIMA report TR
 * 8350.2)](https://nga-rescue.is4s.us/wgs84fin.pdf)
 */
Vector3 llh_to_ecef(const Vector3& Pwgs);

/**
 * Convert fixed local-level NED frame location to ECEF location.
 *
 * In general, an NED frame is defined as a local-level frame at a specified location. In this
 * function, the location of the origin of the NED frame is expressed in ECEF coordinates as
 * \f$\textbf{p}_0^{\text{ECEF}}\f$. Specifying \f$\textbf{p}_0^{\text{ECEF}}\f$ creates a
 * local-level NED frame at that location, which we will call the \f$\text{NED}_{\textbf{p}_0}\f$
 * frame.
 *
 * This function converts the position of an arbitrary point in the NED frame
 * (\f$\textbf{p}^{\text{NED}_{\textbf{p}_0}}\f$) to a representation of that same point
 * coordinatized in the ECEF frame (\f$\textbf{p}^{\text{ECEF}}\f$). Note that this more than just
 * a rotation from NED to ECEF. There is also a translation to accommodate the different origins of
 * the two coordinate frames (\f$\text{ECEF}\f$ and \f$\text{NED}_{\textbf{p}_0}\f$).
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for more definitions of NED
 * and ECEF frames.
 * @see ecef_to_local_level() for the companion function that converts in the opposite direction.
 *
 * @param P0e \f$\text{NED}_{\textbf{p}_0}\f$ frame origin expressed in ECEF coordinates (meters) =
 * \f$\textbf{p}_0^{\text{ECEF}}\f$
 * @param Pn Location of point in NED frame (meters) = \f$\textbf{p}^{\text{NED}_{\textbf{p}_0}}\f$
 *
 * @return Location of `Pn` expressed in the ECEF frame (\f$\textbf{p}_0^{\text{ECEF}}\f$).
 */
Vector3 local_level_to_ecef(const Vector3& P0e, const Vector3& Pn);

/**
 * Converts distance in meters along the North axis of the local level NED frame into radians of
 * latitude with respect to the WGS-84 ellipsoid.
 *
 * Example Use: You are at a position (`lat`, `lon`, `alt`). You want to know what the longitude
 * is at a point 10m to your north (positive latitude). That can be calculated using this function
 * as:
 *
 * `new_lat = lat + north_to_delta_lat(10.0, lat, alt)`
 *
 * @see delta_lat_to_north() for the companion function that converts in the opposite direction.
 * @see delta_lon_to_east()
 * @see east_to_delta_lon()
 *
 * @tparam S1 Type of north_distance.
 * @tparam S2 Type of approx_lat.
 * @tparam S3 Type of altitude.
 * @tparam std::enable_if_t<> Constrains S1, S2, and S3 to be 0 dimensional.
 *
 * @param north_distance Small distance in meters to convert (meters)
 * @param approx_lat Approximate latitude (radians)
 * @param altitude WGS-84 altitude (ellipsoidal) (meters), 0.0 by default.
 *
 * @return Equivalent distance expressed as radians of latitude (radians).
 */
template <typename S1, typename S2, typename S3, IfAllTensorsOfDim<0, S1, S2, S3>* = nullptr>
double north_to_delta_lat(const S1& north_distance, const S2& approx_lat, const S3& altitude) {
	return north_distance / (meridian_radius(approx_lat) + altitude);
}

/**
 * This is a batched version of `north_to_delta_lat`.
 *
 * @overload
 *
 * @see north_to_delta_lat
 *
 * @tparam B1 Type of north_distance.
 * @tparam B2 Type of approx_lat.
 * @tparam B3 Type of altitude.
 * @tparam std::enable_if_t<> Constrains B1, B2, and B3 to have a maximum dimension or precisely 1.
 *
 * @param north_distance Small distance in meters to convert to latitude (meters, north is
 * positive), shape (N).  Can accept initializer lists or a `double` as well.
 * @param approx_lat Approximate latitudes (radians), shape (N).  Can accept initializer lists or a
 * `double` as well.
 * @param altitude WGS-84 altitudes (ellipsoidal) (meters), shape (N).  Can accept initializer lists
 * or a `double` as well.  0.0 by default.
 *
 * @return Equivalent distances in the north direction (meters), shape (N).
 */
template <typename B1                     = Vector,
          typename B2                     = Vector,
          typename B3                     = Vector,
          IfTensorsMaxDim<1, B1, B2, B3>* = nullptr>
Vector north_to_delta_lat(const B1& north_distance, const B2& approx_lat, const B3& altitude) {
	return north_distance / (meridian_radius(approx_lat) + altitude);
}

// default altitude of 0.0
#ifndef NEED_DOXYGEN_EXHALE_WORKAROUND
template <typename A = Vector, typename B = Vector>
auto north_to_delta_lat(const A& north_distance, const B& approx_lat) {
	return north_to_delta_lat(north_distance, approx_lat, 0.0);
}
#endif

/**
 * Converts a quaternion into the equivalent direction cosine matrix.
 *
 * If `quat` is a quaternion cosine matrix which rotates a vector from frame B to frame A
 * (\f$\textbf{q}_\text{B}^\text{A}\f$), then this function calculates the equivalent direction
 * cosine matrix (DCM) \f$\textbf{C}_\text{B}^\text{A}\f$, as defined in [Coordinate
 * Frames](../tutorial/coordinate_frames.html).
 *
 * References: "Strapdown Inertial Technology", Titterton & Weston.
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for more details about
 * quaternion and DCM expressions of attitude.
 * @see dcm_to_quat() for the companion function that converts in the opposite direction.
 *
 * @tparam S Type of quat, Vector by default.
 * @tparam std::enable_if_t<> Constrains S to be 1 dimensional.
 *
 * @param quat The input quaternion, shape (4).  Can accept initializer lists.
 *
 * @return The equivalent DCM, shape (3, 3)
 */
template <typename S = Vector, IfTensorOfDim<S, 1>* = nullptr>
Matrix3 quat_to_dcm(const S& quat) {
	auto q0 = quat(0);
	auto q1 = quat(1);
	auto q2 = quat(2);
	auto q3 = quat(3);
	auto a2 = pow(q0, 2);
	auto b2 = pow(q1, 2);
	auto c2 = pow(q2, 2);
	auto d2 = pow(q3, 2);
	auto ab = q0 * q1;
	auto ac = q0 * q2;
	auto ad = q0 * q3;
	auto bc = q1 * q2;
	auto bd = q1 * q3;
	auto cd = q2 * q3;

	return {{a2 + b2 - c2 - d2, 2 * (bc - ad), 2 * (bd + ac)},
	        {2 * (bc + ad), a2 - b2 + c2 - d2, 2 * (cd - ab)},
	        {2 * (bd - ac), 2 * (cd + ab), a2 - b2 - c2 + d2}};
}

/**
 * This is a batched version of `quat_to_dcm`.
 *
 * @overload
 *
 * @see quat_to_dcm
 *
 * @tparam B Type of quat, Matrix by default.
 * @tparam std::enable_if_t<> Constrains B to be 2 dimensional.
 *
 * @param quat The input quaternions, shape (N, 4).  Can accept initializer lists.
 *
 * @return The equivalent dcms, shape (N, 3, 3).
 */
template <typename B = Matrix, IfTensorOfDim<B, 2>* = nullptr>
Tensor<3> quat_to_dcm(const B& quat) {
	const size_t N = quat.shape()[0];

	Tensor<3> dcms = empty(N, 3, 3);

	Scalar q0, q1, q2, q3, a2, b2, c2, d2, ab, ac, ad, bc, bd, cd;

	for (size_t i = 0; i < N; i++) {
		q0 = quat(i, 0);
		q1 = quat(i, 1);
		q2 = quat(i, 2);
		q3 = quat(i, 3);
		a2 = pow(q0, 2);
		b2 = pow(q1, 2);
		c2 = pow(q2, 2);
		d2 = pow(q3, 2);
		ab = q0 * q1;
		ac = q0 * q2;
		ad = q0 * q3;
		bc = q1 * q2;
		bd = q1 * q3;
		cd = q2 * q3;

		dcms(i, 0, 0) = a2 + b2 - c2 - d2;
		dcms(i, 0, 1) = 2 * (bc - ad);
		dcms(i, 0, 2) = 2 * (bd + ac);
		dcms(i, 1, 0) = 2 * (bc + ad);
		dcms(i, 1, 1) = a2 - b2 + c2 - d2;
		dcms(i, 1, 2) = 2 * (cd - ab);
		dcms(i, 2, 0) = 2 * (bd - ac);
		dcms(i, 2, 1) = 2 * (cd + ab);
		dcms(i, 2, 2) = a2 - b2 - c2 + d2;
	}

	return dcms;
}

/**
 * Converts a quaternion to Euler angles.
 *
 * When provided a `quat` that rotates a vector from frame B to frame A
 * (\f$\textbf{q}_\text{B}^\text{A}\f$), this function returns the yaw, pitch, and roll that
 * correspond to a 3-2-1 frame rotation sequence from frame A to frame B as
 * described in `rpy_to_dcm()`.
 *
 * References: "Strapdown Inertial Technology", Titterton & Weston, 2nd ed eq 3.65. Note that the
 * book equation has a sign error on the 'd' term; it should be
 *
 * \f$ d = \cos{\frac{\phi}{2}}\cos{\frac{\theta}{2}}\sin{\frac{\psi}{2}} -
 * \sin{\frac{\phi}{2}}\sin{\frac{\theta}{2}}\cos{\frac{\psi}{2}} \f$.
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for more details about
 * quaternion and Euler angle expressions of attitude.
 * @see rpy_to_quat()  for the companion function that converts in the opposite direction.
 *
 * @tparam S Type of quat, Vector by default.
 * @tparam std::enable_if_t<> Constrains S to be 1 dimensional.
 *
 * @param quat Quaternion (\f$\textbf{q}_\text{B}^\text{A}\f$).  Can accept initializer lists.
 *
 * @return The equivalent Euler angles [roll pitch yaw] (radians) that define a frame rotation from
 * frame A to frame B.
 */
template <typename S = Vector, IfTensorOfDim<S, 1>* = nullptr>
Vector3 quat_to_rpy(const S& quat) {
	auto q0   = quat[0];
	auto q1   = quat[1];
	auto q2   = quat[2];
	auto q3   = quat[3];
	auto roll = atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2)),

	     pitch = asin(std::min(std::max(2 * (q0 * q2 - q1 * q3), -1.0), 1.0)),

	     yaw = atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3));
	return {roll, pitch, yaw};
}

/**
 * This is a batched version of `quat_to_rpy`.
 *
 * @overload
 *
 * @see quat_to_rpy
 *
 * @tparam B Type of quat, Matrix by default.
 * @tparam std::enable_if_t<> Constrains B to be 2 dimensional.
 *
 * @param quat The input quaternions, shape (N, 4).  Can accept initializer lists.
 *
 * @return The equivalent Euler angle vectors, shape (N, 3).
 */
template <typename B = Matrix, IfTensorOfDim<B, 2>* = nullptr>
Matrix quat_to_rpy(const B& quat) {
	const size_t N = quat.shape()[0];

	// allocate return Matrix
	Matrix rpys = empty(N, 3);

	Scalar q0, q1, q2, q3;

	for (size_t i = 0; i < N; i++) {
		q0 = quat(i, 0);
		q1 = quat(i, 1);
		q2 = quat(i, 2);
		q3 = quat(i, 3);

		// roll, pitch, and yaw
		rpys(i, 0) = atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2));
		rpys(i, 1) = asin(std::min(std::max(2 * (q0 * q2 - q1 * q3), -1.0), 1.0));
		rpys(i, 2) = atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3));
	}
	return rpys;
}

/**
 * Converts from Euler angles to a DCM.
 *
 * This function calculates a `dcm` that rotates a vector from frame B to frame A
 * (\f$\textbf{C}_\text{B}^\text{A}\f$), when provided the yaw, pitch, and roll that correspond to a
 * 3-2-1 frame rotation sequence frame A to frame B as follows:
 *
 * 1. Rotate frame A about third axis of frame A by value of yaw to yield intermediate frame 1
 * 2. Rotate intermediate frame 1 about its second axis by value of pitch to yield intermediate
 * frame 2
 * 3. Rotate intermediate frame 2 about its first axis by value of roll to yield frame B
 *
 * Example: It is common to use Euler angles to describe the attitude of an aircraft relative to a
 * local-level NED frame. In this case, "frame A" is the NED frame. Define the aircraft axes as
 * nose-right wing-down (through the belly). Note that if the aircraft is flying straight and level
 * heading north, then its axes are exactly aligned with the NED axes. To rotate from the NED axes
 * to the actual aircraft attitude, imagine the aircraft starting aligned with NED, perform a yaw
 * rotation about the down axis, a pitch rotation about (new) aircraft pitch axis, and then a roll
 * rotation about the (new) aircraft roll axis. This is a commonly-used approach for using
 * yaw/pitch/roll Euler angles to describe attitude of moving vehicles. In this example, the
 * output `dcm` matrix is \f$\textbf{C}_\text{P}^\text{NED}\f$, where \f$\text{P}\f$ is the aircraft
 * ("Platform") attitude.
 *
 * Note that the yaw, pitch, and roll as defined above are expressed in a vector with the ordering
 * [roll, pitch, yaw].
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for more details about
 * DCM and Euler angle expressions of attitude.
 * @see dcm_to_rpy() for the companion function that converts in the opposite direction.
 *
 * @tparam S Type of rpy, Vector by default.
 * @tparam std::enable_if_t<> Constrains S to be 1 dimensional.
 *
 * @param rpy Euler angles [roll pitch yaw] (radians) that describe a frame rotation from frame A to
 * frame B.  Can accept initializer lists.
 *
 * @return Direction cosine matrix (\f$\textbf{C}_\text{B}^\text{A}\f$).
 */
template <typename S = Vector, IfTensorOfDim<S, 1>* = nullptr>
Matrix3 rpy_to_dcm(const S& rpy) {
	Scalar cph = cos(rpy[0]);
	Scalar sph = sin(rpy[0]);
	Scalar cth = cos(rpy[1]);
	Scalar sth = sin(rpy[1]);
	Scalar cps = cos(rpy[2]);
	Scalar sps = sin(rpy[2]);
	return Matrix{{cps * cth, -sps * cph + cps * sth * sph, sps * sph + cps * sth * cph},
	              {sps * cth, cps * cph + sps * sth * sph, -cps * sph + sps * sth * cph},
	              {-sth, cth * sph, cth * cph}};
}

/**
 * This is a batched version of `rpy_to_dcm`.
 *
 * @overload
 *
 * @see rpy_to_dcm
 *
 * @tparam B Type of rpy, Matrix by default.
 * @tparam std::enable_if_t<> Constrains B to be 2 dimensional.
 *
 * @param rpy The Euler angle vectors, shape (N, 3).  Can accept initializer lists.
 *
 * @return The equivalent dcms, shape (N, 3, 3).
 */
template <typename B = Matrix, IfTensorOfDim<B, 2>* = nullptr>
Tensor<3> rpy_to_dcm(const B& rpy) {
	const size_t N = rpy.shape()[0];

	Tensor<3> dcms = empty(N, 3, 3);

	Scalar cph, sph, cth, sth, cps, sps;

	for (size_t i = 0; i < N; i++) {

		// Negate angles as we are performing
		// left-handed coordinate rotations
		cph = cos(rpy(i, 0));
		sph = sin(rpy(i, 0));
		cth = cos(rpy(i, 1));
		sth = sin(rpy(i, 1));
		cps = cos(rpy(i, 2));
		sps = sin(rpy(i, 2));


		dcms(i, 0, 0) = cps * cth;
		dcms(i, 0, 1) = -sps * cph + cps * sth * sph;
		dcms(i, 0, 2) = sps * sph + cps * sth * cph;
		dcms(i, 1, 0) = sps * cth;
		dcms(i, 1, 1) = cps * cph + sps * sth * sph;
		dcms(i, 1, 2) = -cps * sph + sps * sth * cph;
		dcms(i, 2, 0) = -sth;
		dcms(i, 2, 1) = cth * sph;
		dcms(i, 2, 2) = cth * cph;
	}

	return dcms;
}

/**
 * Converts from Euler angles to a quaternion.
 *
 * This function calculates a `quat` that rotates a vector from frame B to frame A
 * (\f$\textbf{q}_\text{B}^\text{A}\f$), when provided the yaw, pitch, and roll that correspond to a
 * 3-2-1 frame rotation from frame A to frame B as described in `rpy_to_dcm`.
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html)  for more details about
 * quaternion and Euler angle expressions of attitude.
 * @see quat_to_rpy() for the companion function that converts in the opposite direction.
 *
 * @tparam S Type of rpy, Vector by default.
 * @tparam std::enable_if_t<> Constrains S to be 1 dimensional.
 *
 * @param rpy Euler angles [roll pitch yaw] (radians) that describe a frame rotation from frame A to
 * frame B.  Can accept initializer lists.
 *
 * @return Quaternion (\f$\textbf{q}_\text{B}^\text{A}\f$)
 */
template <typename S = Vector, IfTensorOfDim<S, 1>* = nullptr>
Vector4 rpy_to_quat(const S& rpy) {
	auto cr = cos(rpy[0] / 2), cp = cos(rpy[1] / 2), cy = cos(rpy[2] / 2), sr = sin(rpy[0] / 2),
	     sp = sin(rpy[1] / 2), sy = sin(rpy[2] / 2);

	return {cr * cp * cy + sr * sp * sy,
	        sr * cp * cy - cr * sp * sy,
	        cr * sp * cy + sr * cp * sy,
	        cr * cp * sy - sr * sp * cy};
}

/**
 * This is a batched version of `rpy_to_quat`.
 *
 * @overload
 *
 * @see rpy_to_quat
 *
 * @tparam B Type of rpy, Matrix by default.
 * @tparam std::enable_if_t<> Constrains B to be 2 dimensional.
 *
 * @param rpy The Euler angle vectors, shape (N, 3).  Can accept initializer lists.
 *
 * @return The equivalent quaternions, shape (N, 4).
 */
template <typename B = Matrix, IfTensorOfDim<B, 2>* = nullptr>
Matrix rpy_to_quat(const B& rpy) {
	const size_t N = rpy.shape()[0];

	// allocate return Matrix
	Matrix quats = empty(N, 4);

	Scalar cr, cp, cy, sr, sp, sy;

	for (size_t i = 0; i < N; i++) {
		cr = cos(rpy(i, 0) / 2);
		cp = cos(rpy(i, 1) / 2);
		cy = cos(rpy(i, 2) / 2);
		sr = sin(rpy(i, 0) / 2);
		sp = sin(rpy(i, 1) / 2);
		sy = sin(rpy(i, 2) / 2);

		quats(i, 0) = cr * cp * cy + sr * sp * sy;
		quats(i, 1) = sr * cp * cy - cr * sp * sy;
		quats(i, 2) = cr * sp * cy + sr * cp * sy;
		quats(i, 3) = cr * cp * sy - sr * sp * cy;
	}

	return quats;
}

/**
 * Generates a DCM \f$\textbf{C}_\text{ENU}^\text{N}\f$ that can be used to rotate a vector
 * from the ENU frame to the N frame given a specified wander angle.
 * When `wander` = 0 this is equivalent to the identity matrix.
 *
 * An arbitrary vector \f$\textbf{w}\f$ expressed in the ENU frame can be rotated to the N frame
 * according to
 *
 * \f$\textbf{w}^\text{N}=\textbf{C}_\text{ENU}^\text{N}\textbf{w}^\text{ENU}\f$
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for the definitions of the ENU
 * and N frames, and how the wander angle is defined.
 * @see wander_to_C_ned_to_n()
 * @see wander_to_C_ned_to_l()
 *
 * @param wander Wander angle (radians)
 *
 * @return The DCM that rotates a vector from the ENU to N frame:
 * \f$\textbf{C}_\text{ENU}^\text{N}\f$.
 */
Matrix3 wander_to_C_enu_to_n(double wander);

/**
 * Generates a DCM \f$\textbf{C}_\text{NED}^\text{N}\f$ that can be used to rotate a vector
 * from the NED frame to the N frame given a specified wander angle.
 * When `wander` = 0 this is equivalent to \f$\textbf{C}_\text{NED}^\text{ENU}\f$.
 *
 * An arbitrary vector \f$\textbf{w}\f$ expressed in the NED frame can be rotated to the N frame
 * according to
 *
 * \f$\textbf{w}^\text{N}=\textbf{C}_\text{NED}^\text{N}\textbf{w}^\text{NED}\f$
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for the definitions of the NED
 * and N frames, and how the wander angle is defined.
 * @see wander_to_C_enu_to_n()
 * @see wander_to_C_ned_to_l()
 *
 * @param wander Wander angle (radians)
 *
 * @return The DCM that rotates a vector from the NED to N frame:
 * \f$\textbf{C}_\text{NED}^\text{N}\f$.
 */
Matrix3 wander_to_C_ned_to_n(double wander);

/**
 * Generates a DCM \f$\textbf{C}_\text{NED}^\text{L}\f$ that can be used to rotate a vector
 * from the NED frame to the L frame given a specified wander angle.
 * When `wander` = 0 this is equivalent to the identity matrix.
 *
 * An arbitrary vector \f$\textbf{w}\f$ expressed in the NED frame can be rotated to the L frame
 * according to
 *
 * \f$\textbf{w}^\text{L}=\textbf{C}_\text{NED}^\text{L}\textbf{w}^\text{NED}\f$
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for the definitions of the NED
 * and L frames, and how the wander angle is defined.
 * @see wander_to_C_ned_to_n()
 * @see wander_to_C_enu_to_n()
 *
 * @param wander Wander angle (radians)
 *
 * @return The DCM that rotates a vector from the NED to L frame:
 * \f$\textbf{C}_\text{NED}^\text{L}\f$,
 */
Matrix3 wander_to_C_ned_to_l(double wander);

/**
 * Generates a DCM \f$\textbf{C}_\text{N}^\text{E}\f$ that can be used to rotate a vector
 * from the N frame to the E frame given a specified latitude, longitude, and wander angle.
 * When `wander` = 0 this is equivalent to \f$\textbf{C}_\text{ENU}^\text{E}\f$. When
 * `wander` = 0 and `lat` = 0 and `lon` = 0, this is equivalent to the identity matrix.
 *
 * An arbitrary vector \f$\textbf{w}\f$ expressed in the N frame can be rotated to the E frame
 * according to
 *
 *  \f$\textbf{w}^\text{E}=\textbf{C}_\text{N}^\text{E}\textbf{w}^\text{N}\f$
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for the definitions of the N
 * and E frames, and how the wander angle is defined.
 *
 * @see C_n_to_e_h_to_llh()
 * @see C_n_to_e_h_to_ecef()
 * @see C_n_to_e_to_wander()
 * @see C_n_to_e_to_lat_lon_wander()
 * @see ecef_wander_to_C_n_to_e_h()
 *
 * @param lat latitude (radians)
 * @param lon longitude (radians)
 * @param wander Wander angle (radians)
 *
 * @return The DCM that rotates a vector from the N to E frame: \f$\textbf{C}_\text{N}^\text{E}\f$.
 */
Matrix3 lat_lon_wander_to_C_n_to_e(double lat, double lon, double wander = 0.0);

/**
 * Extract the latitude, longitude and wander angle from the N frame to
 * E frame DCM \f$\textbf{C}_\text{N}^\text{E}\f$.
 *
 * When latitude is at or very close to
 * +/- pi/2, both longitude and wander angle are set to 0 (as they
 * become undefined). Latitude will be between -pi/2 and pi/2.
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for
 * the definitions of the N and E frames, and how the wander angle is defined. This tutorial
 * also shows how the latitude, longitude, and wander angle are
 * calculated from the \f$\textbf{C}_\text{N}^\text{E}\f$ DCM.
 * @see lat_lon_wander_to_C_n_to_e() for the companion function that converts in the opposite
 * direction.
 * @see C_n_to_e_h_to_llh()
 * @see C_n_to_e_h_to_ecef()
 * @see C_n_to_e_to_wander()
 * @see ecef_wander_to_C_n_to_e_h()
 * @param C_n_to_e The DCM that rotates a vector from the N to E frame:
 * \f$\textbf{C}_\text{N}^\text{E}\f$
 *
 * @return A 3-length vector that contains the latitude, longitude and wander angle (radians).
 */
Vector3 C_n_to_e_to_lat_lon_wander(const Matrix& C_n_to_e);

/**
 * Extract the wander angle from the N frame to E frame DCM \f$\textbf{C}_\text{N}^\text{E}\f$.
 *
 * When latitude is at or very close to +/- pi/2, the wander angle is set to 0 (as it becomes
 * undefined).
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for the definitions of the N
 * and E frames and how the wander angle is defined. This tutorial also shows how the wander angle
 * is calculated from the \f$\textbf{C}_\text{N}^\text{E}\f$ DCM.
 *
 * @see C_n_to_e_to_lat_lon_wander() for a function that also returns latitude and longitude
 * @see C_n_to_e_h_to_llh()
 * @see C_n_to_e_h_to_ecef()
 * @see ecef_wander_to_C_n_to_e_h()
 * @see lat_lon_wander_to_C_n_to_e()
 *
 * @param C_n_to_e The DCM that rotates a vector from the N to E frame:
 * \f$\textbf{C}_\text{N}^\text{E}\f$
 *
 * @return Wander angle (radians).
 */
double C_n_to_e_to_wander(const Matrix3& C_n_to_e);

/**
 * Generates position expressed as \f$\textbf{C}_\text{N}^\text{E}\f$, \f$ h\f$ given an ECEF
 * position and (optional) wander angle. The height \f$ h\f$ is the height above the WGS-84
 * ellipsoid, and an arbitrary vector \f$\textbf{w}\f$ expressed in the N frame can be rotated to
 * the E frame according to
 *
 *  \f$\textbf{w}^\text{E}=\textbf{C}_\text{N}^\text{E}\textbf{w}^\text{N}\f$
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for
 * the definitions of the ECEF, N and E frames and how the wander angle is defined. This tutorial
 * also describes how position is espressed as \f$\textbf{C}_\text{N}^\text{E}\f$ \f$, h\f$.
 * @see C_n_to_e_h_to_llh()
 * @see C_n_to_e_h_to_ecef()
 * @see C_n_to_e_to_wander()
 * @see C_n_to_e_to_lat_lon_wander()
 * @see lat_lon_wander_to_C_n_to_e()
 *
 * @param ecef_pos ECEF position vector (meters)
 *
 * @param wander (optional) Wander angle (radians)
 *
 * @return A pair containing \f$\textbf{C}_\text{N}^\text{E}\f$ (unitless) and \f$ h\f$ (meters).
 */
std::pair<Matrix3, double> ecef_wander_to_C_n_to_e_h(const Vector3& ecef_pos, double wander = 0.0);

/**
 * Generates position expressed as a vector of latitude, longitude, and height (\f$\textbf{llh}\f$)
 * given position represented as \f$\textbf{C}_\text{N}^\text{E}\f$ and \f$ h\f$.
 * The height \f$ h\f$ is the height above the WGS-84 ellipsoid, and an arbitrary vector
 * \f$\textbf{w}\f$ expressed in the N frame can be rotated to the E frame according to
 *
 *  \f$\textbf{w}^\text{E}=\textbf{C}_\text{N}^\text{E}\textbf{w}^\text{N}\f$
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for the definitions of the
 * ECEF, N and E frames and how the wander angle is defined. This tutorial also describes how
 * position is espressed as \f$\textbf{C}_\text{N}^\text{E}\f$, \f$h\f$.
 *
 * @param C_n_to_e The DCM that rotates a vector from the N to E frame:
 * \f$\textbf{C}_\text{N}^\text{E}\f$
 * @param h Ellipsoidal height (meters)
 * @see C_n_to_e_h_to_ecef()
 * @see C_n_to_e_to_wander()
 * @see C_n_to_e_to_lat_lon_wander()
 * @see ecef_wander_to_C_n_to_e_h()
 * @see lat_lon_wander_to_C_n_to_e()
 *
 * @return Size 3 Vector containing latitude, longitude, and ellipsoidal height (radians, radians,
 * meters).
 */
Vector3 C_n_to_e_h_to_llh(const Matrix3& C_n_to_e, double h);

/**
 * Generates ECEF position vector
 * given position represented as \f$\textbf{C}_\text{N}^\text{E}\f$ and \f$ h\f$.
 * The height \f${h}\f$ is the height above the WGS-84 ellipsoid, the DCM
 * \f$\textbf{C}_\text{N}^\text{E}\f$ relates the N and E frame such that and an arbitrary vector
 * \f$\textbf{w}\f$ expressed in the N frame can be rotated to the E frame according to
 *
 *  \f$\textbf{w}^\text{E}=\textbf{C}_\text{N}^\text{E}\textbf{w}^\text{N}\f$
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for the definitions of the
 * ECEF, N and E frames and how the wander angle is defined. This tutorial also describes how
 * position is espressed as \f$\textbf{C}_\text{N}^\text{E}\f$\f$, h\f$.
 * @see C_n_to_e_h_to_llh()
 * @see C_n_to_e_to_wander()
 * @see C_n_to_e_to_lat_lon_wander()
 * @see ecef_wander_to_C_n_to_e_h()
 * @see lat_lon_wander_to_C_n_to_e()
 *
 * @param C_n_to_e The DCM that rotates a vector from the N to E frame:
 * \f$\textbf{C}_\text{N}^\text{E}\f$
 * @param h Ellipsoidal height (meters)
 *
 * @return Size 3 ECEF position vector (meters).
 */
Vector3 C_n_to_e_h_to_ecef(const Matrix3& C_n_to_e, double h);

/**
 * Generates a DCM \f$\textbf{C}_\text{ECEF}^\text{E}\f$ that can be used to rotate a vector
 * from the ECEF frame to the E frame. This matrix is a constant and has the values
 *
 * \f$   \textbf{C}_\text{ECEF}^\text{E} =
   \begin{bmatrix}
   0 & 1 & 0\\
   0 & 0 & 1\\
   1 & 0 & 0
   \end{bmatrix} \f$
 *
 * An arbitrary vector \f$\textbf{w}\f$ expressed in the ECEF frame can be rotated to the E frame
 * according to
 *
 *  \f$\textbf{w}^\text{E}=\textbf{C}_\text{ECEF}^\text{E}\textbf{w}^\text{ECEF}\f$
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for the definitions of the ECEF
 and E frames.
 *
 * @return \f$\textbf{C}_\text{ECEF}^\text{E}\f$ DCM.
 */
Matrix3 C_ecef_to_e();

/**
 * Conversion from rotation vector to direction cosine matrix using
 * truncated series approach. See eq 3.2.2.1-8 and 3.2.2.1-9
 * (or 7.1.1.1-3) in Savage, or equivalently eq.11.4-11.1 in T+W 2nd ed.
 *
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for more details on the
 * rotation vector and DCM representations of attitude.
 *
 * @tparam S Type of phi, Vector by default.
 * @tparam std::enable_if_t<> Constrains S to be 1 dimensional.
 *
 * @param phi A rotation vector such that rotating the `A` frame about \p phi yields the `B` frame,
 * shape (3).  Can accept an initializer list.
 *
 * @return `dcm` Direction cosine matrix from `B` to `A`, that is the DCM that rotates a vector
 * expressed in the `B` frame into the `A` frame, shape (3, 3).
 */
template <typename S = Vector, IfTensorOfDim<S, 1>* = nullptr>
Matrix3 rot_vec_to_dcm(const S& phi) {
	/*
	// 'Normal' implementation. Long chains of xtensor related operations can cause large slowdowns,
	// especially w/ ASAN testing, so actual implementation does math 'manually' to achieve speed.
	 double phi_mag   = xt::norm_l2(phi)[0];
	 double phi_mag2  = phi_mag * phi_mag;
	 double phi_mag4  = phi_mag2 * phi_mag2;
	 double term1     = 1 - phi_mag2 / 6 + phi_mag4 / 120;
	 double term2     = 0.5 - phi_mag2 / 24 + phi_mag4 / 720;
	 auto phi_cross = skew(phi);
	 return eye(3) + term1 * phi_cross + term2 * dot(phi_cross, phi_cross);
	 */
	auto p0         = phi(0);
	auto p1         = phi(1);
	auto p2         = phi(2);
	double phi_mag  = sqrt(p0 * p0 + p1 * p1 + p2 * p2);
	double phi_mag2 = phi_mag * phi_mag;
	double phi_mag4 = phi_mag2 * phi_mag2;
	double t1       = 1 - phi_mag2 / 6 + phi_mag4 / 120;
	double t2       = 0.5 - phi_mag2 / 24 + phi_mag4 / 720;

	return {{1.0 + t2 * (-p2 * p2 - p1 * p1), -t1 * p2 + t2 * p1 * p0, t1 * p1 + t2 * p0 * p2},
	        {t1 * p2 + t2 * p0 * p1, 1.0 + t2 * (-p2 * p2 - p0 * p0), -t1 * p0 + t2 * p1 * p2},
	        {-t1 * p1 + t2 * p0 * p2, t1 * p0 + t2 * p1 * p2, 1.0 + t2 * (-p1 * p1 - p0 * p0)}};
}

/**
 * Batched version of `rot_vec_to_dcm`.
 *
 * @overload
 *
 * @see rot_vec_to_dcm
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for more details on the
 * rotation vector and DCM representations of attitude.
 *
 * @tparam B Type of input, Matrix by default.
 * @tparam std::enable_if_t<> Constrains B to be 2 dimensional.
 *
 * @param phi Rotation vectors such that rotating the `A` frame about \p phi yields the `B` frame,
 * shape (N, 3).  Can accept an initializer list.
 *
 * @return `dcms` Direction cosine matrices from `B` to `A`, that is the DCM that rotates a vector
 * expressed in the `B` frame into the `A` frame, shape (N, 3, 3)
 */
template <typename B = Matrix, IfTensorOfDim<B, 2>* = nullptr>
Tensor<3> rot_vec_to_dcm(const B& phi) {
	const size_t N = phi.shape()[0];

	Tensor<3> dcms = empty(N, 3, 3);

	Scalar p0, p1, p2, phi_mag, phi_mag2, phi_mag4, t1, t2;

	for (size_t i = 0; i < N; i++) {
		p0       = phi(i, 0);
		p1       = phi(i, 1);
		p2       = phi(i, 2);
		phi_mag  = sqrt(p0 * p0 + p1 * p1 + p2 * p2);
		phi_mag2 = phi_mag * phi_mag;
		phi_mag4 = phi_mag2 * phi_mag2;
		t1       = 1 - phi_mag2 / 6 + phi_mag4 / 120;
		t2       = 0.5 - phi_mag2 / 24 + phi_mag4 / 720;

		dcms(i, 0, 0) = 1.0 + t2 * (-p2 * p2 - p1 * p1);
		dcms(i, 0, 1) = -t1 * p2 + t2 * p1 * p0;
		dcms(i, 0, 2) = t1 * p1 + t2 * p0 * p2;
		dcms(i, 1, 0) = t1 * p2 + t2 * p0 * p1;
		dcms(i, 1, 1) = 1.0 + t2 * (-p2 * p2 - p0 * p0);
		dcms(i, 1, 2) = -t1 * p0 + t2 * p1 * p2;
		dcms(i, 2, 0) = -t1 * p1 + t2 * p0 * p2;
		dcms(i, 2, 1) = t1 * p0 + t2 * p1 * p2;
		dcms(i, 2, 2) = 1.0 + t2 * (-p1 * p1 - p0 * p0);
	}
	return dcms;
}

/**
 * Get geoid undulation, equal to Mean Sea Level (MSL) elevation minus Height Above Ellipsoid (HAE),
 * at the given coordinates.
 *
 * @param latitude The latitude in radians at which to get the elevation.
 * @param longitude The longitude in radians at which to get the elevation.
 * @param path the path to the geoid undulation file for converting between HAE and
 * MSL. The default path of this variable requires setting the NAVTK_DATA_DIR environment
 * variable to the folder containing the undulation file. *
 *
 * @return A `pair` showing whether a height was found (`.first`) and if `true`, the offset between
 * MSL and HAE altitude in meters (`.second`).
 */
std::pair<bool, double> geoid_minus_ellipsoid(double latitude,
                                              double longitude,
                                              const std::string& path = "WW15MGH.GRD");

/**
 * Convert from Height Above Ellipsoid (HAE) to Mean Sea Level (MSL) elevation, using the difference
 * between the two at the given coordinates.
 *
 * Taken from Section 2.4.1.2 of Shockley PhD Dissertation: Ground Vehicle Navigation Using Magnetic
 * Field Variation, 2012 (available on dtic.mil)
 *
 * @param hae The HAE elevation in meters elevation at the given coordinates.
 * @param latitude The latitude in radians at which to get the elevation.
 * @param longitude The longitude in radians at which to get the elevation.
 * @param path the path to the geoid undulation file for converting between HAE and
 * MSL. The default path of this variable requires setting the NAVTK_DATA_DIR environment
 * variable to the folder containing the undulation file, or setting the NAVTK_GEOID_UNDULATION_PATH
 * environment variable to the path of the file itself.
 * @return The MSL elevation in meters.
 */
std::pair<bool, double> hae_to_msl(double hae,
                                   double latitude,
                                   double longitude,
                                   const std::string& path = "WW15MGH.GRD");

/**
 * Convert from Mean Sea Level (MSL) to Height Above Ellipsoid (HAE) elevation, using the difference
 * between the two at the given coordinates.
 *
 * @param msl The MSL elevation in meters at the given coordinates.
 * @param latitude The latitude in radians at which to get the elevation.
 * @param longitude The longitude in radians at which to get the elevation.
 * @param path the path to the geoid undulation file for converting between HAE and
 * MSL. The default path of this variable requires setting the NAVTK_DATA_DIR environment
 * variable to the folder containing the undulation file, or setting the NAVTK_GEOID_UNDULATION_PATH
 * environment variable to the path of the file itself.
 * @return The HAE elevation in meters.
 */
std::pair<bool, double> msl_to_hae(double msl,
                                   double latitude,
                                   double longitude,
                                   const std::string& path = "WW15MGH.GRD");

}  // namespace navutils
}  // namespace navtk
