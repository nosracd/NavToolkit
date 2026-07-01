#pragma once

#include <navtk/factory.hpp>
#include <navtk/inspect.hpp>
#include <navtk/navutils/navigation.hpp>
#include <navtk/tensors.hpp>

namespace navtk {
namespace navutils {

/**
 * Tags for gravity models to use.
 */
enum class GravModels {
	/** Use the gravity model implemented in calculate_gravity_titterton() */
	TITTERTON,
	/** Use the gravity model implemented in calculate_gravity_schwartz() */
	SCHWARTZ,
	/** Use the gravity model implemented in one of the `calculate_gravity_savage`
	 * functions (such as calculate_gravity_savage_ned()).
	 */
	SAVAGE
};

/**
 * NED frame gravity from Titterton provided gravity model (eq 3.8-3.91 in 2nd ed.).
 * Only the down term is provided (no deflections included).
 *
 * @param alt Ellipsoidal altitude (meters)
 * @param lat Latitude (radians)
 * @param R0 Mean radius of curvature, given just after eq. 3.86 in T+W 2nd ed., meters.
 *
 * @return Gravity at the given latitude and altitude.
 */
Vector3 calculate_gravity_titterton(double alt, double lat, double R0);

/**
 * NED frame gravity from Schwartz gravity model.
 * Only the down term is provided (no deflections included).
 *
 * @tparam S1 Type of alt.
 * @tparam S2 Type of lat.
 * @tparam std::enable_if_t<> Constrains S1 and S2 to be 0 dimensional.
 *
 * @param alt Ellipsoidal altitude (meters)
 * @param lat Latitude (radians)
 *
 * @return Gravity at the given latitude and altitude.
 */
template <typename S1, typename S2, IfBothTensorsOfDim<S1, S2, 0>* = nullptr>
Vector3 calculate_gravity_schwartz(const S1& alt, const S2& lat) {
	// Apply gravity model from K.P. Schwartz ENGO 623 Course Notes (University of Calgary)
	// This is the GRS80 gravity model from
	// http://geoweb.mit.edu/~tah/12.221_2005/grs80_corr.pdf
	// combined with the height variation correction given in
	// https://archive.org/details/HeiskanenMoritz1967PhysicalGeodesy/page/n87
	// eq 2-123
	const double a1 = 9.7803267715;
	const double a2 = 0.0052790414;
	const double a3 = 0.0000232718;
	const double a4 = -3.0876910891E-6;
	const double a5 = 4.3977311E-9;
	const double a6 = 7.211E-13;
	auto sin_lat    = sin(lat);
	auto sin2_lat   = sin_lat * sin_lat;
	auto sin4_lat   = sin2_lat * sin2_lat;

	double g = 0.0;
	if (alt >= 0)
		g = a1 * (1 + a2 * sin2_lat + a3 * sin4_lat) + (a4 + a5 * sin2_lat) * alt + a6 * alt * alt;
	else {
		// Original does not account for negative altitudes- implement
		// linear scaling of gravity at altitude 0 as recommended in Savage
		// 5.4-2 and done elsewhere (T+W)
		auto g0  = a1 * (1 + a2 * sin2_lat + a3 * sin4_lat);
		auto R_N = meridian_radius(lat);
		auto R_E = transverse_radius(lat);
		auto R_0 = sqrt(R_N * R_E);
		g        = g0 * (1 + alt / R_0);
	}
	return {0.0, 0.0, g};
}

/**
 * Batch version of `calculate_gravity_schwartz`.
 *
 * @overload
 *
 * @see calculate_gravity_schwartz
 *
 * @tparam B1 Type of alt.
 * @tparam B2 Type of lat.
 * @tparam std::enable_if_t<> Constrains B1 and B2 to have a maximum dimension of precisely 1.
 *
 * @param alt Ellipsoidal altitudes (meters), shape (N).  Can accept initializer lists or a
 * `double` as well.
 * @param lat Latitudes (radians), shape (N).  Can accept initializer lists or a
 * `double` as well.
 *
 * @return Gravities at the given latitude and altitude, shape (N, 3)
 */
template <typename B1 = Vector, typename B2 = Vector, IfTensorsMaxDim<1, B1, B2>* = nullptr>
Matrix calculate_gravity_schwartz(const B1& alt, const B2& lat) {
	auto alt_vec = to_vec(alt);
	auto lat_vec = to_vec(lat);
	// Apply gravity model from K.P. Schwartz ENGO 623 Course Notes (University of Calgary)
	// This is the GRS80 gravity model from
	// http://geoweb.mit.edu/~tah/12.221_2005/grs80_corr.pdf
	// combined with the height variation correction given in
	// https://archive.org/details/HeiskanenMoritz1967PhysicalGeodesy/page/n87
	// eq 2-123
	const size_t N = std::max(alt_vec.shape()[0], lat_vec.shape()[0]);

	Matrix gravity = empty(N, 3);

	const double a1 = 9.7803267715;
	const double a2 = 0.0052790414;
	const double a3 = 0.0000232718;
	const double a4 = -3.0876910891E-6;
	const double a5 = 4.3977311E-9;
	const double a6 = 7.211E-13;
	Scalar sin_lat, sin2_lat, sin4_lat, g0, r_0, altitude;

	// is it better to run these in a batch, or call with the single versions
	// of `meridian_radius` and `transverse_radius`, since they won't always
	// be needed?
	auto R_Ns = meridian_radius(lat_vec);
	auto R_Es = transverse_radius(lat_vec);

	for (size_t i = 0; i < N; i++) {
		gravity(i, 0) = 0;
		gravity(i, 1) = 0;

		altitude = alt_vec(i);

		sin_lat  = sin(lat_vec(i));
		sin2_lat = sin_lat * sin_lat;
		sin4_lat = sin2_lat * sin2_lat;

		if (altitude >= 0)
			gravity(i, 2) = a1 * (1 + a2 * sin2_lat + a3 * sin4_lat) +
			                (a4 + a5 * sin2_lat) * altitude + a6 * altitude * altitude;
		else {
			// Original does not account for negative altitudes- implement
			// linear scaling of gravity at altitude 0 as recommended in Savage
			// 5.4-2 and done elsewhere (T+W)
			g0            = a1 * (1 + a2 * sin2_lat + a3 * sin4_lat);
			r_0           = sqrt(R_Ns(i) * R_Es(i));
			gravity(i, 2) = g0 * (1 + altitude / r_0);
		}
	}

	return gravity;
}

/**
 * Gravity Vector from the 'Savage' gravity model as given in Section 5.4
 * of Strapdown Analytics, 2nd ed., N frame version.
 *
 * @see `calculate_gravity_schwartz()`
 * @see `calculate_gravity_savage_ned()`
 * @see `lat_lon_wander_to_C_n_to_e()`
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for coordinate frame definitions.
 *
 * @param C_n_to_e DCM that rotates from the N frame
 * to the E frame \f$\textbf{C}_\text{N}^\text{E}\f$ (provides knowledge of horizontal position)
 * @param h Ellipsoidal altitude (meters)
 *
 * @return Gravity at the specified location expressed in the N frame
 */
Vector3 calculate_gravity_savage_n(const Matrix& C_n_to_e, double h);

/**
 * Gravity Vector from the 'Savage' gravity model as given in Section 5.4
 * of Strapdown Analytics, 2nd ed., NED frame version.
 *
 * @see `calculate_gravity_schwartz()`
 * @see `calculate_gravity_savage_n()`
 * @see `lat_lon_wander_to_C_n_to_e()`
 * @see [Coordinate Frames](../tutorial/coordinate_frames.html) for coordinate frame definitions.
 *
 * @param C_n_to_e DCM that rotates from the N frame
 * to the E frame \f$\textbf{C}_\text{N}^\text{E}\f$ (provides knowledge of horizontal position)
 * @param h Ellipsoidal altitude (meters)
 *
 * @return Gravity at the specified location expressed in the NED frame.
 */
Vector3 calculate_gravity_savage_ned(const Matrix& C_n_to_e, double h);
}  // namespace navutils
}  // namespace navtk
