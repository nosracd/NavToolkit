#include <navtk/navutils/navigation.hpp>

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <navtk/geospatial/sources/GeoidUndulationSource.hpp>
#include <navtk/linear_algebra.hpp>
#include <navtk/navutils/math.hpp>
#include <navtk/navutils/wgs84.hpp>

namespace navtk {
namespace navutils {


std::pair<bool, double> geoid_minus_ellipsoid(double latitude,
                                              double longitude,
                                              const std::string& path) {
	return navtk::geospatial::GeoidUndulationSource::get_shared(path)->lookup_datum(latitude,
	                                                                                longitude);
}

std::pair<bool, double> hae_to_msl(double hae,
                                   double latitude,
                                   double longitude,
                                   const std::string& path) {
	auto undulation = geoid_minus_ellipsoid(latitude, longitude, path);
	if (undulation.first) return {true, hae - undulation.second};
	return undulation;
}

std::pair<bool, double> msl_to_hae(double msl,
                                   double latitude,
                                   double longitude,
                                   const std::string& path) {
	auto undulation = geoid_minus_ellipsoid(latitude, longitude, path);
	if (undulation.first) return {true, msl + undulation.second};
	return undulation;
}

}  // namespace navutils
}  // namespace navtk
