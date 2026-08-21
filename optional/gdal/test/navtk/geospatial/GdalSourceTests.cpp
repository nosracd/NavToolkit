#include <memory>
#include <string>

#include <gtest/gtest.h>
#include <spdlog_assert.hpp>
#include <tensor_assert.hpp>

#include <navtk/errors.hpp>
#include <navtk/geospatial/providers/SimpleElevationProvider.hpp>
#include <navtk/geospatial/sources/GdalSource.hpp>
#include <navtk/navutils/math.hpp>
#include <navtk/navutils/navigation.hpp>
#include <navtk/tensors.hpp>

namespace navtk {
namespace geospatial {

using navtk::navutils::DEG2RAD;

// These tests demonstrate that the SimpleElevationProvider can properly abstract away any need to
// call the GdalSource methods. By adding multiple sources to the provider, the user can let the
// provider do all the interfacing between its collection of sources. These tests also verify that
// GdalSource can properly handle valid and invalid queries, and that specifying different vertical
// reference input and output frames will result in different outputs.
class GdalSourceTest : public ::testing::Test {
public:
	std::map<GdalSource::MapType, std::shared_ptr<GdalSource>> sources;

	GdalSourceTest() {
		char* map_path = getenv("NAVTK_DATA_DIR");

		spdlog::set_level(spdlog::level::debug);

		if (map_path == NULL) {
			log_or_throw(
			    "Environment variable 'NAVTK_DATA_DIR' not set. Must be set to directory "
			    "containing GDAL files.");
		} else {
			sources.emplace(GdalSource::MapType::GEOTIFF,
			                std::make_shared<GdalSource>(map_path,
			                                             GdalSource::MapType::GEOTIFF,
			                                             ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE,
			                                             ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE));
			sources.emplace(GdalSource::MapType::DTED,
			                std::make_shared<GdalSource>(map_path,
			                                             GdalSource::MapType::DTED,
			                                             ASPN_MEASUREMENT_ALTITUDE_REFERENCE_MSL,
			                                             ASPN_MEASUREMENT_ALTITUDE_REFERENCE_MSL));
		}
	}

	void test_valid_queries(GdalSource::MapType map_type,
	                        const Matrix& valid_coordinates,
	                        const Vector& expected_elevations) {
		auto source               = sources[map_type];
		auto resulting_elevations = zeros(num_rows(valid_coordinates));
		for (Size ii = 0; ii < num_rows(valid_coordinates); ++ii) {
			auto elevation = source->lookup_datum(valid_coordinates(ii, 0) * DEG2RAD,
			                                      valid_coordinates(ii, 1) * DEG2RAD);
			EXPECT_TRUE(elevation.first);
			resulting_elevations(ii) = elevation.second;
		}

		auto batch_elevations =
		    source->lookup_data(xt::view(valid_coordinates, xt::all(), 0) * DEG2RAD,
		                        xt::view(valid_coordinates, xt::all(), 1) * DEG2RAD);

		ASSERT_ALLCLOSE_EX(expected_elevations, resulting_elevations, 0.05, 0.0);
		ASSERT_ALLCLOSE_EX(expected_elevations, batch_elevations, 0.05, 0.0);
	}

	void test_invalid_queries(GdalSource::MapType map_type, const Matrix& invalid_coordinates) {
		auto source = sources[map_type];
		for (Size ii = 0; ii < num_rows(invalid_coordinates); ++ii) {
			auto elevation =
			    EXPECT_DEBUG(source->lookup_datum(invalid_coordinates(ii, 0) * DEG2RAD,
			                                      invalid_coordinates(ii, 1) * DEG2RAD),
			                 "not in known tiles");
			EXPECT_FALSE(elevation.first);
		}

		auto batch_elevations =
		    source->lookup_data(xt::view(invalid_coordinates, xt::all(), 0) * DEG2RAD,
		                        xt::view(invalid_coordinates, xt::all(), 1) * DEG2RAD);

		ASSERT_TRUE(xt::all(xt::isnan(batch_elevations)));
	}
};

// This test asserts that when reading a GeoTIFF file, the gdal elevation source class returns the
// same elevation for a given set of latitudes and longitudes as the GDAL command line utility,
// gdallocationinfo. Since gdallocationinfo doesn't do interpolation, query coordinates have been
// selected such that the resulting pixel offset will be as close to integer values as
// possible. The primary error source is a small amount of interpolation that is still occurring in
// the results but not in the expected results.
TEST_F(GdalSourceTest, geotiff_compare_against_command_line_gdal) {
	// clang-format off
	// Series of arbitrary coordinates inside the test tile. Each row is {latitude, longitude} in
	// degrees.
	const Matrix QUERY_COORDINATES = {{-3.75989522, -78.94605278},
									  {-3.703023681, -79.061434681},
									  {-3.532207830, -78.857118290},
									  {-3.761989244, -78.861024443}};
	// clang-format on

	// The resulting elevations interpolated between the nearest pixels manually.
	const Vector EXPECTED_ELEVATIONS = {41.25, 141.75, 164.5, 70};

	test_valid_queries(GdalSource::MapType::GEOTIFF, QUERY_COORDINATES, EXPECTED_ELEVATIONS);
}

TEST_F(GdalSourceTest, geotiff_get_elevation_outside_of_tile) {
	// pick a location that does not have GeoTIFF coverage
	const Matrix QUERY_COORDINATES = {{1.1, 1.1}};

	test_invalid_queries(GdalSource::MapType::GEOTIFF, QUERY_COORDINATES);
}

// Check both vertical reference frames in a single test to verify that even after caching a tile,
// it can be transformed to a new frame.
TEST_F(GdalSourceTest, geotiff_change_frames_SLOW) {
	// Default reference frame is HAE, so change to MSL
	sources[GdalSource::MapType::GEOTIFF]->set_output_vertical_reference_frame(
	    ASPN_MEASUREMENT_ALTITUDE_REFERENCE_MSL);
	auto elevation = sources[GdalSource::MapType::GEOTIFF]->lookup_datum(-3.63817361 * DEG2RAD,
	                                                                     -78.90906349 * DEG2RAD);
	EXPECT_TRUE(elevation.first);
	EXPECT_NEAR(navutils::hae_to_msl(33, -3.63817361 * DEG2RAD, -78.90906349 * DEG2RAD).second,
	            elevation.second,
	            0.5);

	// Back to HAE
	sources[GdalSource::MapType::GEOTIFF]->set_output_vertical_reference_frame(
	    ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE);
	elevation = sources[GdalSource::MapType::GEOTIFF]->lookup_datum(-3.63817361 * DEG2RAD,
	                                                                -78.90906349 * DEG2RAD);
	EXPECT_TRUE(elevation.first);
	EXPECT_NEAR(33, elevation.second, 0.5);
}

// Ensure the same tests work on DTED. The DTED file's default vertical reference frame is MSL.
TEST_F(GdalSourceTest, dted_compare_against_command_line_gdal) {
	// West-Center of test file, just to make sure the file can be read.
	const Matrix QUERY_COORDINATES   = {{30.1666772, -81.8866530}};
	const Vector EXPECTED_ELEVATIONS = {27};

	test_valid_queries(GdalSource::MapType::DTED, QUERY_COORDINATES, EXPECTED_ELEVATIONS);
}

TEST_F(GdalSourceTest, dted_get_elevation_outside_of_tile) {
	// pick a location outside the DTED file
	const Matrix QUERY_COORDINATES = {{2.2, 2.2}};

	test_invalid_queries(GdalSource::MapType::DTED, QUERY_COORDINATES);
}

TEST_F(GdalSourceTest, dted_change_frames_SLOW) {
	// Default reference frame is MSL, so change to HAE
	sources[GdalSource::MapType::DTED]->set_output_vertical_reference_frame(
	    ASPN_MEASUREMENT_ALTITUDE_REFERENCE_MSL);
	auto elevation = sources[GdalSource::MapType::DTED]->lookup_datum(30.1666772 * DEG2RAD,
	                                                                  -81.8866530 * DEG2RAD);
	EXPECT_TRUE(elevation.first);
	EXPECT_NEAR(27, elevation.second, 0.5);

	// Back to MSL
	sources[GdalSource::MapType::DTED]->set_output_vertical_reference_frame(
	    ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE);
	elevation = sources[GdalSource::MapType::DTED]->lookup_datum(30.1666772 * DEG2RAD,
	                                                             -81.8866530 * DEG2RAD);
	EXPECT_TRUE(elevation.first);
	EXPECT_NEAR(navutils::msl_to_hae(27, 30.1666772 * DEG2RAD, -81.8866530 * DEG2RAD).second,
	            elevation.second,
	            0.5);
}

TEST_F(GdalSourceTest, map_path) {
	auto guard = ErrorModeLock(ErrorMode::DIE);

	// Pass no path, will load relative path and should throw because could not find GEOTIFF files
	EXPECT_THROW({ GdalSource source("", GdalSource::MapType::GEOTIFF); }, std::runtime_error);

	// Pass an invalid path, should throw because cannot open folder
	EXPECT_THROW(
	    { GdalSource source("some/bad/path", GdalSource::MapType::DTED); }, std::invalid_argument);
}

TEST_F(GdalSourceTest, internal_storage_meta) {
	EXPECT_TRUE(sources[GdalSource::MapType::GEOTIFF]->get_size() == 1);
	EXPECT_TRUE(sources[GdalSource::MapType::DTED]->get_size() == 1);

	std::string tif_path = getenv("NAVTK_DATA_DIR");
	tif_path += "/bogota.tif";

	std::string dted_path = getenv("NAVTK_DATA_DIR");
	dted_path += "/n30_w082.dt2";

	EXPECT_TRUE(sources[GdalSource::MapType::GEOTIFF]->is_stored(tif_path));
	EXPECT_TRUE(sources[GdalSource::MapType::DTED]->is_stored(dted_path));
}

TEST_F(GdalSourceTest, coordinate_transform) {
	// top left corner coordinates of example file
	auto lat_deg = -3.529274236;
	auto lon_deg = -79.1045964;

	Coordinate expected_coord = {440720.000, 100000.000};

	auto transformed_coord =
	    sources[GdalSource::MapType::GEOTIFF]->wgs84_to_map(lat_deg * DEG2RAD, lon_deg * DEG2RAD);

	EXPECT_NEAR(expected_coord.x, transformed_coord.x, 0.1);
	EXPECT_NEAR(expected_coord.y, transformed_coord.y, 0.1);
}

TEST_F(GdalSourceTest, cache_management) {
	EXPECT_EQ(sources[GdalSource::MapType::GEOTIFF]->get_cached_num(), 0);

	// invalid coordinates
	auto elevation =
	    sources[GdalSource::MapType::GEOTIFF]->lookup_datum(0.0 * DEG2RAD, 0.0 * DEG2RAD);

	EXPECT_FALSE(elevation.first);
	EXPECT_EQ(sources[GdalSource::MapType::GEOTIFF]->get_cached_num(), 0);

	// valid coordinates
	elevation = sources[GdalSource::MapType::GEOTIFF]->lookup_datum(-3.63817361 * DEG2RAD,
	                                                                -78.90906349 * DEG2RAD);

	EXPECT_TRUE(elevation.first);
	EXPECT_EQ(sources[GdalSource::MapType::GEOTIFF]->get_cached_num(), 1);
}

TEST_F(GdalSourceTest, unsupported_reference_frame) {
	char* map_path = getenv("NAVTK_DATA_DIR");
	auto map_type  = GdalSource::MapType::GEOTIFF;

	// init input reference to unsupported type
	EXPECT_WARN(GdalSource(map_path, map_type, ASPN_MEASUREMENT_ALTITUDE_REFERENCE_AGL),
	            "ASPN_MEASUREMENT_ALTITUDE_REFERENCE_AGL is unsupported");
	// init output reference to unsupported type
	EXPECT_WARN(

	    GdalSource(map_path,
	               map_type,
	               ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE,
	               ASPN_MEASUREMENT_ALTITUDE_REFERENCE_AGL),
	    "ASPN_MEASUREMENT_ALTITUDE_REFERENCE_AGL is unsupported");

	// try setting output reference to unsupported type
	EXPECT_WARN(sources[map_type]->set_output_vertical_reference_frame(
	                ASPN_MEASUREMENT_ALTITUDE_REFERENCE_AGL),
	            "ASPN_MEASUREMENT_ALTITUDE_REFERENCE_AGL is not supported");
}
}  // namespace geospatial
}  // namespace navtk
