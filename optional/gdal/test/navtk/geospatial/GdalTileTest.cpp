#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>

#include <navtk/errors.hpp>
#include <navtk/fs/filesystem.hpp>
#include <navtk/geospatial/Tile.hpp>
#include <navtk/tensor_assert.hpp>

namespace navtk {
namespace geospatial {

class TileTest : public testing::Test {
protected:
	void SetUp() override {

		auto path = getenv("NAVTK_DATA_DIR");

		if (path == NULL) {
			log_or_throw("NAVTK_DATA_DIR is not set.  Cannot create GDAL tile.");
		}

		std::string file = "";
		auto extension   = ".tif";

		auto map_path = std::string(path);

		auto absolute_map_path = fs::absolute(map_path);

		if (!map_path.empty()) {

			if (map_path[map_path.size() - 1] != fs::path::preferred_separator) {
				absolute_map_path = fs::absolute(map_path + fs::path::preferred_separator);
			}
		}

		// By default constructs an end iterator which will cause no paths to be searched
		fs::recursive_directory_iterator file_search_iterator;

		try {
			file_search_iterator = fs::recursive_directory_iterator(
			    absolute_map_path, fs::directory_options::follow_directory_symlink);
		} catch (fs::filesystem_error& e) {
			log_or_throw<std::invalid_argument>("{}", e.what());
		}

		for (const auto& entry : file_search_iterator) {
			fs::path filename = fs::path(entry.path());
			// Use `find` instead of `compare` to find extensions like `dt2`
			if (filename.filename().string().at(0) != '.' &&
			    filename.extension().string().find(extension) != std::string::npos) {

				file = filename;
			}
		}

		if (file == "") {
			log_or_throw<std::runtime_error>("No test data file found.");
		}

		tile = std::make_shared<Tile>(file);
	}
	std::shared_ptr<Tile> tile;
};

TEST_F(TileTest, is_valid) { ASSERT_TRUE(tile->is_valid()); }

TEST_F(TileTest, get_filename) {
	auto filename = tile->get_filename();
	ASSERT_TRUE(filename.substr(filename.length() - 10) == "bogota.tif");
}

TEST_F(TileTest, contains) {
	Coordinate coord_false = {0.0, 0.0};
	Coordinate coord_true  = {456080.000, 84640.000};

	ASSERT_FALSE(tile->contains(coord_false));
	ASSERT_TRUE(tile->contains(coord_true));
}

TEST_F(TileTest, contains_edge) {
	const double EPSILON = 1e-6;

	Coordinate coord_edge      = {440720.000, 100000.000};
	Coordinate coord_near_edge = {440720.000 + EPSILON, 100000.000 - EPSILON};

	ASSERT_FALSE(tile->contains(coord_edge));
	ASSERT_TRUE(tile->contains(coord_near_edge));
}

TEST_F(TileTest, lookup_edge) {
	// test what happens when we perform a lookup within the 1/2 pixel width boarder of the tile
	auto top_left_corner    = tile->pixel_to_map({0.25, 0.25});
	auto top_right_corner   = tile->pixel_to_map({tile->get_width() - 0.25, 0.25});
	auto bottom_left_corner = tile->pixel_to_map({0.25, tile->get_height() - 0.25});
	auto bottom_right_corner =
	    tile->pixel_to_map({tile->get_width() - 0.25, tile->get_height() - 0.25});

	auto top_left_pixel    = tile->pixel_to_map({0.5, 0.5});
	auto top_right_pixel   = tile->pixel_to_map({tile->get_width() - 0.5, 0.5});
	auto bottom_left_pixel = tile->pixel_to_map({0.5, tile->get_height() - 0.5});
	auto bottom_right_pixel =
	    tile->pixel_to_map({tile->get_width() - 0.5, tile->get_height() - 0.5});

	ASSERT_TRUE(tile->contains(top_left_corner));
	ASSERT_TRUE(tile->contains(top_right_corner));
	ASSERT_TRUE(tile->contains(bottom_left_corner));
	ASSERT_TRUE(tile->contains(bottom_right_corner));
	ASSERT_TRUE(tile->contains(top_left_pixel));
	ASSERT_TRUE(tile->contains(top_right_pixel));
	ASSERT_TRUE(tile->contains(bottom_left_pixel));
	ASSERT_TRUE(tile->contains(bottom_right_pixel));

	// the value of the lookup between the corner pixel and the corner of the tile should be the
	// value of the lookup right on the corner pixel
	ASSERT_EQ(tile->lookup_datum(top_left_corner), tile->lookup_datum(top_left_pixel));
	ASSERT_EQ(tile->lookup_datum(top_right_corner), tile->lookup_datum(top_right_pixel));
	ASSERT_EQ(tile->lookup_datum(bottom_left_corner), tile->lookup_datum(bottom_left_pixel));
	ASSERT_EQ(tile->lookup_datum(bottom_right_corner), tile->lookup_datum(bottom_right_pixel));

	auto boarder_of_tile   = tile->pixel_to_map({1.0, 0.25});
	auto boarder_of_pixels = tile->pixel_to_map({1.0, 0.5});

	// the value of the lookup on the 1/2 pixel width board, should be the value interpolated
	// between the two nearest pixels at the edge
	ASSERT_EQ(tile->lookup_datum(boarder_of_tile), tile->lookup_datum(boarder_of_pixels));
}

TEST_F(TileTest, scan_and_unload) {
	// just calls the functions...since the internals are private, there's not much to test!
	tile->scan_tile();
	tile->unload();
}

TEST_F(TileTest, dimension) {
	const size_t width  = 512;
	const size_t height = 512;

	ASSERT_EQ(tile->get_width(), width);
	ASSERT_EQ(tile->get_height(), height);
}

TEST_F(TileTest, lookup_datum) {
	double elevation_expected, elevation_result;

	Coordinate center_coord = tile->pixel_to_map({256, 256});
	// manually interpolated between the four pixel surrounding the center point
	elevation_expected = 143.75;

	elevation_result = tile->lookup_datum(center_coord);
	ASSERT_NEAR(elevation_result, elevation_expected, 1e-6);

	// map coordinate that correponds precisely to the pixel indices (9, 19)
	Coordinate point_on_pixel = tile->pixel_to_map({9.5, 19.5});
	// manually queried value at that pixel
	elevation_expected = 148.0;

	elevation_result = tile->lookup_datum(point_on_pixel);
	ASSERT_NEAR(elevation_result, elevation_expected, 1e-6);

	Coordinate far_corner =
	    tile->pixel_to_map({double(tile->get_width()), double(tile->get_height())});
	// elevation manually queried at the farthest pixel, since that is the best guess available for
	// the elevation at the farthest corner of the tile
	elevation_expected = 148.0;

	elevation_result = tile->lookup_datum(far_corner);
	ASSERT_NEAR(elevation_result, elevation_expected, 1e-6);

	// the two closest pixel indices to this point would be (199, 0) and (200, 0)
	Coordinate on_edge = tile->pixel_to_map({200.3, 0.0});
	// manually interpolated between the values at these two pixels
	elevation_expected = 237.0;

	elevation_result = tile->lookup_datum(on_edge);
	ASSERT_NEAR(elevation_result, elevation_expected, 1e-6);
}

}  // namespace geospatial
}  // namespace navtk
