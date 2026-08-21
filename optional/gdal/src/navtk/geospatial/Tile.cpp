#include <navtk/geospatial/Tile.hpp>

#include <iostream>
#include <memory>

#ifndef GDAL_INCLUDE_IN_SUBFOLDER
#	include <gdal_priv.h>
#	include <ogr_spatialref.h>
#else
#	include <gdal/gdal_priv.h>
#	include <gdal/ogr_spatialref.h>
#endif

#include <spdlog/spdlog.h>
#include <navtk/factory.hpp>
#include <navtk/geospatial/ElevationInterpolator.hpp>
#include <navtk/geospatial/detail/custom_deleters.hpp>
#include <navtk/geospatial/detail/transformations.hpp>

namespace navtk {
namespace geospatial {

Tile::Tile(const std::string& filename) : filename(filename) {
	GDALAllRegister();
	GDALDataset* gdal_handle = (GDALDataset*)GDALOpen(filename.c_str(), GA_ReadOnly);

	if (gdal_handle != NULL) {
		valid = true;

		double pixel_transform[6];

		dataset = std::unique_ptr<GDALDataset, detail::DatasetDelete>{gdal_handle};
		dataset->GetGeoTransform(pixel_transform);

		// The Geo Transform is structured like this:
		// pixel_transform[0] = upper left x coordinate
		// pixel_transform[1] = pixel width (number of map units (e.g. meters) / pixel)
		// pixel_transform[2] = row rotation, will be 0 for north up images (virtually every
		// dataset)
		// pixel_transform[3] = upper left y coordinate
		// pixel_transform[4] = column rotation, will be 0 for north up images (virtually every
		// dataset)
		// pixel_transform[5] = pixel height
		map_offset_x    = pixel_transform[0];
		map_offset_y    = pixel_transform[3];
		pixel_width     = pixel_transform[1];
		pixel_height    = pixel_transform[5];
		row_rotation    = pixel_transform[2];
		column_rotation = pixel_transform[4];

		no_data_value = dataset->GetRasterBand(1)->GetNoDataValue();

		size_x = dataset->GetRasterXSize();
		size_y = dataset->GetRasterYSize();

		// find the corners of the tile in map space
		bounds = get_bounds({
		    pixel_to_map(Coordinate(0, 0)),
		    pixel_to_map(Coordinate(size_x, 0)),
		    pixel_to_map(Coordinate(0, size_y)),
		    pixel_to_map(Coordinate(size_x, size_y)),
		});
	} else {
		spdlog::warn("Skipping {} because it is not a valid GDAL file.", filename);
	}
}

Tile::Bounds Tile::get_bounds(std::array<Coordinate, 4> corners) {
	std::sort(corners.begin(), corners.end(), [](const Coordinate& a, const Coordinate& b) {
		return (a.y < b.y) ? true : (a.x < b.x);
	});

	return {corners[0], corners[1], corners[2], corners[3]};
}

bool Tile::is_valid() const { return valid; }

bool Tile::is_rotated() const { return (row_rotation != 0.0) || (column_rotation != 0.0); }

bool Tile::is_cached() const { return cached; }

size_t Tile::get_width() const { return size_x; }

size_t Tile::get_height() const { return size_y; }

std::string Tile::get_filename() const { return filename; }

Coordinate Tile::map_to_pixel(const Coordinate& map_coords) const {
	// NOTE: we drop the rotation transform values, since rotated tiles are not supported
	return {(map_coords.x - map_offset_x) / pixel_width,
	        (map_coords.y - map_offset_y) / pixel_height};
}

Coordinate Tile::pixel_to_map(const Coordinate& pixel_coords) const {
	// NOTE: we drop the rotation transform values, since rotated tiles are not supported
	return {map_offset_x + pixel_width * pixel_coords.x,
	        map_offset_y + pixel_height * pixel_coords.y};
}

double Tile::read_pixel(const Pixel& idx) const {
	if (!cached) {
		scan_tile();
		cached = true;
	}

	auto cache_index = idx.y * size_x + idx.x;
	auto value       = cached_tile[cache_index];
	return (value != no_data_value) ? value : NAN;
}

double Tile::lookup_datum(const Coordinate& coord) const {
	/* Offset is the decimal distance from top left of tile in pixel widths.  According to the
	 documentation here
	 (https://gdal.org/en/stable/user/raster_data_model.html#affine-geotransform), this pixel offset
	 does not correspond directly to the pixel indices.  Rather, each pixel "lives" in the center of
	 a area of 1x1 in pixel space.  Thus, the pixel located at the 0th row and column is actually
	 located at (0.5, 0.5) in pixel space. This function, then, subtracts 0.5 from both offset
	 coordinates before clamping the result to be within the bounds of the pixel indices, to ensure
	 that accessing points on the 1/2 pixel rim of the tile doesn't lead to invalid indexing.  Then,
	 from the adjusted offset value, it gets the 4 closest pixel indices and interpolates to find
	 the elevation difference at provided map coordinates.

	 (idx.x,idx.y)***********(idx.x + 1,idx.y)
	 *****************************************
	 *****************************************
	 *****************************************
	 *****************************************
	 *****************************************
	 *****************************************
	 **(offset.x, offset.y)*******************
	 *****************************************
	 *****************************************
	 *****************************************
	 *****************************************
	 *****************************************
	 (idx.x,idx.y + 1)***(idx.x + 1,idx.y + 1)
	*/
	auto tile_offset = map_to_pixel(coord);
	// clamp x and y to be relative to actual pixel locations in the tile, rather than just pixel
	// space.  Since there is a 1/2 pixel width/height region between the actual pixel
	// coordinates and the "bounds" of the tile, this conversion will ensure that x and y can
	// validly be cast to a pixel index, and that their interpolation posts are the best
	// possible choices.
	auto x     = std::clamp(tile_offset.x - 0.5, 0.0, static_cast<double>(size_x - 1));
	auto y     = std::clamp(tile_offset.y - 0.5, 0.0, static_cast<double>(size_y - 1));
	auto x_idx = static_cast<size_t>(x);
	auto y_idx = static_cast<size_t>(y);

	// adjust indices, if they lie on the bottom or right edge of the tile
	if (x_idx == size_x - 1) {
		x_idx -= 1;
	}
	if (y_idx == size_y - 1) {
		y_idx -= 1;
	}

	auto pixel_idx = Pixel({x_idx, y_idx});

	// Get the 4 corners
	auto top_left     = read_pixel(pixel_idx);
	auto top_right    = read_pixel(Pixel{pixel_idx.x + 1, pixel_idx.y});
	auto bottom_left  = read_pixel(Pixel{pixel_idx.x, pixel_idx.y + 1});
	auto bottom_right = read_pixel(Pixel{pixel_idx.x + 1, pixel_idx.y + 1});

	// NOTE: if any of the corners is NAN (meaning that a no-data-value was found at that pixel),
	// the elevation interpolation will automatically return NAN as the output
	auto elevation = ElevationInterpolator(top_left, top_right, bottom_left, bottom_right)
	                     .interpolate({x - pixel_idx.x, y - pixel_idx.y});

	return elevation;
}

bool Tile::contains(const Coordinate& map_coords) const {
	return (map_coords.x > bounds.down_left.x && map_coords.x < bounds.down_right.x &&
	        map_coords.y > bounds.down_left.y && map_coords.y < bounds.up_left.y);
}

void Tile::scan_tile() const {
	cached_tile = std::vector<double>(size_x * size_y, no_data_value);

	if (dataset->GetRasterBand(1)->RasterIO(
	        GF_Read, 0, 0, size_x, size_y, cached_tile.data(), size_x, size_y, GDT_Float64, 0, 0) !=
	    CPLE_None) {
		spdlog::warn("Unable to read tile.");
	}
}


void Tile::unload() const {
	if (cached) {
		cached_tile.clear();
		cached_tile.shrink_to_fit();
		cached = false;
	}
}

std::unique_ptr<OGRCoordinateTransformation, detail::TransformDelete> Tile::wgs84_to_map_transform()
    const {
	return detail::create_wgs84_to_map_transformation(*dataset);
}

std::unique_ptr<OGRCoordinateTransformation, detail::TransformDelete> Tile::map_to_wgs84_transform()
    const {
	return detail::create_map_to_wgs84_transformation(*dataset);
}

const std::ostream& operator<<(std::ostream& os, Tile& tile) {

	for (size_t y = 0; y < tile.get_height(); ++y) {
		for (size_t x = 0; x < tile.get_width(); ++x) {
			auto elevation = tile.read_pixel(Pixel{x, y});
			if (elevation == NAN) {
				os << "Point (" << x << ", " << y << ") could not be read." << std::endl;
			} else {
				os << elevation << " ";
			}
		}
		os << std::endl;
	}

	return os;
}

}  // namespace geospatial
}  // namespace navtk
