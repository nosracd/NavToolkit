#pragma once

#include <memory>
#include <ostream>

#ifndef GDAL_INCLUDE_IN_SUBFOLDER
#	include <gdal_priv.h>
#	include <ogr_spatialref.h>
#else
#	include <gdal/gdal_priv.h>
#	include <gdal/ogr_spatialref.h>
#endif

#include <navtk/geospatial/Coordinates.hpp>
#include <navtk/geospatial/detail/custom_deleters.hpp>
#include <navtk/geospatial/detail/transformations.hpp>

namespace navtk {
namespace geospatial {

/**
 * This class represents a valid GDAL tile.
 */
class Tile {
public:
	/**
	 * Constructor
	 *
	 * @param filename the Gdal file path
	 */
	Tile(const std::string& filename);

	/**
	 * Scan the whole dataset tile to store in the cache.
	 */
	void scan_tile() const;

	/**
	 * Check whether the tile has a valid file.
	 *
	 * @return Whether raster tile is associated with valid file.
	 */
	bool is_valid() const;

	/**
	 * Check if the tile is rotated in map space.
	 *
	 * @return Whether the tile is rotated.
	 */
	bool is_rotated() const;

	/**
	 * Check if a tile is currently cached.
	 *
	 * @return Cached state of the tile.
	 */
	bool is_cached() const;

	/**
	 * Gets the elevation from a GDAL tile.
	 *
	 * @param map_coords coordinate in map space.
	 * @return found elevation.  Note that this function does not perform a bounds check, as the
	 * `GdalSource` confirms that the coordinates are in bounds before trying to look up the datum.
	 * If an out of bound map coordinate is provided, it will return the closest elevation contained
	 * in the tile.  If data is missing from the nearest pixels in the tile, it will return NAN.
	 */
	double lookup_datum(const Coordinate& map_coords) const;

	/**
	 * Gets the elevations from a GDAL tile.
	 *
	 * @param map_coords batch of coordinates in map space.
	 * @return A vector of elevations.
	 */
	Vector lookup_data(const Coordinates& map_coords) const;

	/**
	 * Check if the given coordinates are in the bounds of this tile
	 *
	 * @param map_coords coordinates in map space
	 *
	 * @return `true` if the given coordinates are in the tile.
	 */
	bool contains(const Coordinate& map_coords) const;

	/**
	 * Check if the given set of coordinates are in the bounds of this tile
	 *
	 * @param map_coords batch of N coordinates in map space.
	 * @param search_mask optional mask of length N, where indices of coordinates to look for are
	 * true. If an index is false, `contains` will ignore that index, even if it is contained in
	 * the tile. This parameter exists to improve performance by avoiding unnecessary checks when a
	 * coordinate may have already been found in another Tile. This is faster than passing in only
	 * the set of Coordinates to search for, which would require substantial memory re-allocation.
	 *
	 * @return indices of coordinates in \p map_coords contained in tile.
	 */
	std::vector<size_t> contains(const Coordinates& map_coords,
	                             const std::vector<bool>& search_mask = {}) const;

	/**
	 * Remove the tile data from memory, if cached
	 */
	void unload() const;

	/**
	 * Return a human-readable name for this object.
	 *
	 * @return The file name.
	 */
	std::string get_filename() const;

	/**
	 * Returns the total number of pixels in each line.
	 *
	 * @return The number of pixels in the line.
	 */
	size_t get_width() const;

	/**
	 * Returns the total number of lines in the dataset object (i.e. in the file).
	 *
	 * @return The number of lines.
	 */
	size_t get_height() const;

	/**
	 * Get transformation from wgs84 to map for this tile.
	 *
	 * @return pointer to transformation
	 */
	std::unique_ptr<OGRCoordinateTransformation, detail::TransformDelete> wgs84_to_map_transform()
	    const;

	/**
	 * Get transformation from map to wgs84 for this tile.
	 *
	 * @return pointer to transformation
	 */
	std::unique_ptr<OGRCoordinateTransformation, detail::TransformDelete> map_to_wgs84_transform()
	    const;

	/**
	 * Transform coordinates in map space into the pixel space of this particular tile.
	 *
	 * @param map_coords coordinates in map space
	 * @return the projected pixel coordinates for this tile
	 */
	Coordinate map_to_pixel(const Coordinate& map_coords) const;

	/**
	 * Transform a batch of coordinates in map space into the pixel space of this particular tile.
	 *
	 * @param map_coords a batch of coordinates in map space
	 * @return the projected batch of pixel coordinates for this tile
	 */
	Coordinates map_to_pixel(const Coordinates& map_coords) const;

	/**
	 * Transform coordinates in the pixel space of this particular tile into map space.
	 *
	 * @param pixel_coords coordinates in the pixel space of this tile
	 * @return the projected map coordinates
	 */
	Coordinate pixel_to_map(const Coordinate& pixel_coords) const;

	/**
	 * Transform a batch of coordinates in the pixel space of this particular tile into map space.
	 *
	 * @param pixel_coords a batch of coordinates in the pixel space of this tile
	 * @return the projected batch of map coordinates
	 */
	Coordinates pixel_to_map(const Coordinates& pixel_coords) const;

	/**
	 * Overload insertion operator to print the tile's data
	 *
	 * @param os the `ostream` object
	 * @param tile the tile to print
	 *
	 * @return The output stream `os`.
	 */
	friend const std::ostream& operator<<(std::ostream& os, Tile& tile);

private:
	struct Bounds {
		Coordinate down_left;
		Coordinate down_right;
		Coordinate up_left;
		Coordinate up_right;
	};

	Bounds get_bounds(std::array<Coordinate, 4> corners);

	std::string filename;

	double no_data_value;

	bool valid = false;

	std::unique_ptr<GDALDataset, detail::DatasetDelete> dataset;

	size_t size_x, size_y;

	Bounds bounds;

	mutable std::vector<double> cached_tile;
	mutable bool cached = false;

	double map_offset_x;
	double map_offset_y;
	double pixel_width;
	double pixel_height;
	double row_rotation;
	double column_rotation;

	double read_pixel(const Pixel& idx) const;
	Vector read_pixels(const Pixels& idx) const;
};
}  // namespace geospatial
}  // namespace navtk
