#pragma once

#include <memory>
#include <utility>

#include <navtk/geospatial/Tile.hpp>
#include <navtk/geospatial/sources/ElevationSource.hpp>
#include <navtk/not_null.hpp>

namespace navtk {
namespace geospatial {
/**
 * This is an abstract class for reading data files that conform to MIL-PRF-89020B, using Geospatial
 * Data Abstraction Library (GDAL). All GDAL map data sources should extend from this class.
 *
 * To use this package GDAL must be installed and built with navtk.
 */
class GdalSource : public ElevationSource {
public:
	/**
	 * Current GDAL implementations
	 */
	enum class MapType {
		/**
		 * A TIFF file with georeferencing information embedded in it.
		 */
		GEOTIFF,
		/**
		 * An NGA standard of terrain elevation data.
		 */
		DTED
	};

	/**
	 * Constructor
	 *
	 * @param map_path to directory containing one or more GDAL data files. All GDAL files in this
	 * directory must have the same file format and vertical reference frame.
	 * @param type the gdal format used by the dataset.
	 * @param in_ref the vertical reference frame stored in the input directory. Defaults to
	 * ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE.
	 * @param out_ref the vertical reference frame for output elevations. Defaults to
	 * ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE.
	 * @param undulation_path the path to the geoid undulation file for converting between HAE and
	 * MSL. The default path of this variable requires setting the NAVTK_DATA_DIR environment
	 * variable to the folder containing the undulation file, or setting the
	 * NAVTK_GEOID_UNDULATION_PATH environment variable to the path of the file itself.
	 * @param num_tiles the max number of tiles to store in memory.
	 */
	GdalSource(const std::string& map_path,
	           MapType type,
	           AspnMeasurementAltitudeReference in_ref  = ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE,
	           AspnMeasurementAltitudeReference out_ref = ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE,
	           unsigned int num_tiles                   = 10,
	           const std::string& undulation_path       = "WW15MGH.GRD");

	/**
	 * Gets the elevation at a given latitude and longitude.
	 *
	 * @param latitude latitude value in radians
	 * @param longitude longitude value in radians
	 * @return A `pair` showing whether a valid elevation was found (`.first`) and if `true`, the
	 * elevation in meters above either geoid -- aka mean sea level -- or ellipsoid (`.second`).
	 */
	std::pair<bool, double> lookup_datum(double latitude, double longitude) const override;

	/**
	 * @param new_ref the output vertical reference frame to change to. If `new_ref` is
	 * `ASPN_MEASUREMENT_ALTITUDE_REFERENCE_AGL`, then this function will do nothing.`
	 */
	void set_output_vertical_reference_frame(AspnMeasurementAltitudeReference new_ref) override;

	/**
	 * Transform input lat/lon cooridnate into map space, using the coordinate transform common to
	 * all the tiles added to this store.
	 *
	 * @param latitude_rad latitude in radians
	 * @param longitude_rad longitude in radians
	 * @return coordinate in map space, stored as {x_map, y_map}
	 */
	Coordinate wgs84_to_map(double latitude_rad, double longitude_rad) const;

	/**
	 * Checks to see if a tile with the given filename
	 * is currently stored.
	 *
	 * @param filename the filename of the tile
	 * @return `true` if found, `false` otherwise.
	 */
	bool is_stored(const std::string& filename) const;

	/**
	 * Returns the number of tiles currently cached.
	 *
	 * @return Number of tiles cached in the store.
	 */
	size_t get_cached_num() const;

	/**
	 * Returns the number of tiles currently stored.
	 *
	 * @return Number of tiles stored.
	 */
	size_t get_size() const;

private:
	/**
	 * Find tiles from the provided path and add them to the internal store.
	 *
	 * @param map_path path to the location of a dataset
	 */
	void find_tiles(const std::string& map_path);

	/**
	 * Adds a tile to the storage, running validation to ensure it belongs in the store
	 *
	 * @param filename filename for tile
	 */
	void add_tile(const std::string& filename);

	/**
	 * Check if the provided tile is compatible with the current dataset.
	 *
	 * This function will compare the coordinate systems of the incoming tile and the coordinate
	 * system of the tiles already added to the store, and ensure that they are the same.  This
	 * allows for a single wgs84 to map transformation that applies to all tiles.
	 *
	 * @param tile tile to check
	 */
	bool is_valid_tile(const Tile& tile) const;

	/**
	 * Marks a tile as cached in the internal store.  This will move the tile reference to the first
	 * slot in tile_indices.  If the internal number of cached tiles has already reached it's
	 * maximum length, it will also unload the oldest cached tile to save on memory.  This function
	 * should be called on any tile before it is read from, as reading from a tile will
	 * automatically cache it internally, and the `GdalSource` should keep track of what tiles are
	 * cached.
	 *
	 * NOTE: this function does not actually cache the tile.
	 *
	 * @param iter iterator into search_order that correponds to the tile to cache
	 */
	void mark_tile_as_cached(const std::vector<size_t>::iterator& iter) const;

	/**
	 * Vector of the tiles owned by the source.
	 *
	 * For efficiency, this vector will never change order, but will always be arranged in the order
	 * of insertion.
	 */
	std::vector<Tile> tiles;

	/**
	 * Vector of tile indices that specifies the search order.
	 *
	 * This vector contains indices into the tiles vector that can change order.  This is done for
	 * efficiency, as a `Tile` object is much larger than a `size_t` object, and so is easier to
	 * move around inside a vector.  Any cached tiles are gauranteed to have their indices come
	 * first in this vector, before uncached tile indices.  These indices are then used as the
	 * search order for the internally stored tiles.
	 */
	mutable std::vector<size_t> search_order;

	size_t max_size;

	GdalSource(const GdalSource&)            = delete;
	GdalSource& operator=(const GdalSource&) = delete;

	std::unique_ptr<OGRCoordinateTransformation, detail::TransformDelete> wgs84_to_map_transform =
	    nullptr;
	bool need_transform = true;

	const MapType map_type;
	std::string undulation_path;
};
}  // namespace geospatial
}  // namespace navtk
