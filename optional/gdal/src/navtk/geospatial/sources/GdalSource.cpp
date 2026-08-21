#include <navtk/geospatial/sources/GdalSource.hpp>

#include <cmath>
#include <memory>
#include <stdexcept>

#include <navtk/errors.hpp>
#include <navtk/fs/filesystem.hpp>
#include <navtk/geospatial/Tile.hpp>
#include <navtk/navutils/math.hpp>
#include <navtk/navutils/navigation.hpp>

namespace navtk {
namespace geospatial {

using navtk::navutils::RAD2DEG;

GdalSource::GdalSource(const std::string& map_path,
                       MapType type,
                       AspnMeasurementAltitudeReference in_ref,
                       AspnMeasurementAltitudeReference out_ref,
                       unsigned int num_tiles,
                       const std::string& undulation_path)
    : map_type(type), undulation_path(undulation_path) {

	input_reference  = in_ref;
	output_reference = out_ref;
	if (in_ref == ASPN_MEASUREMENT_ALTITUDE_REFERENCE_AGL ||
	    out_ref == ASPN_MEASUREMENT_ALTITUDE_REFERENCE_AGL) {
		spdlog::warn(
		    "ASPN_MEASUREMENT_ALTITUDE_REFERENCE_AGL is unsupported. Setting input and output "
		    "reference to HAE.");
		input_reference  = ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE;
		output_reference = ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE;
	}

	max_size = num_tiles;

	// find all tiles at the provided map path
	find_tiles(map_path);
}

void GdalSource::find_tiles(const std::string& map_path) {
	std::string extension;
	switch (map_type) {
	case MapType::GEOTIFF:
		extension = ".tif";
		break;
	case MapType::DTED:
		extension = ".dt";
		break;
	default:
		log_or_throw<std::invalid_argument>("Invalid map type.");
	}

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

			add_tile(filename);
		}
	}

	if (get_size() == 0) {
		log_or_throw("GdalSource: No elevation files found in path {}",
		             fmt::streamed(absolute_map_path));
	}

	if (map_type == MapType::DTED) {
		// sort so that '.dt5' files are found before '.dt0'
		std::sort(search_order.begin(),
		          search_order.end(),
		          [this](const size_t idx_1, const size_t idx_2) {
			          return tiles[idx_1].get_filename().back() >
			                 tiles[idx_2].get_filename().back();
		          });
	}
}

void GdalSource::add_tile(const std::string& filename) {

	if (is_stored(filename)) {
		log_or_throw<std::invalid_argument>("{} is already stored in tile store.  Ignoring Tile.",
		                                    filename);
		return;
	}

	const auto idx = tiles.size();
	search_order.push_back(idx);
	tiles.emplace_back(filename);

	if (!tiles[idx].is_valid()) {
		search_order.pop_back();
		tiles.pop_back();
		log_or_throw<std::invalid_argument>("Could not find file {}.  Ignoring Tile.", filename);
		return;
	}

	if (!is_valid_tile(tiles[idx])) {
		search_order.pop_back();
		tiles.pop_back();
		log_or_throw<std::invalid_argument>(
		    "Tile coordinate system doesn't match store.  Ignoring Tile.");
		return;
	}

	// if no transform has yet been set for the source, use this tile's transformation
	if (!wgs84_to_map_transform) {
		wgs84_to_map_transform = tiles[idx].wgs84_to_map_transform();

		if (wgs84_to_map_transform->GetSourceCS() == wgs84_to_map_transform->GetTargetCS())
			need_transform = false;
	}
}

bool GdalSource::is_valid_tile(const Tile& tile) const {
	// currently, we don't support rotated tiles
	if (tile.is_rotated()) {
		return false;
	}

	auto tile_transform = tile.wgs84_to_map_transform();

	if (!tile_transform) {
		// if (for some reason), the tile returned a nullptr, it is automatically not valid
		return false;
	}

	if (!wgs84_to_map_transform) {
		// if no reference transform exists (i.e., no tiles have been added yet), than any transform
		// is valid
		return true;
	}

	auto src        = tile_transform->GetSourceCS();
	auto target     = tile_transform->GetTargetCS();
	auto ref_src    = wgs84_to_map_transform->GetSourceCS();
	auto ref_target = wgs84_to_map_transform->GetTargetCS();

	if (!src || !target || !ref_src || !ref_target) {
		// if any of the coordinate systems are null, throw error
		throw std::runtime_error("Found a coordinate transform with a null cooridnate system.");
	}

	if (src->IsSame(ref_src) == FALSE || target->IsSame(ref_target) == FALSE) {
		// if the source or target coordinate systems don't match the reference, the transform
		// is not valid
		return false;
	}

	return true;
}

void GdalSource::set_output_vertical_reference_frame(AspnMeasurementAltitudeReference new_ref) {
	if (new_ref == ASPN_MEASUREMENT_ALTITUDE_REFERENCE_AGL) {
		spdlog::warn(
		    "Setting output reference to ASPN_MEASUREMENT_ALTITUDE_REFERENCE_AGL is not "
		    "supported.");
	} else {
		output_reference = new_ref;
	}
}

std::pair<bool, double> GdalSource::lookup_datum(double latitude, double longitude) const {

	auto coordinate = wgs84_to_map(latitude, longitude);

	size_t tile_idx;

	for (auto tile_iter = search_order.begin(); tile_iter != search_order.end(); tile_iter++) {

		tile_idx = *tile_iter;

		if (tiles[tile_idx].contains(coordinate)) {
			mark_tile_as_cached(tile_iter);
			auto elevation = tiles[tile_idx].lookup_datum(coordinate);

			if (std::isnan(elevation)) {
				continue;
			}

			// TODO (#733): ideally, this conversion won't need to occur here because it would
			// happen at initialization
			if (input_reference == ASPN_MEASUREMENT_ALTITUDE_REFERENCE_MSL &&
			    output_reference == ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE) {
				return navtk::navutils::msl_to_hae(elevation, latitude, longitude, undulation_path);
			} else if (input_reference == ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE &&
			           output_reference == ASPN_MEASUREMENT_ALTITUDE_REFERENCE_MSL) {
				return navtk::navutils::hae_to_msl(elevation, latitude, longitude, undulation_path);
			}
			return {true, elevation};
		}
	}

	spdlog::debug("GdalSource::lookup_datum failed!  {}/{} not in known tiles.",
	              latitude * RAD2DEG,
	              longitude * RAD2DEG);

	return {false, NAN};
}

Coordinate GdalSource::wgs84_to_map(double latitude_rad, double longitude_rad) const {

	if (wgs84_to_map_transform) {
		double x_geo = longitude_rad * RAD2DEG;
		double y_geo = latitude_rad * RAD2DEG;
		if (need_transform) wgs84_to_map_transform->Transform(1, &x_geo, &y_geo);

		return {x_geo, y_geo};
	} else {
		throw std::runtime_error(
		    "Cannot convert to map space if no tiles have been added to storage!");
	}
}

void GdalSource::mark_tile_as_cached(const std::vector<size_t>::iterator& iter) const {

	if (iter == search_order.begin()) {
		return;
	}

	// move the current tile to the front of the search_order
	std::rotate(search_order.begin(), iter, iter + 1);

	// if we have the specified number of cached tiles already, unload the least recently used one
	if (tiles.size() > max_size && tiles[search_order[max_size]].is_cached()) {
		tiles[search_order[max_size]].unload();
	}

	// NOTE: since a tile will automatically cache itself when read from, there is no need to
	// manually cache it here.  The assumption is that if the user is about to read from the
	// tile (hense the call to `cache_tile`), it will get cached posthaste.  But in the mean
	// time, no need to cache it before it is used.
}

size_t GdalSource::get_size() const { return tiles.size(); }

size_t GdalSource::get_cached_num() const {
	size_t total_cached = 0;

	for (size_t i = 0; i < tiles.size(); i++) {
		if (tiles[i].is_cached()) {
			total_cached++;
		}
	}

	return total_cached;
}

bool GdalSource::is_stored(const std::string& filename) const {
	for (size_t i = 0; i < tiles.size(); i++) {
		if (tiles[i].get_filename() == filename) {
			return true;
		}
	}
	return false;
}
}  // namespace geospatial
}  // namespace navtk
