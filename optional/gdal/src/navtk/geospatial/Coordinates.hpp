#pragma once

#include <cstddef>
#include <vector>

#include <navtk/tensors.hpp>

namespace navtk {
namespace geospatial {

using navtk::Vector;

/**
 * A two-dimensional index into a GDAL tile (i.e. the pixel index), where #y is the line and #x is
 * the element within the line.
 */
struct Pixel {
	/**
	 * Horizontal position within an individual line.
	 */
	size_t x;

	/**
	 * Vertical (line) index.
	 */
	size_t y;
};

/**
 * A point in some space.
 */
struct Coordinate {
	/**
	 * x-coordinate
	 */
	double x;

	/**
	 * y-coordinate
	 */
	double y;
};

}  // namespace geospatial
}  // namespace navtk
