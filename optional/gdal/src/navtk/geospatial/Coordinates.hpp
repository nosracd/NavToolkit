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
 * A batch of two-dimensional index into a GDAL tile (i.e. the pixel index), where #y is the line
 * and #x is the element within the line.
 */
struct Pixels {
	/**
	 * Horizontal positions within an individual line.
	 */
	std::vector<size_t> x;

	/**
	 * Vertical (line) indices.
	 */
	std::vector<size_t> y;

	/**
	 * Allocate memory for a batch of pixels.
	 *
	 * @param N number of pixels in batch
	 */
	Pixels(size_t N) : x(N), y(N) {};
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

/**
 * A vector of points in some space.
 */
struct Coordinates {
	/**
	 * x-coordinates
	 */
	Vector x;

	/**
	 * y-coordinates
	 */
	Vector y;
};

}  // namespace geospatial
}  // namespace navtk
