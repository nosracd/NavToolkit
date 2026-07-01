#pragma once

#include <navtk/factory.hpp>
#include <navtk/inspect.hpp>
#include <navtk/tensors.hpp>
#include <xtensor/containers/xadapt.hpp>

namespace navtk {
namespace navutils {
/**
 * Constant definition of PI.
 */
extern const double PI;
/**
 * Ratio to convert degrees to radians.
 */
extern const double DEG2RAD;
/**
 * Ratio to convert radians to degrees.
 */
extern const double RAD2DEG;

/**
 * Forms skew symmetric matrices from 3-length vectors of values.
 * Typically used to form a matrix to correct a Direction Cosine Matrix
 * from the platform frame or sensor frame to the navigation frame with
 * estimated tilt angles in the navigation frame.
 *
 * If the input vector is \f$[x, y, z]\f$, then the resulting skew-symmetric
 * matrix is
 *
 * \f$    \begin{bmatrix}
        0 & -z & y \\
        z & 0 & -x \\
        -y & x & 0
    \end{bmatrix}\f$
 *
 * @tparam S Type of angles, Vector by default.
 * @tparam std::enable_if_t<> Constrains S to be 1 dimensional.
 *
 * @param angles 3-length vectors of angular values, shape (3).  Can accept initializer lists.
 *
 * @return Equivalent skew-symmetric matrices, shape (3, 3)
 */
template <typename S = Vector, IfTensorOfDim<S, 1>* = nullptr>
Matrix3 skew(const S& angles) {
	auto x = angles(0);
	auto y = angles(1);
	auto z = angles(2);

	return {{0, -z, y}, {z, 0, -x}, {-y, x, 0}};
}

/**
 * Batched version of `skew`.
 *
 * @overload
 *
 * @see skew
 *
 * @tparam B Type of angles, Matrix by default.
 * @tparam std::enable_if_t<> Constrains B to be 2 dimensional.
 *
 * @param angles 3-length vectors of angular values, shape (N, 3).  Can accept initializer lists.
 *
 * @return Equivalent skew-symmetric matrices, shape (N, 3, 3)
 */
template <typename B = Matrix, IfTensorOfDim<B, 2>* = nullptr>
Tensor<3> skew(const B& angles) {
	const size_t N = angles.shape()[0];

	// allocate return tensor
	auto skews = empty(N, 3, 3);

	Scalar x, y, z;

	for (size_t i = 0; i < N; i++) {
		x = angles(i, 0);
		y = angles(i, 1);
		z = angles(i, 2);

		skews(i, 0, 0) = 0;
		skews(i, 0, 1) = -z;
		skews(i, 0, 2) = y;
		skews(i, 1, 0) = z;
		skews(i, 1, 1) = 0;
		skews(i, 1, 2) = -x;
		skews(i, 2, 0) = -y;
		skews(i, 2, 1) = x;
		skews(i, 2, 2) = 0;
	}

	return skews;
}

/**
 * Performs orthonormalization of a Direction Cosine Matrix. Uses an
 * iterative gradient projection method from Bar-Itzhack, as referenced
 * by Mao.
 *
 * Reference: Optimal Orthonormalization of the Strapdown Matrix by
 * Using Singular Value Decomposition, Jianqin Mao, sec 2.
 *
 * @param dcm Original DCM (3x3)
 *
 * @return Orthonormalized matrix (3x3)
 */
Matrix3 ortho_dcm(const Matrix3& dcm);

/**
 * Adjust an angle value so that it lies within (-PI, PI].
 *
 * @param orig Original angle measure, radians.
 *
 * @return Adjusted angle, radians.
 */
double wrap_to_pi(double orig);

/**
 * Adjust an angle value so that it lies within [0, 2PI).
 *
 * @param orig Original angle measure, radians.
 *
 * @return Adjusted angle, radians.
 */
double wrap_to_2_pi(double orig);

}  // namespace navutils
}  // namespace navtk
