#pragma once

#include <type_traits>

#include <xtensor/core/xexpression.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/views/xbroadcast.hpp>

#include <navtk/inspect.hpp>
#include <navtk/tensors.hpp>

namespace navtk {

/**
 * Converts an object of 1 dimension to a 1-D vector.
 *
 * @tparam T An 1 dimensional object.
 * @tparam IfTensorOfDim<> Invalidates the template for non-matching types. (Has no effect on the
 * function).
 * @param m Input object containing coefficients.
 * @return 1-D vector of the container object.
 */
template <class T, IfTensorOfDim<T, 1>* = nullptr>
inline Vector to_vec(const T& m) {
	if constexpr (xt::is_xexpression<T>::value == true) {
		return m;
	} else {
		return Vector(m);
	}
}

/**
 * Converts an xtensor container object of 2 dimensions to a 2-D matrix.
 *
 * @tparam T An xtensor container object.
 * @tparam IfTensorOfDim<> Invalidates the template for non-matching types. (Has no effect on the
 * function).
 * @param m Input object containing coefficients.
 * @return 2-D matrix of the container object.
 */
template <class T, IfTensorOfDim<T, 2>* = nullptr>
inline Matrix to_matrix(const T& m, std::size_t = 1) {
	if constexpr (xt::is_xexpression<T>::value == true) {
		return m;
	} else {
		return Matrix(m);
	}
}

/**
 * Converts an object of 3 dimensions to a 3-D tensor.
 *
 * @tparam T An object of 3 dimensions.
 * @tparam IfTensorOfDim<> Invalidates the template for non-matching types. (Has no effect on the
 * function).
 * @param m Input object containing coefficients.
 * @return 3-D tensor of the container object.
 */
template <class T, IfTensorOfDim<T, 3>* = nullptr>
inline Tensor<3> to_tensor_3d(const T& m, std::size_t = 1) {
	if constexpr (xt::is_xexpression<T>::value == true) {
		return m;
	} else {
		return Tensor<3>(m);
	}
}

#ifndef NEED_DOXYGEN_EXHALE_WORKAROUND

/**
 * Converts an object of 0 dimensions to a 1-D vector.
 *
 * @tparam T An 0 dimensional object.
 * @tparam IfTensorOfDim<> Invalidates the template for non-matching types. (Has no effect on the
 * function).
 * @param m Input object containing coefficients.
 * @return 1-D vector of the container object.
 */
template <class T, IfTensorOfDim<T, 0>* = nullptr>
inline Vector to_vec(const T& m) {
	if constexpr (std::ranges::range<T>) {
		if (has_zero_size(m)) return Vector{};
		return Vector{m()};
	} else if constexpr (std::is_arithmetic<T>::value == true) {
		return Vector({m});
	} else {
		return Vector{m()};
	}
}

/**
 * Converts an object of 2 dimensions to a 1-D vector.
 * Data is not lost in this conversion.  It may be flattened if necessary.
 *
 * @tparam T An object of two dimensions.
 * @tparam IfTensorOfDim<> Invalidates the template for non-matching types. (Has no effect on the
 * function).
 * @param m Input object containing coefficients.
 * @return 1-D vector of the container object.
 */
template <class T, IfTensorOfDim<T, 2>* = nullptr>
Vector to_vec(const T& m) {
	auto mat = to_matrix(m);
	if (has_zero_size(mat)) {
		return Vector{};
	} else if (mat.shape()[0] == 1 && mat.shape()[1] == 1) {
		return Tensor<1>{mat(0, 0)};
	} else if (mat.shape()[0] == 1 || mat.shape()[1] == 1) {
		return xt::squeeze(mat);
	}

	// Combine data from all rows into one row
	return xt::flatten(mat);
}

/**
 * Converts an object of 3 dimensions to a 1-D vector.
 * Data is not lost in this conversion.  It may be flattened if necessary.
 *
 * @tparam T A 3 dimensional object.
 * @tparam IfTensorOfDim<> invalidates the template for non-matching types. (Has no effect on the
 * function).
 * @param m Input object containing coefficients.
 * @return 1-D vector of the container object.
 */
template <class T, IfTensorOfDim<T, 3>* = nullptr>
Vector to_vec(const T& m) {
	auto tensor = to_tensor_3d(m);
	if (has_zero_size(tensor)) {
		return Vector{};
	} else if (tensor.shape()[0] == 1 && tensor.shape()[1] == 1 && tensor.shape()[2] == 1) {
		return Tensor<1>{tensor(0, 0, 0)};
	}

	// if there is at least one dimmension that is not being used
	if (tensor.shape()[0] == 1 || tensor.shape()[1] == 1 || tensor.shape()[2] == 1) {
		auto squeezed = xt::squeeze(tensor);
		if (squeezed.dimension() == 1) {
			return squeezed;
		} else if (squeezed.dimension() == 2) {
			return xt::flatten(squeezed);
		}
	}

	return xt::flatten(tensor);
}

/**
 * Converts an Eigen-like container object to a 1-D vector.  The container must have
 * implemented a `rows()` function, a `cols()` function, and the `operator() `to work properly.
 *
 * @tparam T An Eigen-like container object.
 * @tparam IfEigenInterface<> invalidates the template for non-matching types. (Has no effect on the
 * function).
 * @param m Input object containing coefficients.
 * @return 1-D vector of the container object.
 */
template <typename T, IfEigenInterface<T>* = nullptr>
Vector to_vec(const T& m) {

	// TODO: PNTOS-56 Instead, could attempt to block memory copy if we can determine memory layout
	// of m
	auto rows = m.rows();
	auto cols = m.cols();

	Vector out = xt::zeros<Scalar>({rows * cols});
	for (decltype(rows) i = 0; i < rows; i++) {
		for (decltype(cols) j = 0; j < cols; j++) {
			out(i * cols + j) = m(i, j);
		}
	}
	return out;
}

/**
 * Converts an object of 0 dimensions to a 2-D matrix.
 *
 * @tparam T A 0 dimensional object.
 * @tparam IfTensorOfDim<> Invalidates the template for non-matching types. (Has no effect on the
 * function).
 * @param m Input object containing coefficients.
 * @return 2-D matrix of the container object.
 */
template <class T, IfTensorOfDim<T, 0>* = nullptr>
inline Matrix to_matrix(const T& m, std::size_t = 1) {
	if constexpr (std::ranges::range<T>) {
		if (has_zero_size(m)) return Matrix{};
		return Matrix{{m()}};
	} else if constexpr (std::is_arithmetic<T>::value == true) {
		return Matrix{{m}};
	} else {
		return Matrix{{m()}};
	}
}

/**
 * Converts an object of 1 dimension to a 2-D matrix.
 *
 * @tparam T An object of 1 dimension.
 * @tparam IfTensorOfDim<> Invalidates the template for non-matching types. (Has no effect on the
 * function).
 * @param m Input object containing coefficients.
 * @param axis Which axis to expand. When 0, a 4-element 1-D vector will return a 1x4 matrix. When
 * 1, a 4-element vector will return a 4x1 matrix.
 * @return 2-D matrix of the container object.
 */
template <class T, IfTensorOfDim<T, 1>* = nullptr>
Matrix to_matrix(const T& m, std::size_t axis = 1) {
	auto vec = to_vec(m);
	return xt::expand_dims(vec, axis);
}

/**
 * Converts an xtensor container object of 3 dimensions to a 2-D matrix.
 * Data is not lost in this conversion.  It may be flattened and placed into one of the matrix
 * dimensions, depending on the axis parameter.
 *
 * @tparam T An xtensor container object.
 * @tparam IfTensorOfDim<> Invalidates the template for non-matching types. (Has no effect on the
 * function).
 * @param m Input object containing coefficients.
 * @param axis Which axis to place flattened data into, if shape could not be squeezed.  When 0,
 * the result will be a 1xN matrix. When 1, the result will be an Nx1 matrix.
 * @return 2-D matrix of the container object.
 */
template <class T, IfTensorOfDim<T, 3>* = nullptr>
Matrix to_matrix(const T& m, std::size_t axis = 1) {
	auto tensor = to_tensor_3d(m);
	if (has_zero_size(tensor)) {
		return Matrix{};
	}

	if (tensor.shape()[0] == 1 && tensor.shape()[1] == 1 && tensor.shape()[2] == 1) {
		return Tensor<2>{{tensor(0, 0, 0)}};
	}

	if (tensor.shape()[0] == 1 || tensor.shape()[1] == 1 || tensor.shape()[2] == 1) {
		auto squeezed = xt::squeeze(tensor);
		if (squeezed.dimension() == 1) {
			return xt::expand_dims(squeezed, axis);
		} else if (squeezed.dimension() == 2) {
			return squeezed;
		}
	}

	auto flattened = xt::flatten(tensor);

	return xt::expand_dims(flattened, axis);
}

/**
 * Converts an Eigen-like container object to a 2-D matrix.  The container must have
 * implemented a `rows()` function, a `cols()` function, and the `operator()` to work properly.
 *
 * @tparam T An Eigen-like container object.
 * @tparam IfEigenInterface<> Invalidates the template for non-matching types. (Has no effect on the
 * function).
 * @param m Input object containing coefficients.
 * @return 2-D matrix of the container object.
 */
template <typename T, IfEigenInterface<T>* = nullptr>
Matrix to_matrix(const T& m, std::size_t = 1) {

	// TODO: PNTOS-56 Instead, could attempt to block memory copy if we can determine memory layout
	// of m
	auto rows = m.rows();
	auto cols = m.cols();

	Matrix out = xt::zeros<Scalar>({rows, cols});
	for (decltype(rows) i = 0; i < rows; i++) {
		for (decltype(cols) j = 0; j < cols; j++) {
			out(i, j) = m(i, j);
		}
	}
	return out;
}

/**
 * Converts any fixed-size 2d array to a Matrix.
 */
template <typename T, std::size_t rows, std::size_t cols>
Matrix to_matrix(T (&data)[rows][cols]) {
	Matrix out = xt::zeros<Scalar>({rows, cols});
	for (Size ii = 0; ii < rows; ++ii)
		for (Size jj = 0; jj < cols; ++jj) out(ii, jj) = data[ii][jj];
	return out;
}

/**
 * Converts an object of 0 dimensions to a 3-D tensor.
 *
 * @tparam T An object of 0 dimensions.
 * @tparam IfTensorOfDim<> Invalidates the template for non-matching types. (Has no effect on the
 * function).
 * @param m Input object containing coefficients.
 * @return 3-D tensor of the container object.
 */
template <class T, IfTensorOfDim<T, 0>* = nullptr>
inline Tensor<3> to_tensor_3d(const T& m, std::size_t = 1) {
	if constexpr (xt::is_xexpression<T>::value == true) {
		if (has_zero_size(m)) return Tensor<3>{};
		return Tensor<3>{{{m()}}};
	} else if constexpr (std::is_arithmetic<T>::value == true) {
		return Tensor<3>{{{m}}};
	} else {
		return Tensor<3>{{{m()}}};
	}
}

/**
 * Converts an object of 1 dimension to a 3-D tensor.
 *
 * @tparam T An object of 1 dimension.
 * @tparam IfTensorOfDim<> Invalidates the template for non-matching types. (Has no effect on the
 * function).
 * @param m Input object containing coefficients.
 * @param axis Which axis to place the vector in. When 0, a 4-element 1-D vector will return a 4x1x1
 * tensor. When 1, a 4-element vector will return a 1x4x1 tensor, etc.
 * @return 3-D tensor of the container object.
 */
template <class T, IfTensorOfDim<T, 1>* = nullptr>
Tensor<3> to_tensor_3d(const T& m, std::size_t axis = 2) {
	auto vec       = to_vec(m);
	const size_t N = vec.shape()[0];

	std::array<size_t, 3> shape = {1, 1, 1};
	shape[axis]                 = N;

	return xt::reshape_view(vec, shape);
}

/**
 * Converts an object of 2 dimensions to a 3-D tensor.
 *
 * @tparam T An object of 2 dimensions.
 * @tparam IfTensorOfDim<> Invalidates the template for non-matching types. (Has no effect on the
 * function).
 * @param m Input object containing coefficients.
 * @param axis Which axis to expand. When 0, a 2x2 matrix will return a 1x2x2 tensor. When
 * 1, a 2x2 matrix will return a 2x1x2 tensor, etc.
 * @return 3-D tensor of the container object.
 */
template <class T, IfTensorOfDim<T, 2>* = nullptr>
Tensor<3> to_tensor_3d(const T& m, std::size_t axis = 0) {
	auto mat = to_matrix(m);
	return xt::expand_dims(mat, axis);
}

/**
 * Converts any fixed-size 3d array to a 3-D Tensor.
 */
template <typename T, std::size_t index, std::size_t rows, std::size_t cols>
Tensor<3> to_tensor_3d(T (&data)[index][rows][cols]) {
	auto out = xt::zeros<Scalar>({index, rows, cols});
	for (Size ii = 0; ii < index; ++ii)
		for (Size jj = 0; jj < rows; ++jj)
			for (Size kk = 0; kk < cols; ++kk) out(ii, jj, kk) = data[ii][jj][kk];
	return out;
}
#endif

/**
 * Returns the NxM Matrix with given row and column size. The
 * placement of diagonal ones is specified with  `diagonal_index`.
 * By default, the diagonal ones are placed starting at `diagonal_index=0`,
 * or matrix index (0,0). `diagonal_index` can range from `-(rows-1)` to
 * `(cols-1)`.
 *
 * @param rows Number of rows for desired matrix.
 * @param cols Number of columns for desired matrix.
 * @param diagonal_index Placement index of the diagonal.
 *
 * @throw Throws an `invalid_argument` exception when `diagonal_index` is less than `-(cols-1)` or
 * greater than `(rows-1)` and the error mode is ErrorMode::DIE for either case.
 *
 * @return NxM Matrix where N is the number of rows and M is the number of columns.
 */
Matrix eye(Size rows, Size cols, int diagonal_index = 0);

/**
 * Returns the square identity Matrix with given number size.
 *
 * @param size Size of desired square Matrix.
 *
 * @return Identity Matrix of given size.
 */
Matrix eye(Size size);

#ifndef NEED_DOXYGEN_EXHALE_WORKAROUND
/**
 * Returns an empty tensor of a certain shape.
 *
 * The return value's dimensionality will match the number of arguments you pass in. In other words,
 * the function returns a Vector given one parameter, a Matrix given two parameters, and
 * higher-order Tensors for more parameters.
 *
 * This function is much faster than zeros for large Tensor objects.
 *
 * @param dim dimensions of the Tensor you're building.
 * @return An uninitializer Tensor.
 */
template <typename... T>
Tensor<sizeof...(T)> empty(T... dim) {
	using tensor_shape_type = typename Tensor<sizeof...(T)>::shape_type::value_type;
	return Tensor<sizeof...(T)>::from_shape({static_cast<tensor_shape_type>(dim)...});
}

/**
 * Returns a tensor filled with zeros.
 *
 * The return value's dimensionality will match the number of arguments you pass in. In other words,
 * the function returns a Vector given one parameter, a Matrix given two parameters, and
 * higher-order Tensors for more parameters.
 *
 * @param dim dimensions of the Tensor you're building.
 * @return A Tensor filled with zeros.
 */
template <typename... T>
Tensor<sizeof...(T)> zeros(T... dim) {
	return xt::zeros<Scalar>({Size(dim)...});
}

/**
 * Returns a tensor filled with the given value.
 *
 * The return value's dimensionality will match the number of arguments in the \p dim tuple. In
 * other words, the function returns a Vector given one parameter, a Matrix given two parameters,
 * and higher-order Tensors for more parameters.
 *
 * @param value Number with which to populate the resulting tensor.
 * @param dim dimensions of the Tensor you're building.
 * @return A Tensor filled with copies of the given value.
 */
template <typename... T>
Tensor<sizeof...(T)> fill(Scalar value, T... dim) {
	return zeros(dim...) + value;
}

/**
 * Returns a tensor filled with ones.
 *
 * The return value's dimensionality will match the number of arguments you pass in. In other words,
 * the function returns a Vector given one parameter, a Matrix given two parameters, and
 * higher-order Tensors for more parameters.
 *
 * @param dim dimensions of the Tensor you're building.
 * @return A Tensor filled with ones.
 */
template <typename... T>
Tensor<sizeof...(T)> ones(T... dim) {
	return xt::ones<Scalar>({Size(dim)...});
}
#endif

/**
 * Create a block-diagonal matrix from the provided matrices.
 *
 * Given square matrix inputs A, B and C, the output will have these arrays
 * arranged along the diagonal:
 *
 * ```
 * {{A, 0, 0},
 *  {0, B, 0},
 *  {0, 0, C}}
 * ```
 *
 * Vectors are assumed to be horizontal matrices (single-row), matching the
 * behavior of `scipy.linalg.block_diag`.
 *
 * @param matrices A series of matrices to be arranged.
 *
 * @return A single matrix containing the input matrices arranged diagonally,
 * such that the index of the top-left of a given matrix is (1, 1) plus the
 * index of the bottom-right of the preceding matrix.
 */
Matrix block_diag(std::initializer_list<Matrix> matrices);

/**
 * Create a block-diagonal matrix from the provided matrices.
 *
 * Given square matrix inputs A, B and C, the output will have these arrays
 * arranged along the diagonal:
 *
 * ```
 * {{A, 0, 0},
 *  {0, B, 0},
 *  {0, 0, C}}
 * ```
 *
 * Vectors are assumed to be horizontal matrices (single-row), matching the
 * behavior of `scipy.linalg.block_diag`.
 *
 * @param matrices A series of Tensor objects to be arranged.
 *
 * @return A single matrix containing the input matrices arranged diagonally,
 * such that the index of the top-left of a given matrix is (1, 1) plus the
 * index of the bottom-right of the preceding matrix.
 */
template <typename... T>
Matrix block_diag(T&&... matrices) {
	return block_diag({to_matrix(matrices, 0)...});
}

}  // namespace navtk
