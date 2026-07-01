#pragma once

#include <initializer_list>
#include <navtk/tensors.hpp>
#include <type_traits>
#include <xtensor/utils/xutils.hpp>

namespace navtk {

/**
 * Checks whether there is any data in a given iterable object.
 * Intended as a safer alternative to `!expression.size()`, since some xtensor
 * types can crash when `.size()` is called.
 * @param expression An iterable object.
 * @return The boolean result.
 */
template <typename T>
bool has_zero_size(const T& expression) {
	return expression.begin() == expression.end();
}

/**
 * @param mat The matrix whose symmetry is evaluated.
 * @param rtol Relative tolerance allowed between 2 elements; unitless.
 * @param atol Absolute tolerance allowed between 2 elements; units determined by elements being
 * compared.
 * @return `true` if \p mat is symmetric within a tolerance and `false` otherwise. Does not check if
 *  square.
 */
bool is_symmetric(const Matrix& mat, double rtol = 1e-5, double atol = 1e-8);

/**
 * Check if a square matrix is diagonal.
 *
 * @param mat Matrix to check. Equal rows/cols are required but not checked.
 *
 * @return True if all off-diagonal elements are 0.
 */
bool is_diagonal(const Matrix& mat);

/**
 * Check if a square matrix is identity.
 *
 * @param mat Matrix to check. Equal rows/cols are required but not checked.
 *
 * @return True if all off-diagonal elements are 0 and all diagonal elements are
 *  1. Returns false if matrix has 0 size.
 */
bool is_identity(const Matrix& mat);

/**
 * Generate a set of row/column indices for non-symmetric matrix elements.
 *
 * @param mat Matrix to check. Equal rows/cols are required but not checked.
 * @param rtol Relative tolerance allowed between 2 elements; unitless.
 * @param atol Absolute tolerance allowed between 2 elements; units determined by elements being
 * compared.
 *
 * @return A vector of pairs of row/col indices (upper triangular only) for corresponding elements
 * that do not match within tolerance.
 */
std::vector<std::pair<Size, Size>> non_symmetric_elements(const Matrix& mat,
                                                          double rtol = 1e-5,
                                                          double atol = 1e-8);

/**
 * Base case of initializer list dimension checking.
 *
 * The intension is that if you have some initailizer list of unspecified dimension as the template
 * argument T, `InitializerListDepth<T>::value` will be the dimension of that initializer list type.
 *
 * @tparam T A value or initializer list.
 */
template <typename T>
struct InitializerListDepth {
	/**
	 * The internal type of the initializer list.
	 */
	using type = T;

	/**
	 * The dimension of the initializer list.
	 */
	static constexpr size_t value = 0;
};

/**
 * Recursive specialization of initializer list dimension checking.
 *
 * The intension is that if you have some initailizer list of unspecified dimension as the template
 * argument T, `InitializerListDepth<T>::value` will be the dimension of that initializer list type.
 *
 * @tparam T A value or initializer list.
 */
template <typename T>
struct InitializerListDepth<std::initializer_list<T>> {
	/**
	 * The internal type of the initializer list.
	 */
	using type = typename InitializerListDepth<T>::type;

	/**
	 * The dimension of the initializer list.
	 */
	static constexpr size_t value = 1 + InitializerListDepth<T>::value;
};

// don't document initializer helper stuff, but only the final result.
#ifndef NEED_DOXYGEN_EXHALE_WORKAROUND
template <size_t D, typename C>
struct InitializerListNested {
	using type =
	    std::initializer_list<typename InitializerListNested<(D > 1) ? D - 1 : 1, C>::type>;
};
// Base case of 1 dimesional
template <typename C>
struct InitializerListNested<1, C> {
	using type = std::initializer_list<C>;
};
// Handle 0 dimensional case so tha compiler doesn't explode, but not intended for use.
template <typename C>
struct InitializerListNested<0, C> {
	using type = C;
};
#endif

/**
 * Recursive initializer list helper type for templating a specific dimension of list.
 *
 * Usage would be Initializer<{dimension of overload you are aiming for}, Scalar>.  Example:
 * `Initializer<2, double>` would resolve as the type
 * `std::initializer_list<std::initializer_list<double>>`.
 *
 * @tparam D dimension of initializer list
 * @tparam C type of contents of the initializer list
 */
template <size_t D, typename C>
using Initializer = typename InitializerListNested<D, C>::type;



/**
 * Templated struct which allows us to obtain metadata information from a matrix/vector type.
 * This template specialization is matched if no additional information can be obtained.
 *
 * @tparam T A type from which we are attempting to gain additional information.
 * @tparam class class = void permits this to be a fallback if template specializations match.
 */
template <typename T, class = void>
struct TensorMeta {
	/**
	 * Indicates whether the dimensions of the tensor are fixed.
	 */
	static constexpr bool FIXED_DIMS = false;
};

/**
 * Templated struct which allows us to obtain metadata information from an xtensor type,
 * intended to support template functions and classes that use xtensor objects.
 *
 * @tparam T An xtensor type from which we are attempting to gain additional information.
 * @tparam std::enable_if_t<> Invalidates this structure for types that are not xtensor expressions.
 */
template <typename T>
struct TensorMeta<T, std::enable_if_t<xt::is_xexpression<T>::value>> {
	/**
	 * Whether the dimensions of the tensor are fixed.
	 */
	static constexpr bool FIXED_DIMS = true;

	/**
	 * The number of dimensions.
	 */
	static constexpr auto DIM_COUNT =
	    std::tuple_size<typename std::remove_reference<T>::type::shape_type>::value;
};


/**
 * Templated struct which allows us to obtain metadata information from an xtensor type,
 * intended to support template functions and classes that use xtensor objects.
 *
 * This particular instantiation allows for scalars (doubles) to be treated as xtensors for
 * templates, in the sense that they have dimension 0.
 *
 * @tparam T A Scalar type.
 * @tparam std::enable_if_t<> Invalidates this structure for types that are not Scalars.
 */
template <typename T>
struct TensorMeta<T, std::enable_if_t<std::is_arithmetic<std::remove_reference_t<T>>::value>> {
	/**
	 * Whether the dimensions of the tensor are fixed.
	 */
	static constexpr bool FIXED_DIMS = true;

	/**
	 * The number of dimensions, fixed at 0.
	 */
	static constexpr size_t DIM_COUNT = 0;
};


/**
 * Templated struct which allows us to obtain metadata information from an xtensor type,
 * intended to support template functions and classes that use xtensor objects.
 *
 * This particular instantiation allows for initializers willed with arithmetic types to be treated
 * as xtensors for template dimensioning.
 *
 * @tparam T A Scalar type.
 * @tparam std::enable_if_t<> Invalidates this structure for types that are not Scalars.
 */
template <typename T>
struct TensorMeta<
    T,
    std::enable_if_t<InitializerListDepth<T>::value != 0 &&
                     std::is_arithmetic<typename InitializerListDepth<T>::type>::value>> {
	/**
	 * Whether the dimensions of the tensor are fixed.
	 */
	static constexpr bool FIXED_DIMS = true;

	/**
	 * The number of dimensions, fixed at 0.
	 */
	static constexpr size_t DIM_COUNT = InitializerListDepth<T>::value;
};

/**
 * Check if a given type is actually a tensor.
 */
template <typename T>
inline constexpr bool IsValid = (TensorMeta<T>::FIXED_DIMS);

/**
 * `TensorsAreDim` can be used as a condition in SFINAE expressions.  To evaluate to true, all
 * listed tensors must have Dim dimensions.
 */
template <size_t Dim, typename... T>
inline constexpr bool TensorsAreDim =
    ((IsValid<T> && requires { typename std::enable_if_t<TensorMeta<T>::DIM_COUNT == Dim>; }) &&
     ...);

/**
 * `TensorsAreLessThanDim` can be used as a condition in SFINAE expressions.  To evaluate to true,
 * all listed tensors must have a dimension less than the Dim.
 */
template <size_t Dim, typename... T>
inline constexpr bool TensorsAreLessThanDim =
    ((IsValid<T> && requires { typename std::enable_if_t<(TensorMeta<T>::DIM_COUNT < Dim)>; }) &&
     ...);

/**
 * `TensorsHaveMaxDim` can be used as a condition in SFINAE expressions.  To evaluate to true, all
 * listed tensors must have a maximum dimension of _precisely_ Dim.
 */
template <size_t Dim, typename... T>
inline constexpr bool TensorsHaveMaxDim =
    TensorsAreLessThanDim<Dim + 1, T...> && !TensorsAreLessThanDim<Dim, T...>;

/**
 * SFINAE tool to check against one condition.
 */
template <bool Condition>
using IfCondition = std::enable_if_t<Condition>;

/**
 * Catchall SFINAE tool that enables a template based on all the given conditions being true.
 */
template <bool... Condition>
using IfAllConditions = IfCondition<(Condition && ...)>;

/**
 * Catchall SFINAE tool that enables a template based on any of the given conditions being true.
 */
template <bool... Condition>
using IfAnyConditions = IfCondition<(Condition || ...)>;

/**
 * `IfTensorOfDim` can be used in a template definition to invalidate the template. To be valid,
 * type T must be a Tensor type with `Dim` number of dimensions.
 */
template <typename T, std::size_t Dim>
using IfTensorOfDim = IfCondition<TensorsAreDim<Dim, T>>;

/**
 * `IfBothTensorsOfDim` can be used in a template definition to invalidate the template. To be
 * valid, type A and type B must both be Tensor types with `Dim` number of dimensions.
 */
template <typename A, typename B, std::size_t Dim>
using IfBothTensorsOfDim = IfCondition<TensorsAreDim<Dim, A, B>>;

/**
 * `IfFirstTensorOfDim` can be used in a template definition to invalidate the template. To be
 * valid, type A must have `Dim` number of dimensions, and type B must not.  Both must be Tensor
 * types (or scalars or initializer lists).
 */
template <typename A, typename B, std::size_t Dim>
using IfFirstTensorOfDim = IfAllConditions<TensorsAreDim<Dim, A>, !TensorsAreDim<Dim, B>>;

/**
 * `IfSecondTensorOfDim` can be used in a template definition to invalidate the template. To be
 * valid, type B must have `Dim` number of dimensions, and type A must not.  Both must be Tensor
 * types (or scalars or initializer lists).
 */
template <typename A, typename B, std::size_t Dim>
using IfSecondTensorOfDim = IfAllConditions<TensorsAreDim<Dim, B>, !TensorsAreDim<Dim, A>>;

/**
 * `IfAnyTensorOfDim` can be used in a template definition to invalidate the template. To be
 * valid, at least one type must have `Dim` number of dimensions.  Both must be Tensor
 * types (or scalars or initializer lists).
 */
template <std::size_t Dim, typename... T>
using IfAnyTensorOfDim = IfAnyConditions<TensorsAreDim<Dim, T>...>;

/**
 * `IfAllTensorsOfDim` can be used in a template definition to invalidate the template. To be
 * valid, all types must have `Dim` number of dimensions.  Both must be Tensor
 * types (or scalars or initializer lists).
 */
template <std::size_t Dim, typename... T>
using IfAllTensorsOfDim = IfCondition<TensorsAreDim<Dim, T...>>;

/**
 * `IfNoTensorsOfDim` can be used in a template definition to invalidate the template. To be
 * valid, no types must have `Dim` number of dimensions.  Both must be Tensor
 * types (or scalars or initializer lists).
 */
template <std::size_t Dim, typename... T>
using IfNoTensorsOfDim = IfAllConditions<!TensorsAreDim<Dim, T>...>;

/**
 * `IfTensorsMaxDim` can be used in a template definition to invalidate the template. To be
 * valid, all types must have `Dim` or less number of dimensions.  Both must be Tensor
 * types (or scalars or initializer lists).
 */
template <std::size_t Dim, typename... T>
using IfTensorsMaxDim =
    IfAllConditions<TensorsAreLessThanDim<Dim + 1, T...>, !TensorsAreLessThanDim<Dim, T...>>;

/**
 * `IfEigenInterface` can be used in a template definition to invalidate the template. To be valid,
 * type `T` must have member functions named `rows()` and `cols()`.  This enables support for
 * Eigen-like interfaces.
 */
template <typename T>
using IfEigenInterface =
    IfAllConditions<std::is_member_function_pointer<decltype(&T::rows)>::value,
                    std::is_member_function_pointer<decltype(&T::cols)>::value>;

/**
 * Returns the number of rows in a Matrix. For empty matrices, returns zero.
 *
 * @param m The Matrix to inspect.
 *
 * @return The number of rows.
 */
Size num_rows(const Matrix& m);

/**
 * Returns the number of rows in a column Vector. For empty vectors, returns
 * zero.
 *
 * @param c The Vector to inspect.
 *
 * @return The number of rows.
 */
Size num_rows(const Vector& c);

/**
 * Returns the number of columns in a Matrix. For empty matrices, returns zero.
 * @param m The Matrix to inspect.
 *
 * @return The number of columns.
 */
Size num_cols(const Matrix& m);

/**
 * Returns the number of columns in a row Vector. For empty vectors, returns
 * zero.
 *
 * @param r The Vector to inspect.
 *
 * @return The number of columns.
 */
Size num_cols(const Vector& r);

}  // namespace navtk
