#include <memory>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <navtk/aspn.hpp>
#include <navtk/geospatial/providers/SimpleElevationProvider.hpp>
#include <navtk/geospatial/providers/SimpleProvider.hpp>
#include <navtk/geospatial/providers/SpatialMapDataProvider.hpp>
#include <navtk/geospatial/sources/ElevationSource.hpp>
#include <navtk/geospatial/sources/GeoidUndulationSource.hpp>
#include <navtk/geospatial/sources/SpatialMapDataSource.hpp>

#ifdef NAVTK_GDAL_ENABLED
#	include <navtk/geospatial/sources/GdalSource.hpp>
#	include <navtk/geospatial/detail/custom_deleters.hpp>
#	include <navtk/geospatial/detail/transformations.hpp>
#	ifndef GDAL_INCLUDE_IN_SUBFOLDER
#		include <gdal_priv.h>
#		include <ogr_spatialref.h>
#	else
#		include <gdal/gdal_priv.h>
#		include <gdal/ogr_spatialref.h>
#	endif
#endif

#include "binding_helpers.hpp"

namespace geo = navtk::geospatial;
using namespace pybind11::literals;

using geo::ElevationSource;
using geo::GeoidUndulationSource;
using geo::SimpleElevationProvider;
using geo::SimpleProvider;
using geo::SpatialMapDataProvider;
using geo::SpatialMapDataSource;
using navtk::not_null;

#ifdef NAVTK_GDAL_ENABLED
using geo::GdalSource;
using navtk::Vector;
#endif

template <class SpatialMapDataProviderBase = SpatialMapDataProvider>
class PySpatialMapDataProvider : public SpatialMapDataProviderBase,
                                 public py::trampoline_self_life_support {
public:
	using SpatialMapDataProviderBase::SpatialMapDataProviderBase;

	std::pair<bool, double> lookup_datum(double latitude, double longitude) const override {
		PYBIND11_OVERRIDE_PURE(PARAMS(std::pair<bool, double>),
		                       SpatialMapDataProviderBase,
		                       lookup_datum,
		                       latitude,
		                       longitude);
	}
};

template <class SpatialMapDataSourceBase = SpatialMapDataSource>
class PySpatialMapDataSource : public SpatialMapDataSourceBase,
                               public py::trampoline_self_life_support {
public:
	std::pair<bool, double> lookup_datum(double latitude, double longitude) const override {
		PYBIND11_OVERRIDE_PURE(PARAMS(std::pair<bool, double>),
		                       SpatialMapDataSourceBase,
		                       lookup_datum,
		                       latitude,
		                       longitude);
	}
};

template <class ElevationSourceBase = ElevationSource>
class PyElevationSource : public PySpatialMapDataSource<ElevationSourceBase> {
public:
	AspnMeasurementAltitudeReference get_output_vertical_reference_frame() const override {
		PYBIND11_OVERRIDE(AspnMeasurementAltitudeReference,
		                  ElevationSourceBase,
		                  get_output_vertical_reference_frame, );
	}

	void set_output_vertical_reference_frame(AspnMeasurementAltitudeReference new_ref) override {
		PYBIND11_OVERRIDE_PURE(
		    void, ElevationSourceBase, set_output_vertical_reference_frame, new_ref);
	}
};

void add_geospatial_functions(pybind11::module& m) {
	m.doc() = "Classes and utilties for reading geographic spatial map data.";

	CLASS(SpatialMapDataSource, PySpatialMapDataSource<>)
	CTOR_NODOC_DEFAULT
	METHOD(SpatialMapDataSource, lookup_datum, "latitude"_a, "longitude"_a)
	CDOC(SpatialMapDataSource);

	CLASS(ElevationSource, SpatialMapDataSource, PyElevationSource<>)
	METHOD_VOID(ElevationSource, get_output_vertical_reference_frame)
	METHOD(ElevationSource, set_output_vertical_reference_frame, "new_ref"_a)
	CDOC(ElevationSource);

	// clang-format off
	CLASS(GeoidUndulationSource, SpatialMapDataSource)
	.def_static("get_shared",
	     &GeoidUndulationSource::get_shared,
	     PROCESS_DOC(GeoidUndulationSource_get_shared), "path"_a = std::string("WW15MGH.GRD"))
	METHOD(GeoidUndulationSource, set_chunk_size, "size"_a)
	METHOD_VOID(GeoidUndulationSource, get_chunk_size)
	CDOC(GeoidUndulationSource);

	CLASS(SpatialMapDataProvider, PySpatialMapDataProvider<>)
	CTOR(SpatialMapDataProvider, std::shared_ptr<SpatialMapDataSource>, "src"_a)
	CTOR_OVERLOAD(SpatialMapDataProvider,
	              std::vector<not_null<std::shared_ptr<SpatialMapDataSource>>>,
	              _2,
	              "srcs"_a = std::vector<not_null<std::shared_ptr<SpatialMapDataSource>>>{})
	METHOD(SpatialMapDataProvider, add_source, "src"_a)
	METHOD(SpatialMapDataProvider, lookup_datum, "latitude"_a, "longitude"_a)
	CDOC(SpatialMapDataProvider);

	CLASS(SimpleProvider, SpatialMapDataProvider)
	// have to redefine constructor bindings because pybind classes don't inherit constructors
	CTOR_NODOC(std::shared_ptr<SpatialMapDataSource>, "src"_a)
	CTOR_NODOC(std::vector<not_null<std::shared_ptr<SpatialMapDataSource>>>,
	           "srcs"_a = std::vector<not_null<std::shared_ptr<SpatialMapDataSource>>>{})
	CDOC(SimpleProvider);

	CLASS(SimpleElevationProvider, SimpleProvider)
	CTOR(SimpleElevationProvider,
	     PARAMS(std::shared_ptr<ElevationSource>, AspnMeasurementAltitudeReference),
	     "src"_a,
		 py::arg_v("out_ref", ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE, "AspnMeasurementAltitudeReference.ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE"))
	CTOR_OVERLOAD(SimpleElevationProvider,
	              PARAMS(std::vector<not_null<std::shared_ptr<ElevationSource>>>,
	                     AspnMeasurementAltitudeReference),
	              _2,
	              "srcs"_a    = std::vector<not_null<std::shared_ptr<ElevationSource>>>{},
		 		  py::arg_v("out_ref", ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE, "AspnMeasurementAltitudeReference.ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE"))
	CDOC(SimpleElevationProvider);

#ifdef NAVTK_GDAL_ENABLED

	auto gdal_source = CLASS(GdalSource, ElevationSource);
	ENUM_SCOPED(MapType, GdalSource, gdal_source)
	CHOICE_SCOPED(MapType, GdalSource, GEOTIFF)
	CHOICE_SCOPED(MapType, GdalSource, DTED).finalize();

	gdal_source CTOR(GdalSource,
	                 PARAMS(const std::string &,
	                        GdalSource::MapType,
	                        AspnMeasurementAltitudeReference,
	                        AspnMeasurementAltitudeReference,
	                        unsigned int,
							const std::string &
							),
	                 "map_path"_a,
	                 "type"_a,
					 py::arg_v("in_ref", ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE, "AspnMeasurementAltitudeReference.ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE"),
					 py::arg_v("out_ref",ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE, "AspnMeasurementAltitudeReference.ASPN_MEASUREMENT_ALTITUDE_REFERENCE_HAE"),
	                 "num_tiles"_a = 10,
					 "undulation_path"_a = "WW15MGH.GRD"
					 )
	METHOD_OVERLOAD_CONST(GdalSource, lookup_datum, PARAMS(double, double), , "latitude"_a, "longitude"_a)
	METHOD_OVERLOAD_CONST(GdalSource, lookup_data, PARAMS(const Vector&, const Vector&), , "latitudes"_a, "longitudes"_a)
	METHOD_OVERLOAD_CONST(GdalSource, is_stored, PARAMS(const std::string &), , "filename"_a)
	METHOD_OVERLOAD_CONST_VOID(GdalSource, get_size, )
	METHOD_OVERLOAD_CONST_VOID(GdalSource, get_cached_num, )
	CDOC(GdalSource);
	// clang-format on

#endif
}
