from __future__ import annotations

from typing import Union, Tuple, Dict, TYPE_CHECKING

import warnings

import numpy as np

from pyresample import SwathDefinition, AreaDefinition
from pyresample.kd_tree import get_neighbour_info, get_sample_from_neighbour_info
import msgpack
import msgpack_numpy

if TYPE_CHECKING:
    from .raster_geometry import RasterGeometry
    from .raster_grid import RasterGrid
    from .raster_geolocation import RasterGeolocation
    from .raster import Raster

class KDTree:
    def __init__(
            self,
            source_geometry: RasterGeometry,
            target_geometry: RasterGeometry,
            radius_of_influence: float = None,
            neighbours: int = 1,
            epsilon: float = 0,
            reduce_data: bool = True,
            nprocs: int = 1,
            segments=None,
            resample_type="nn",
            valid_input_index: np.ndarray = None, 
            valid_output_index: np.ndarray = None, 
            index_array: np.ndarray = None, 
            distance_array: np.ndarray = None,
            **kwargs):
        from .raster_geometry import RasterGeometry
        from .raster_geolocation import RasterGeolocation
        from .raster_grid import RasterGrid

        self.neighbours = neighbours
        self.epsilon = epsilon
        self.reduce_data = reduce_data
        self.nprocs = nprocs
        self.segments = segments
        self.resample_type = resample_type

        # validate destination geometry
        if not isinstance(target_geometry, RasterGeometry):
            raise TypeError("destination geometry must be a RasterGeometry object")

        # build pyresample data structure for source
        if isinstance(source_geometry, RasterGeolocation):
            # transform swath geolocation arrays to lat/lon
            source_lat, source_lon = source_geometry.latlon_matrices
            self.source_geo_def = SwathDefinition(lats=source_lat, lons=source_lon)
        elif isinstance(source_geometry, RasterGrid):
            source_proj4 = source_geometry.proj4

            source_area_extent = [
                source_geometry.x_min,
                source_geometry.y_min,
                source_geometry.x_max,
                source_geometry.y_max
            ]

            source_rows, source_cols = source_geometry.shape

            self.source_geo_def = AreaDefinition(
                area_id="source",
                description=source_proj4,
                proj_id=source_proj4,
                projection=source_proj4,
                width=source_cols,
                height=source_rows,
                area_extent=source_area_extent
            )
        else:
            raise ValueError("source type must be swath or grid")

        self.source_geometry = source_geometry
        self.target_geometry = target_geometry

        # build pyresample data structure for destination
        if isinstance(target_geometry, RasterGeolocation):
            # transform grid geolocation arrays to lat/lon
            dest_lat, dest_lon = target_geometry.latlon_matrices
            self.target_geo_def = SwathDefinition(lats=dest_lat, lons=dest_lon)
        elif isinstance(target_geometry, RasterGrid):
            destination_proj4 = target_geometry.proj4

            # Area extent as a list of ints (LL_x, LL_y, UR_x, UR_y)
            destination_area_extent = [
                target_geometry.x_min,
                target_geometry.y_min,
                target_geometry.x_max,
                target_geometry.y_max
            ]

            # destination_proj_dict = proj4_str_to_dict(destination_proj4)
            destination_rows, destination_cols = target_geometry.shape

            self.target_geo_def = AreaDefinition(
                area_id="destination",
                description=destination_proj4,
                proj_id=destination_proj4,
                projection=destination_proj4,
                width=destination_cols,
                height=destination_rows,
                area_extent=destination_area_extent
            )
        else:
            raise ValueError("destination type must be swath or grid")

        if radius_of_influence is None:
            source_cell_size = source_geometry.cell_size_meters
            destination_cell_size = target_geometry.cell_size_meters

            self.radius_of_influence = CELL_SIZE_TO_SEARCH_DISTANCE_FACTOR * np.nanmax([
                source_cell_size,
                destination_cell_size
            ])
        else:
            self.radius_of_influence = radius_of_influence

        self.radius_of_influence = float(self.radius_of_influence)

        if any(item is None for item in [valid_input_index, valid_output_index, index_array, distance_array]):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                self.valid_input_index, self.valid_output_index, self.index_array, self.distance_array = \
                    get_neighbour_info(
                        source_geo_def=self.source_geo_def,
                        target_geo_def=self.target_geo_def,
                        radius_of_influence=self.radius_of_influence,
                        neighbours=self.neighbours,
                        epsilon=self.epsilon,
                        reduce_data=self.reduce_data,
                        nprocs=self.nprocs,
                        segments=self.segments,
                        **kwargs
                    )
        else:
            self.valid_input_index = valid_input_index
            self.valid_output_index = valid_output_index
            self.index_array = index_array
            self.distance_array = distance_array
    
    def to_dict(self, output_dict: Dict = None) -> Dict:
        if output_dict is None:
            output_dict = {}

        output_dict.update({
            "source_geometry": self.source_geometry.to_dict(),
            "target_geometry": self.target_geometry.to_dict(),
            "neighbours": self.neighbours,
            "epsilon": self.epsilon,
            "reduce_data": self.reduce_data,
            "nprocs": self.nprocs,
            "segments": self.segments,
            "resample_type": self.resample_type,
            "radius_of_influence": self.radius_of_influence,
            "valid_input_index": self.valid_input_index,
            "valid_output_index": self.valid_output_index,
            "index_array": self.index_array,
            "distance_array": self.distance_array,
            "neighbors": self.neighbours
        })

        return output_dict
    
    def save(self, filename: str):
        with open(filename, "wb") as file:
            file.write(msgpack.packb(self.to_dict(), default=msgpack_numpy.encode))
    
    @classmethod
    def from_dict(cls, input_dict: Dict) -> KDTree:
        from .kdtree import KDTree
        from .raster_geometry import RasterGeometry

        return KDTree(
            source_geometry=RasterGeometry.from_dict(input_dict["source_geometry"]),
            target_geometry=RasterGeometry.from_dict(input_dict["target_geometry"]),
            radius_of_influence=input_dict["radius_of_influence"],
            neighbors=input_dict["neighbors"],
            epsilon=input_dict["epsilon"],
            reduce_data=input_dict["reduce_data"],
            nprocs=input_dict["nprocs"],
            segments=input_dict["segments"],
            resample_type=input_dict["resample_type"],
            valid_input_index=input_dict["valid_input_index"],
            valid_output_index= input_dict["valid_output_index"],
            index_array=input_dict["index_array"],
            distance_array=input_dict["distance_array"]
        )
    
    @classmethod
    def load(cls, filename: str) -> KDTree:
        with open(filename, "rb") as file:
            return cls.from_dict(msgpack.unpackb(file.read(), object_hook=msgpack_numpy.decode))

    def resample(
            self,
            source,
            fill_value=0,
            **kwargs):
        from .raster import Raster

        source = np.array(source)

        if not source.shape == self.source_geo_def.shape:
            raise ValueError("source data does not match source geometry")

        bool_conversion = str(source.dtype) == "bool"

        if bool_conversion:
            source = source.astype(np.uint16)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            resampled_data = get_sample_from_neighbour_info(
                resample_type=self.resample_type,
                output_shape=self.target_geo_def.shape,
                data=source,
                valid_input_index=self.valid_input_index,
                valid_output_index=self.valid_output_index,
                index_array=self.index_array,
                distance_array=self.distance_array,
                fill_value=fill_value,
                **kwargs
            )

        if bool_conversion:
            resampled_data = resampled_data.astype(bool)

        output_raster = Raster(
            array=resampled_data,
            geometry=self.target_geometry,
            nodata=fill_value
        )

        return output_raster