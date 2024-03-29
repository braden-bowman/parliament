
import sys

from parliament._version import __version__  # noqa: F401
# from parliament.utils.logging import LOGGER
# from parliament.coordinates import Coordinate
# from parliament.structures import (
# )

# from parliament.collections import FeatureCollection, Track
# from parliament.utils.conditional_imports import ConditionalPackageInterceptor


# ConditionalPackageInterceptor.permit_packages(
#     {
#         'geopandas': 'geopandas>=0.13,<1',
#         'h3': 'h3>=3.7,<4',
#         'mgrs': 'mgrs>=1.4.5,<2',
#         'pandas': 'pandas>=2,<3',
#         'plotly': 'plotly>=5,<6',
#         'pyproj': 'pyproj>=3.6,<4',
#         'scipy': 'scipy>=3.0.7,<4.0',
#     }
# )
# sys.meta_path.append(ConditionalPackageInterceptor)  # type: ignore

# __all__ = [
#     'Coordinate',
#     'FeatureCollection',
#     'GeoBox',
#     'GeoCircle',
#     'GeoEllipse',
#     'GeoLineString',
#     'GeoPoint',
#     'GeoPolygon',
#     'GeoRing',
#     'Track',
#     'LOGGER',
# ]