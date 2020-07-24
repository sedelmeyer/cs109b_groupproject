import logging
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

# Initialize ``logging.NullHandler`` for caproj package
logging.getLogger('caproj').addHandler(logging.NullHandler())
