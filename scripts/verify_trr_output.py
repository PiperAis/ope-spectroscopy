import xarray as xr
from opespec.utilities import load_config

cfg = load_config('CrSBr-4')

file_name = '04.nc'

golden_directory = cfg['processed_trr_dir'] / 'golden'

golden_path = golden_directory / file_name

golden = xr.open_dataset(golden_path)

golden_array = golden['processed']

result_path = cfg['processed_trr_dir'] / 'test' / '04.nc'

new_result = xr.open_dataset(result_path)

new_result_array = new_result['processed']

xr.testing.assert_allclose(new_result_array, golden_array)