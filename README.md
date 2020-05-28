# CarbonFluxTools

The primary purpose of this module is to organize utilitarian code for low
level procedures having to do with my research of carbon fluxes.

## Installation Instructions
To install, clone this repository and run
`$ pip install carbonfluxtools`

## Functionality
There are currently two available sub-modules
1. `carbonfluxtools.io`
2. `carbonfluxtools.computation`
3. `carbonfluxtools.plotting`

In `carbonfluxtools.io`, we find functions for reading in `bpch` files, GOSAT
observations, and inversion output.

In `carbonfluxtools.computation`, we find function for computing posterior flux
fields and regional scale factors or fluxes.

In `carbonfluxtools.plotting`, there are a few functions for making geographic
plots to see where things are.

## Generating NetCDF Files from BPCH Files
In order to make custom carbon flux input files, we need to be able to transform the original `bpch` files into `.nc`.

```
import carbonfluxtools.io as cio

# define tracer and diag paths
DIAGINFO_PATH = BASE_PATH + '/data/JULES/diaginfo.dat'
TRACERINFO_PATH = BASE_PATH + '/data/JULES/tracerinfo.dat'

# define input files, i.e. the bpch files
bpch_files = sorted(
    glob(BASE_PATH + '/data/NEE_fluxes/nep.geos.4x5.*'),
    key = lambda x: int(x[-3:])
)

# define output directory
BASE_NETCDF = BASE_PATH + '/data/NEE_fluxes_nc'

# save the first month of netcdf files
cio.generate_nc_files(
    bpch_files=bpch_files,
    output_dir=BASE_NETCDF,
    tracer_path=TRACERINFO_PATH,
    diag_path=DIAGINFO_PATH
)
```

The NetCDF files will not be available for use in `BASE_NETCDF`. Note, the `cio.generate_nc_files` function can take `co2_var_nm` and `dims` arguments, which specify the variable name under which the CO2 field is stored and the dimensions of the data.