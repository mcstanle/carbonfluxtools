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