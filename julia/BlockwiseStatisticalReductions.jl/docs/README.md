TODO:

[ ] THOROUGH DOCUMENTATION! EXAMPLES, DOCS SECTIONS, THE WORKS!!!
[ ] attach metadata to outputs (coordinates etc)
[ ] verbose plan progress logging (support debug levels, debug warn info etc, custom loggers, custom exceptions, etc)
[ ] plan pretty print show(), print(), display(), summary() methods
[ ] plan optimizations to fully reduce repeated computations and allocations, and allow for prioritizing one over the other
[ ] mixed block and rolling reductions
[ ] support external kernel reductions (user supplied function)
    - all that is requird is inferring the output, or , the user supplying the output contract like some LinearAlgebra and like xarray dask stuff to nowadays, eigen, etc
[ ] massively expand and enhance documentation
[ ] stop passing symbols where we dont need to and dispatch on function objects directly or Val objects for symbols that don't tie to functions
    - e.g. do we need to use :mean instead of Statistics.mean? this ties in with supporting supplied fucntions as well
[ ] Automatic ideal buffer size/shape from plan
[ ] reduce to ND histograms (fixed bins, or fixed n_1 × n_2 × ... × n_n bins)
[ ] Streaming data support from Zarr and NetCDF or any other stream, streaming outputs to disk
[ ] Fancy progress logging (plain text and REPL) for long running computations.
[ ] rules for keeping or dropping data (e.g. all 0 drop) and support for pruning the plan tree/dag accordingly
[ ] Fast optinos structs for controlling these kinds of options across the package where it's not just simple single or double booleans.
