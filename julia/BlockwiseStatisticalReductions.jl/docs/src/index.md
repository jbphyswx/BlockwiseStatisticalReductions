# BlockwiseStatisticalReductions.jl

Mergeable statistics of an N-D tensor over many window sizes at once, computed by a planned tree of
reductions that touches the data as close to once as possible, on CPU, GPU and distributed backends.
