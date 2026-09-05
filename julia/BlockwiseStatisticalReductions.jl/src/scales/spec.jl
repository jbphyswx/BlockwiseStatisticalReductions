# Size bounds are `nothing` (1 / the axis extent), an `Integer` (cells) or a `Length` (physical units).

abstract type SizeGenerator end

"Explicit window sizes in cells."
struct Sizes{V<:AbstractVector{<:Integer}} <: SizeGenerator
    sizes::V
end
"Powers of two within `[min, max]`."
struct Dyadic{L,U} <: SizeGenerator
    min::L
    max::U
end
Dyadic(; min = nothing, max = nothing) = Dyadic(min, max)
"Products of powers of `primes` within `[min, max]`."
struct Smooth{P<:Tuple,L,U} <: SizeGenerator
    primes::P
    min::L
    max::U
end
Smooth(primes::Tuple = (2, 3); min = nothing, max = nothing) = Smooth(primes, min, max)
"Every integer within `[min, max]`."
struct Every{L,U} <: SizeGenerator
    min::L
    max::U
end
Every(; min = nothing, max = nothing) = Every(min, max)
"Divisors of the axis extent within `[min, max]`."
struct Divisors{L,U} <: SizeGenerator
    min::L
    max::U
end
Divisors(; min = nothing, max = nothing) = Divisors(min, max)
"Exactly one size; `Fixed(1)` leaves the axis unreduced."
struct Fixed <: SizeGenerator
    size::Int
end
"At most `budget` evenly spaced members of `gen`, endpoints kept."
struct Subsample{G<:SizeGenerator} <: SizeGenerator
    gen::G
    budget::Int
end

abstract type Placement end
"Windows tile the axis (stride equals size)."
struct Tiled <: Placement end
"Windows every `stride` cells."
struct Stride <: Placement
    stride::Int
end
"Windows overlapping by `fraction` of their size."
struct Overlap <: Placement
    fraction::Rational{Int}
end
"Windows at every cell."
struct Dense <: Placement end
"Windows at explicit origins."
struct Anchors{V<:AbstractVector{<:Integer}} <: Placement
    origins::V
end
"`count` windows from the axis start to its end, spaced by integer division."
struct Spread <: Placement
    count::Int
end

abstract type Combination end
"""
    Isotropic(axes = nothing)

One common size on the coupled `axes` (positions or names; `nothing` couples every reduced axis); the
remaining reduced axes vary independently.
"""
struct Isotropic{A} <: Combination
    axes::A
end
Isotropic() = Isotropic(nothing)
"Every combination of the per-axis sizes."
struct Product <: Combination end
"The i-th size of every reduced axis together."
struct Zip <: Combination end

"""
    ScaleSet(axes; combine = Isotropic(), placement = Tiled(), edge = Truncate(), filter = Returns(true),
             include_full = false, min_elements = 1, max_elements = typemax(Int))

Which windows to compute. `axes` is one generator for every axis, a tuple of generators (one per axis) or a
`NamedTuple` keyed by axis name (unnamed axes are `Fixed(1)`); an `Integer` stands for `Fixed`, a vector of
integers for `Sizes`. `placement` is one `Placement` or a tuple with one per axis. `filter` receives each
candidate `Window`.
"""
struct ScaleSet{A,C<:Combination,P,E<:EdgePolicy,F}
    axes::A
    combine::C
    placement::P
    edge::E
    filter::F
    include_full::Bool
    min_elements::Int
    max_elements::Int
end
ScaleSet(axes; combine::Combination = Isotropic(), placement = Tiled(), edge::EdgePolicy = Truncate(),
         filter = Returns(true), include_full::Bool = false, min_elements::Integer = 1, max_elements::Integer = typemax(Int)) =
    ScaleSet(axes, combine, placement, edge, filter, include_full, Int(min_elements), Int(max_elements))
