module ReduceDims

using Base: mapreducedim!, reduce_empty, reducedim_initarray, reducedim_init

export along

const ArrayLike{N} = Union{
    AbstractArray{<:Any, N},
    Broadcast.Broadcasted{<:Any, <:NTuple{N}},
}

abstract type Axed{T, N} <: AbstractArray{T, N} end
# TODO: better name

Base.size(x::Axed) = size(x.x)

struct OnDims{T, N, D <: Tuple, X <: ArrayLike{N}} <: Axed{T, N}
    dims::D
    x::X

    function OnDims{T, N, D, X}(dims::D, x::X) where {T, N, D <: Tuple, X <: ArrayLike{N}}
        @assert length(dims) <= N
        return new{T, N, D, X}(dims, x)
    end
end

OnDims(dims, x) =
    OnDims{eltype(x), ndims(x), typeof(dims), typeof(x)}(dims, x)

Base.getindex(x::OnDims, I...) = x.x[I...]

struct MapReduce{
    N,
    D <: Tuple,
    F,
    B,
    X <: ArrayLike{N},
    K <: NamedTuple,
} <: Axed{Any, N}

    dims::D
    f::F
    op::B
    x::X
    kwargs::K

    function MapReduce{N, D, F, B, X, K}(dims::D, f::F, op::B, x::X, kwargs::K) where {N, D, F, B, X, K}
        @assert length(dims) <= N
        return new{N, D, F, B, X, K}(dims, f, op, x, kwargs)
    end
end

MapReduce(dims::D, f::F, op::B, x::X, kwargs::K) where {D, F, B, X, K} =
    MapReduce{ndims(x), D, F, B, X, K}(dims, f, op, x, kwargs)

# TODO: improve
Base.show(io::IO, x::MapReduce) = print(io, "MapReduce(...)")
Base.show(io::IO, ::MIME"text/plain", x::MapReduce) = show(io, x)

@inline foldlargs(op, x) = x
@inline foldlargs(op, x1, x2, xs...) = foldlargs(op, op(x1, x2), xs...)

dims(axes::Tuple{Vararg{Union{typeof(*), typeof(:)}}}) =
    foldlargs((1, ()), axes...) do (n, dims), ax
        (n + 1, ax === (*) ? (dims..., n) : dims)
    end[2]

along(x::ArrayLike{N}, axes::Vararg{Union{typeof(*), typeof(:)}, N}) where N =
    OnDims(dims(axes), x)

Base.mapreduce(f, op, x::OnDims; dims::Nothing=nothing, kwargs...) =
    MapReduce(x.dims, f, op, x.x, kwargs.data)

Base.dropdims(x::MapReduce) = dropdims(copy(x); dims=x.dims)
Base.dropdims(x::OnDims) = dropdims(x.x; dims=x.dims)

Base.similar(x::MapReduce) =
    if haskey(x.kwargs, :init)
        reducedim_initarray(x.x, x.dims, x.kwargs.init)
    else
        reducedim_init(x.f, x.op, x.x, x.dims)
    end
# TODO: don't do extra work in `similar`

Base.copy(x::MapReduce) = copyto!(similar(x), x)

fillinit!(dest, x) =
    if haskey(x.kwargs, :init)
        fill!(dest, x.kwargs.init)
    else
        fill!(dest, reduce_empty(x.op, eltype(dest)))
    end

Base.copyto!(dest::AbstractArray, x::MapReduce) =
    mapreducedim!(x.f, x.op, fillinit!(dest, x), x.x)

Broadcast.broadcasted(::typeof(identity), x::MapReduce) = x
Broadcast.materialize!(dest, x::MapReduce) = copyto!(dest, x)

# Support `mean(::OnDims)` etc.
Broadcast.broadcastable(x::MapReduce) = copy(x)

end # module
