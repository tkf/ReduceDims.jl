module ReduceDims

using Base: mapreducedim!, reduce_empty, reducedim_initarray, reducedim_init

export along

struct OnDims{D <: Tuple, T}
    dims::D
    x::T
end

struct MapReduce{D <: Tuple, F, B, T, K <: NamedTuple}
    dims::D
    f::F
    op::B
    x::T
    kwargs::K
end

const ArrayLike{N} = Union{
    AbstractArray{<:Any, N},
    Broadcast.Broadcasted{<:Any, <:NTuple{N}},
}

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

# Support `mean(::MapReduce)` etc.
# Broadcast.broadcastable(x::MapReduce) = copy(x)

end # module
