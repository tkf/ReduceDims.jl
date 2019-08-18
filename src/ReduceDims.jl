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

showapply(io, x::OnDims) =
    print(io, "along(", "::", typeof(x.x), ", ", x.dims, ")")

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
function Base.show(io::IO, x::MapReduce)
    print(io, "mapreduce(", x.f, ", ", x.op, ", ")
    showapply(io, OnDims(x.dims, x.x))
    if !isempty(x.kwargs)
        print(io, "; ...")
    end
    print(io, ")")
end
Base.show(io::IO, ::MIME"text/plain", x::MapReduce) = show(io, x)

@inline foldlargs(op, x) = x
@inline foldlargs(op, x1, x2, xs...) = foldlargs(op, op(x1, x2), xs...)

countisa(T, xs) =
    foldlargs(0, xs...) do n, x
        x isa T ? n + 1 : n
    end

const OpMarker = typeof(&)
const OneOrMore = typeof(+)
const SpecType = Union{typeof(:), OpMarker, OneOrMore}

@inline function asdims(N, axes::Tuple{Vararg{SpecType}})
    N < length(axes) && error("Too many axes specifiers.")
    isempty(axes) && return ()
    axes[1] isa OneOrMore && error("First axes specifier cannot be `+`.")

    oom = countisa(OneOrMore, axes)
    if oom > 1
        error("More than one `+` is specified.")
    elseif oom === 0
        error("Number of axes specifier does not match with ndim.\n",
              "Note: Use `+` to avoid repeating `:` or `&`.")
    end

    pre, post = foldlargs(((), (), Val(false)), axes...) do (pre, post, seen), ax
        if seen === Val(true)
            (pre, (post..., ax), Val(true))
        elseif ax isa OneOrMore
            (pre, post, Val(true))
        else
            ((pre..., ax), post, Val(false))
        end
    end
    predims = asdims(pre)
    postdims = asdims(post) .+ (N - length(post))
    if last(pre) === (:)
        middims = ()
    else
        middims = ntuple(x -> x + length(pre), N - (length(pre) + length(post)))
    end
    return (predims..., middims..., postdims...)
end

asdims(axes::Tuple{Vararg{Union{typeof(:), OpMarker}}}) =
    foldlargs((1, ()), axes...) do (n, dims), ax
        (n + 1, ax isa OpMarker ? (dims..., n) : dims)
    end[2]

along(x::ArrayLike{N}, axes::Vararg{Union{typeof(:), OpMarker}, N}) where N =
    OnDims(asdims(axes), x)

along(x::ArrayLike{N}, axes::SpecType...) where N =
    OnDims(asdims(N, axes), x)

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

function Base.copyto!(dest::AbstractArray, x::MapReduce)
    if ndims(dest) === ndims(x.x)
        y = dest
    elseif ndims(dest) === ndims(x.x) - length(x.dims)
        newdims, dropped = foldlargs(
            ((), (), 1),
            size(x.x)...,
        ) do (newdims, dropped, n), s
            ((newdims..., n in x.dims ? 1 : s),
             n in x.dims ? dropped : (dropped..., s),
             n + 1)
        end
        @assert dropped === size(dest)  # TODO: proper error
        y = reshape(dest, newdims)
    end
    mapreducedim!(x.f, x.op, fillinit!(y, x), x.x)
    return dest
end

Broadcast.broadcasted(::typeof(identity), x::MapReduce) = x
Broadcast.materialize!(dest, x::MapReduce) = copyto!(dest, x)

# Support `mean(::OnDims)` etc.
Broadcast.broadcastable(x::MapReduce) = copy(x)

end # module
