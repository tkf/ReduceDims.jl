module ReduceDims

export on

struct OnDims{D <: Tuple, T}
    dims::D
    x::T
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

on(x::ArrayLike{N}, axes::Vararg{Union{typeof(*), typeof(:)}, N}) where N =
    OnDims(dims(axes), x)

@inline Base.mapreduce(f, op, x::OnDims; dims::Nothing=nothing, kwargs...) =
    OnDims(x.dims, mapreduce(f, op, x.x; dims=x.dims, kwargs...))

Base.dropdims(x::OnDims) = dropdims(x.x; dims=x.dims)

end # module
