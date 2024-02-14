# Provides utilities to contract and manipulate tensors

"""
    contract(x::Array{}, y::Array{}, idx1::Int, idx2::Int)
    contract(x::Array{}, y::Array{}, idxs1::Vector{Int}, idxs2::Vector{Int})

Contract two tensors across specified indexs.
"""
@inline function contract(x::Array{}, y::Array{}, idx1::Int, idx2::Int)
    sz = length(size(x))
    dims1 = [i == idx1 ? 0 : i for i = 1:sz]
    dims2 = [j == idx2 ? 0 : sz + j for j = 1:length(size(y))]
    return tensorcontract(x, dims1, y, dims2)
end

@inline function contract(x::Array{}, y::Array{}, idxs1::Vector{Int}, idxs2::Vector{Int})
    length(idxs1) != length(idxs2) && error("The length of contracting indexs differ.")
    labels = [-i for i = 1:length(idxs1)]
    dims1 = [i in idxs1 ? -findall(x -> x == i, idxs1)[1] : i for i = 1:length(size(x))]
    dims2 = [i in idxs2 ? -findall(x -> x == i, idxs2)[1] : i+length(size(x)) for i = 1:length(size(y))]
    return tensorcontract(x, dims1, y, dims2)
end