abstract type AbstractMPO end

### Indexing a MPS/MPO
Base.getindex(O::AbstractMPO, i) = tensors(O)[i]
function Base.setindex!(O::AbstractMPO, x, i::Int)
    O.tensors[i] = x
    return O
end

### Properties of a MPO
"""
    eltype(::AbstractMPS)

Return the element type of an MPO.
"""
Base.eltype(O::AbstractMPO) = typeof(O[1])


"""
    length(::AbstractMPS)

The length of an MPO.
"""
Base.length(O::AbstractMPO) = length(O.tensors)


"""
    dim(::AbstractMPO)

The size of the physical dimensions in an MPO.
"""
dim(O::AbstractMPO) = O.d


"""
    tensors(::AbstractMPO)

Return the tensor within an MPO
"""
tensors(O::AbstractMPO) = O.tensors

"""
    bonddim(::AbstractMPO, idx::Int)

Return the bond dimension size between idx and idx + 1. Returns nothing if
out of range.
"""
function bonddim(O::AbstractMPO, site::Int)
    (site < 1 || site > length(O)) && return nothing
    return size(O[site+1])[1]
end


"""
    maxbonddim(::AbstractMPO)

Calculate the maximum bond dimension within an MPO.
"""
function maxbonddim(O::AbstractMPO)
    D = 0
    for i = 1:length(O)-1
        D = max(D, bonddim(O, i))
    end
    return O
end


function Base.show(io::IO, O::AbstractMPO)
    println(io, "$(typeof(O))")
    for i = 1:length(O)
        println(io, "[$(i)] $(size(O[i]))")
    end
end


### Creating copies
Base.copy(O::AbstractMPO) = typeof(O)(O.d, O.N, tensors(O))
Base.deepcopy(O::AbstractMPO) = typeof(O)(copy(O.d), copy(O.N), copy(tensors(O)))
