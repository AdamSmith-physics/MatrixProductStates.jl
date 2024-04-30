abstract type AbstractMPS end

### Indexing a MPS/MPO
Base.getindex(psi::AbstractMPS, i) = tensors(psi)[i]
function Base.setindex!(psi::AbstractMPS, x, i::Int)
    psi.tensors[i] = x
    return psi
end

### Properties of a MPS/MPO
"""
    eltype(::AbstractMPS)

Return the element type of an MPS.
"""
Base.eltype(psi::AbstractMPS) = typeof(psi[1])


"""
    length(::AbstractMPS)

The length of an MPS or MPO.
"""
Base.length(psi::AbstractMPS) = length(psi.tensors)


"""
    dim(::AbstractMPS)

The size of the physical dimensions in an MPS or MPO.
"""
dim(psi::AbstractMPS) = psi.d


"""
    center(::AbstractMPS)

The orthogonal center of an MPS or MPO. Returns 0 if not set.
"""
centre(psi::AbstractMPS) = psi.centre


"""
    tensors(::AbstractMPS)

Return the tensor within an MPS or MPO
"""
tensors(psi::AbstractMPS) = psi.tensors

"""
    bonddim(::AbstractMPS, idx::Int)

Return the bond dimension size between idx and idx + 1. Returns nothing if
out of range.
"""
function bonddim(psi::AbstractMPS, site::Int)
    (site < 1 || site > length(psi)) && return nothing
    return size(psi[site+1])[1]
end


"""
    maxbonddim(::AbstractMPS)

Calculate the maximum bond dimension within an MPS.
"""
function maxbonddim(psi::AbstractMPS)
    D = 0
    for i = 1:length(psi)-1
        D = max(D, bonddim(psi, i))
    end
    return D
end


function Base.show(io::IO, M::AbstractMPS)
    println(io, "$(typeof(M))")
    for i = 1:length(M)
        asterisk = i == M.centre ? "*" : ""
        println(io, "[$(i)$(asterisk)] $(size(M[i]))")
    end
end



### Creating copies
Base.copy(psi::AbstractMPS) = typeof(psi)(psi.d, psi.N, tensors(psi), centre(psi), psi.chiMax, psi.threshold)
Base.deepcopy(psi::AbstractMPS) = typeof(psi)(copy(psi.d), copy(psi.N), copy(tensors(psi)), copy(centre(psi)),
                                                copy(psi.chiMax), copy(psi.threshold))


### Products with numbers
function Base.:*(psi::AbstractMPS, a::Number)
    phi = deepcopy(psi)
    if center(psi) != 0
        phi.tensors[center(phi)] *= a
    else
        phi.tensors[1] *= a
    end
    return phi
end
Base.:*(a::Number, psi::AbstractMPS) = *(psi, a)
Base.:/(psi::AbstractMPS, a::Number) = *(psi, 1/a)