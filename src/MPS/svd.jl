
""" trucated matrix SVD
"""
function svd_truncated(A::Array{ComplexF64,2}, chiMax::Int, threshold::Float64; normalised::Bool=true)
    local F
    try
        F = svd(A, alg=LinearAlgebra.DivideAndConquer())
    catch e
        F = svd(A, alg=LinearAlgebra.QRIteration())
    end
    U, S, Vt = F.U, F.S, F.Vt

    if chiMax != 0 || threshold != 0.0
        # Truncate
        chi = length(S)
        if chiMax != 0
            chi = min(chiMax, chi)
        end
        if threshold != 0.0
            println(cumsum(S.^2)/sum(S.^2))
            println(sum(cumsum(S.^2)/sum(S.^2) .<= 1-threshold))
            chi = min(chi, sum(cumsum(S.^2)/sum(S.^2) .<= 1-threshold)+1)
        end
        
        U = U[:, 1:chi]
        S = S[1:chi]
        Vt = Vt[1:chi, :]
    end

    # can normalise even without truncation
    if normalised
        S /= norm(S)  # add flag to decide when to normalize!
    end

    return U, S, Vt
end

""" trucated tensor SVD along a specified index  (1 or 3)
"""
function svd_tensor(C::Array{ComplexF64,3}, idx::Int; chiMax::Int=0, threshold::Float64=0.0, normalised::Bool=true)
    sizeC = size(C)
    if idx == 1
        D = reshape(C, (sizeC[1],sizeC[2]*sizeC[3]))
    elseif idx == 3
        D = reshape(C, (sizeC[1]*sizeC[2],sizeC[3]))
    else
        throw(ArgumentError("Invalid index for SVD."))
    end
    A, S, B = svd_truncated(D, chiMax, threshold, normalised=normalised)
    if idx == 1
        B = reshape(B, (size(B,1), sizeC[2], sizeC[3]))
    elseif idx == 3
        A = reshape(A, (sizeC[1], sizeC[2], size(A,2)))
    end
    
    return A, S, B
end