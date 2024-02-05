
""" trucated matrix SVD
"""
function svd(A::Array{ComplexF64,2}, chiMax::Int, threshold::Float64)
    local F
    try
        F = svd(A, alg=LinearAlgebra.DivideAndConquer())
    catch e
        F = svd(A, alg=LinearAlgebra.QRIteration())
    end
    U, S, Vt = F.U, F.S, F.Vt

    if chiMax != 0 || threshold != 0.0
        # Truncate
        chi = min(chiMax, length(S))
        chi = min(chi, sum(cumsum(S.^2)/sum(S.^2) .<= 1-threshold))
        
        U = U[:, 1:chi]
        S = S[1:chi]
        S = S ./ norm(S)
        Vt = Vt[1:chi, :]
    end

    return U, S, Vt
end

""" trucated tensor SVD along a specified index  (1 or 3)
"""
function svd(C::Array{ComplexF64,3}, idx::Int, chiMax::Int=0, threshold::Float64=0.0)
    sizeC = size(C)
    if idx == 1
        D = reshape(C, (sizeC[1],sizeC[2]*sizeC[3]))
    elseif idx == 3
        D = reshape(C, (sizeC[1]*sizeC[2],sizeC[3]))
    else
        throw(ArgumentError("Invalid index for SVD."))
    end
    A, S, B = svd(D, chiMax, threshold)
    if idx == 1
        B = reshape(A, (size(B,1), sizeC[2], sizeC[3]))
    elseif idx == 3
        A = reshape(A, (sizeC[1], sizeC[2], size(A,2)))
    end
    
    return A, S, B
end