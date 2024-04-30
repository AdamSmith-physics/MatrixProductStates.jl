export TFIM
export XXZ
export QuantumEast


"""
An MPO construction of the Transverse Field Ising Model Hamiltonian.

The form of H is 
```math
\\mathcal{H} = -J \\left ( \\sum_{\\langle i,j\\rangle} \\sigma_z^{(i)}\\sigma_z^{(j)} + g \\sum_i \\sigma_x^{(i)} + h \\sum_i \\sigma_z^{(i)} \\right ).
```
"""
function TFIM(N::Int, J::Real, g::Real, h::Real)::MPO

    Id = [1.0 0.0; 0.0 1.0]
    X = [0.0 1.0; 1.0 0.0]
    Y = [0.0 -im; im 0.0]
    Z = [1.0 0.0; 0.0 -1.0]

    O = MPO(N, 2)
    W = zeros(ComplexF64, 3, 2, 2, 3)
    W[1, :, :, 1] = Id
    W[1, :, :, 2] = Z
    W[1, :, :, 3] = -J*(h*Z + g*X)
    W[2, :, :, 3] = -J*Z
    W[3, :, :, 3] = Id

    for i in 1:N
        O.tensors[i] = copy(W)
    end
    O[1] = reshape(O[1][1, :, :, :], 1, 2, 2, size(O[1],4))
    O[N] = reshape(O[N][:, :, :, 3], size(O[N],1), 2, 2, 1)

    return O
end


function XXZ(N::Int, J::Real, Delta::Real)::MPO

    Id = [1.0 0.0; 0.0 1.0]
    X = [0.0 1.0; 1.0 0.0]
    Y = [0.0 -im; im 0.0]
    Z = [1.0 0.0; 0.0 -1.0]

    O = MPO(N, 2)
    W = zeros(ComplexF64, 5, 2, 2, 5)
    W[1, :, :, 1] = Id
    W[1, :, :, 2] = X
    W[1, :, :, 3] = Y
    W[1, :, :, 4] = Z
    W[2, :, :, 5] = J*X
    W[3, :, :, 5] = J*Y
    W[4, :, :, 5] = J*Delta*Z
    W[5, :, :, 5] = Id

    for i in 1:N
        O.tensors[i] = copy(W)
    end
    O[1] = reshape(O[1][1, :, :, :], 1, 2, 2, size(O[1],4))
    O[N] = reshape(O[N][:, :, :, 5], size(O[N],1), 2, 2, 1)

    return O
end


function QuantumEast(N::Int, s::Real, c::Real)::MPO

    Id = [1.0 0.0; 0.0 1.0]
    X = [0.0 1.0; 1.0 0.0]
    n = [0.0 0.0; 0.0 1.0]

    O = MPO(N, 2)
    W = zeros(ComplexF64, 3, 2, 2, 3)

    W[1, :, :, 1] = Id
    W[1, :, :, 2] = n
    W[1, :, :, 3] = c * n
    W[2, :, :, 3] = (1-2*c) * n - exp(-s)*sqrt(c*(1-c)) * X
    W[3, :, :, 3] = Id

    for i in 1:N
        O.tensors[i] = copy(W)
    end
    O[1] = reshape(O[1][1, :, :, :], 1, 2, 2, size(O[1],4))
    O[N] = reshape(O[N][:, :, :, 3], size(O[N],1), 2, 2, 1)

    return O
end