using Revise
using TestItems
using MatrixProductStates
using LinearAlgebra

@testitem "Test TFIM MPO" begin

    using LinearAlgebra

    function TFIM_matrix(N::Int, J::Real, h::Real)
        X = [0 1; 1 0]
        Z = [1 0; 0 -1]

        H_mat = zeros(ComplexF64, 2^N, 2^N)

        # Z terms
        for i in 1:N
            Z_term = Matrix(I, 2^(i-1), 2^(i-1))
            Z_term = kron(Z_term, Z)
            Z_term = kron(Z_term, Matrix(I,2^(N-i), 2^(N-i)))
            H_mat += h * Z_term
        end

        # X terms
        for i in 1:N-1
            X_term = Matrix(I, 2^(i-1), 2^(i-1))
            X_term = kron(X_term, X)
            X_term = kron(X_term, X)
            X_term = kron(X_term, Matrix(I,2^(N-i-1), 2^(N-i-1)))
            H_mat += J * X_term
        end

        return H_mat
    end

    for N in 1:6
        for _ in 1:10
            J = rand()
            h = rand()

            H = TFIM(N, J, h);

            # exact matrix to compare to
            H_mat = TFIM_matrix(N, J, h);

            @test to_matrix(H) ≈ H_mat
        end
    end

end