using Revise
using TestItems
using MatrixProductStates
using BenchmarkTools
using LinearAlgebra

@testitem "Test TFIM MPO" begin

    using LinearAlgebra

    function TFIM_matrix(N::Int, J::Real, g::Real, h::Real)
        X = [0 1; 1 0]
        Y = [0 -im; im 0]
        Z = [1 0; 0 -1]

        H_mat = zeros(ComplexF64, 2^N, 2^N)

        # Z terms
        for i in 1:N
            Z_term = Matrix(I, 2^(i-1), 2^(i-1))
            Z_term = kron(Z_term, Z)
            Z_term = kron(Z_term, Matrix(I,2^(N-i), 2^(N-i)))
            H_mat += -J*h * Z_term
        end

        # X terms
        for i in 1:N
            X_term = Matrix(I, 2^(i-1), 2^(i-1))
            X_term = kron(X_term, X)
            X_term = kron(X_term, Matrix(I,2^(N-i), 2^(N-i)))
            H_mat += -J*g * X_term
        end

        # Z coupling
        for i in 1:N-1
            Z_term = Matrix(I, 2^(i-1), 2^(i-1))
            Z_term = kron(Z_term, Z)
            Z_term = kron(Z_term, Z)
            Z_term = kron(Z_term, Matrix(I,2^(N-i-1), 2^(N-i-1)))
            H_mat += -J * Z_term
        end

        return H_mat
    end

    for N in 1:6
        for _ in 1:10
            J = rand()
            g = rand()
            h = rand()

            H = TFIM(N, J, g, h);

            # exact matrix to compare to
            H_mat = TFIM_matrix(N, J, g, h);

            @test to_matrix(H) ≈ H_mat
        end
    end
end

@testitem "Test XXZ MPO" begin

    using LinearAlgebra

    function XXZ_matrix(N::Int, J::Real, Delta::Real)
        X = [0 1; 1 0]
        Y = [0 -im; im 0]
        Z = [1 0; 0 -1]

        H_mat = zeros(ComplexF64, 2^N, 2^N)

        # XX term
        for i in 1:N-1
            X_term = Matrix(I, 2^(i-1), 2^(i-1))
            X_term = kron(X_term, X)
            X_term = kron(X_term, X)
            X_term = kron(X_term, Matrix(I,2^(N-i-1), 2^(N-i-1)))
            H_mat += J * X_term
        end

        # YY term
        for i in 1:N-1
            Y_term = Matrix(I, 2^(i-1), 2^(i-1))
            Y_term = kron(Y_term, Y)
            Y_term = kron(Y_term, Y)
            Y_term = kron(Y_term, Matrix(I,2^(N-i-1), 2^(N-i-1)))
            H_mat += J * Y_term
        end

        # ZZ term
        for i in 1:N-1
            Z_term = Matrix(I, 2^(i-1), 2^(i-1))
            Z_term = kron(Z_term, Z)
            Z_term = kron(Z_term, Z)
            Z_term = kron(Z_term, Matrix(I,2^(N-i-1), 2^(N-i-1)))
            H_mat += J*Delta * Z_term
        end

        return H_mat
    end

    for N in 1:6
        for _ in 1:10
            J = rand()
            Delta = rand()

            H = XXZ(N, J, Delta);

            # exact matrix to compare to
            H_mat = XXZ_matrix(N, J, Delta);

            @test to_matrix(H) ≈ H_mat
        end
    end
end


@testitem "Test QuantumEast MPO" begin

    using LinearAlgebra

    function QuantumEast_matrix(N::Int, s::Real, c::Real)
        X = [0 1; 1 0]
        n = [0 0; 0 1]

        H_mat = zeros(ComplexF64, 2^N, 2^N)

        # nX term  
        for i in 1:N-1
            X_term = Matrix(I, 2^(i-1), 2^(i-1))
            X_term = kron(X_term, n)
            X_term = kron(X_term, X)
            X_term = kron(X_term, Matrix(I,2^(N-i-1), 2^(N-i-1)))
            H_mat += -exp(-s)*sqrt(c*(1-c)) * X_term
        end

        # nn term
        for i in 1:N-1
            nn_term = Matrix(I, 2^(i-1), 2^(i-1))
            nn_term = kron(nn_term, n)
            nn_term = kron(nn_term, n)
            nn_term = kron(nn_term, Matrix(I,2^(N-i-1), 2^(N-i-1)))
            H_mat += (1-2*c) * nn_term
        end

        # n term
        for i in 1:N
            n_term = Matrix(I, 2^(i-1), 2^(i-1))
            n_term = kron(n_term, n)
            n_term = kron(n_term, Matrix(I,2^(N-i), 2^(N-i)))
            H_mat += c * n_term
        end

        return H_mat
    end

    for N in 1:6
        for _ in 1:10
            s = rand()
            c = rand()

            H = QuantumEast(N, s, c);

            # exact matrix to compare to
            H_mat = QuantumEast_matrix(N, s, c);

            invert!(H) ### Note that the terms are backwards due to column major!
            @test to_matrix(H) ≈ H_mat
        end
    end
end



N = 20
J = rand()
g = rand()
h = rand()

psi = randomMPS(N, 2, 20, 0, 0.0)
normalise!(psi)
psi

energy_TFIM2(psi, J, g, h)

H = TFIM(N, J, g, h)

expectation(psi, H)

Threads.nthreads()

@benchmark energy_TFIM2(psi, J, g, h)

BLAS.set_num_threads(10)
@benchmark expectation(psi, H)
BLAS.get_num_threads()