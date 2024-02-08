using Revise
using TestItems
using MatrixProductStates


@testitem "Product Test" begin

    X = [0 1; 1 0];
    Y = [0 -im; im 0];
    Z = [1 0; 0 -1];

    """
    Create a product mps from a computational basis vector of the form |01101001...>.
    """
    function product_state(state::Vector{Int})::MPS
        d = 2
        N = length(state)
        tensors = [zeros(1, 2, 1) for _ in 1:N]
        centre = 1
        for i in 1:N
            if state[i] > 1 || state[i] < 0
                throw(ArgumentError("Invalid state: $state"))
            end
            tensors[i][1, state[i]+1, 1] = 1
        end
        return MPS(d, N, tensors, centre, 0, 0.0)
    end


    psi = product_state([0, 1, 1, 0, 1, 0, 0, 1, 0, 1])

    @test expectation_1site(psi, Z, 1) ≈ 1.0 + 0.0im
    @test expectation_1site(psi, Z, 5) ≈ -1.0 + 0.0im
    @test expectation_1site(psi, Z, 10) ≈ -1.0 + 0.0im
    @test expectation_1site(psi, X, 10) ≈ 0.0 + 0.0im
    @test expectation_1site(psi, Y, 6) ≈ 0.0 + 0.0im
end


@testitem "Superposition Test" begin

    X = [0 1; 1 0];
    Y = [0 -im; im 0];
    Z = [1 0; 0 -1];

    """
    Create a product mps from a computational basis vector of the form |01101001...>.
    """
    function product_state_x(state::Vector{Int})::MPS
        d = 2
        N = length(state)
        tensors = [zeros(1, 2, 1) for _ in 1:N]
        centre = 1
        for i in 1:N
            if state[i] > 1 || state[i] < 0
                throw(ArgumentError("Invalid state: $state"))
            end
            tensors[i][1, 1, 1] = 1
            tensors[i][1, 2, 1] = 1*(-1)^(state[i])
        end

        psi = MPS(d, N, tensors, centre, 0, 0.0)
        normalise!(psi)
        return psi
    end


    psi = product_state_x([0, 1, 1, 0, 1, 0, 0, 1, 0, 1])

    @test isapprox(expectation_1site(psi, X, 1), 1.0 + 0.0im; atol=1e-15)
    @test isapprox(expectation_1site(psi, X, 5), -1.0 + 0.0im; atol=1e-15)
    @test isapprox(expectation_1site(psi, X, 10), -1.0 + 0.0im; atol=1e-15)
    @test isapprox(expectation_1site(psi, Z, 10), 0.0 + 0.0im; atol=1e-15)
    @test isapprox(expectation_1site(psi, Y, 6), 0.0 + 0.0im; atol=1e-15)
end


@testitem "two qubit test" begin

    X = [0 1; 1 0];
    Y = [0 -im; im 0];
    Z = [1 0; 0 -1];
    XX = kron(X, X);
    ZZ = kron(Z, Z);

    """
    Create a product mps from a computational basis vector of the form |01101001...>.
    """
    function product_state_x(state::Vector{Int})::MPS
        d = 2
        N = length(state)
        tensors = [zeros(1, 2, 1) for _ in 1:N]
        centre = 1
        for i in 1:N
            if state[i] > 1 || state[i] < 0
                throw(ArgumentError("Invalid state: $state"))
            end
            tensors[i][1, 1, 1] = 1
            tensors[i][1, 2, 1] = 1*(-1)^(state[i])
        end

        psi = MPS(d, N, tensors, centre, 0, 0.0)
        normalise!(psi)
        return psi
    end

    psi = product_state_x([0, 1, 1, 0, 1, 0, 0, 1, 0, 1])

    @test isapprox(expectation_2site(psi, XX, 1), -1.0 + 0.0im; atol=1e-15)
    @test isapprox(expectation_2site(psi, XX, 2), 1.0 + 0.0im; atol=1e-15)
    @test isapprox(expectation_2site(psi, XX, 9), -1.0 + 0.0im; atol=1e-15)
    @test isapprox(expectation_2site(psi, ZZ, 9), 0.0 + 0.0im; atol=1e-15)
    @test isapprox(expectation_2site(psi, ZZ, 6), 0.0 + 0.0im; atol=1e-15)

end

# Add tests to check I'm not messing up transposes anywhere! Soemthing with Y