using Revise
using TestItems
using MatrixProductStates
using BenchmarkTools

@testitem "TFIM energy Z-basis" begin

    for N in 4:10
        psi = MPS(N)

        @test isapprox(energy_TFIM(psi, 1.0, 0.0), 0.0, atol=1e-15)
        @test isapprox(energy_TFIM(psi, 0.0, 1.0), N, atol=1e-15)
    end

end


@testitem "TFIM energy X-basis" begin

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

    for N in 4:10
        state_vector = rand(0:1, N)
        state_diff = diff(state_vector)
        state_diff = [1 - 2*abs(change) for change in state_diff]
        energy = sum(state_diff)

        psi_x = product_state_x(state_vector)

        @test isapprox(energy_TFIM(psi_x, 1.0, 0.0), energy, atol=1e-14)
    end

end