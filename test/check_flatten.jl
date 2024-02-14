using Revise
using TestItems
using MatrixProductStates

@testitem "Test Flatten" begin
    
    psi = MPS(4)
    psi_flat_exact = zeros(16)
    psi_flat_exact[1] = 1.0
    
    psi_flattened = flatten(psi)
    @test length(psi_flattened) == 16
    @test isapprox(psi_flattened[1], 1.0, atol=1e-14)
    @test isapprox(psi_flattened, psi_flat_exact, atol=1e-14)



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

    psi = product_state([0, 1, 0, 1])  # 5 in lexicographic, 10 in anti-lexicographic
    psi_flattened = flatten(psi)

    psi_flat_lexi = zeros(16)
    psi_flat_lexi[6] = 1.0  # 5 in lexicographic (not 1th element is zero!)
    @test !isapprox(psi_flattened, psi_flat_lexi, atol=1e-14)  # lexicographic ordering

    psi_flat_antilexi = zeros(16)
    psi_flat_antilexi[11] = 1.0
    @test isapprox(psi_flattened, psi_flat_antilexi, atol=1e-14)  # anti-lexicographic ordering

end