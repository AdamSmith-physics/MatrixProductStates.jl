using Revise
using TestItems
using MatrixProductStates

@testitem "Test apply 1-site" begin
    
    X = [0 1; 1 0]
    Y = [0 -1im; 1im 0]

    psi = MPS(5)

    flatten(psi)

    apply_1site!(psi, X, 1)  # |10000>
    check_state = zeros(32)
    check_state[2] = 1.0
    @test isapprox(flatten(psi), check_state, atol=1e-14)

    apply_1site!(psi, Y, 3)  # i|10100>
    check_state = 1im*zeros(32)
    check_state[6] = 1.0im
    @test isapprox(flatten(psi), check_state, atol=1e-14)

    apply_1site!(psi, Y, 1)  # |00100>
    check_state = 1im*zeros(32)
    check_state[5] = 1.0
    @test isapprox(flatten(psi), check_state, atol=1e-14)

end


@testitem "Test apply 2-site" begin
    
    H = [1 1; 1 -1]/sqrt(2)
    CNOT = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]

    psi = MPS(5)

    apply_1site!(psi, H, 2)  # (|00000> + |01000>) / sqrt(2)

    apply_2site!(psi, CNOT, 2)  # (|00000> + |01100>) / sqrt(2)

    check_state = zeros(32)
    check_state[1] = 1.0/sqrt(2)
    check_state[7] = 1.0/sqrt(2)
    @test isapprox(flatten(psi), check_state, atol=1e-14)

    apply_2site!(psi, CNOT, 3)  # (|00000> + |01110>) / sqrt(2)

    check_state = zeros(32)
    check_state[1] = 1.0/sqrt(2)
    check_state[15] = 1.0/sqrt(2)
    @test isapprox(flatten(psi), check_state, atol=1e-14)

end