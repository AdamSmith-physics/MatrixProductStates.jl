export normalise!

function normalise!(mps::MPS)
    # start at left end
    movecentre!(mps, 1)
    movecentre!(mps, length(mps); normalised=true)
    movecentre!(mps, 1; normalised=true)  # maybe unneccessary?
end