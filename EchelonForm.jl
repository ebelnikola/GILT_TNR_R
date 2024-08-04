using LinearAlgebra


function flip_rows!(A, rows)
    for i in range(1, size(A)[2])
        tmp = A[rows[1], i]
        A[rows[1], i] = A[rows[2], i]
        A[rows[2], i] = tmp
    end
    return A
end;

function flip_columns!(A, cols)
    tmp_A = deepcopy(A[:, cols[1]])
    A[:, cols[1]] .= A[:, cols[2]]
    A[:, cols[2]] .= tmp_A
    return A
end;

function reduce_rows_below!(A, row, col)
    factors = A[(row+1):size(A)[1], col] ./ A[row, col]
    for j in range(1, size(A)[2])
        for i in range(row + 1, size(A)[1])
            A[i, j] -= factors[i-row] * A[row, j]
        end
    end
    return A
end



function reduce_rows_below!(A::Matrix{Bool}, row, col)
    for i = (row+1):size(A, 1)
        if A[i, col]
            A[i, :] .= xor.((A[i, :]), (A[row, :]))
        end
    end
end


function echelon_form!(A)
    z = zero(eltype(A))
    hight = size(A, 1)
    rank = 0
    for i in range(1, hight)
        y = findfirst(x -> x != z, A[i:end, i:end])
        if isnothing(y)
            rank = i - 1
            break
        end
        rank = i
        flip_rows!(A, [i, y[1] + i - 1])
        reduce_rows_below!(A, i, y[2] + i - 1)
    end
    return A, rank
end
