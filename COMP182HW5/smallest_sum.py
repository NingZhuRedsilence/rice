

def smallest_descent_sum(matrix_triangle):
    # initialize an empty matrix to hold sums at each number, a list of lists
    sums = []
    # fill boundary cases j =0, sums[i, 0] = inf
    size = len(matrix_triangle)
    for i in range(0, (size)):
        sums.append([float("inf")])

    # base case sums[0, 1] = input[0,0]
    sums[0].append(matrix_triangle[0][0])
    sums[0].append(float("inf"))
    sums[0].append(float("inf"))

    # recursive case
    # todo: "better" (less error-prone) way to fill up the matrix?
    for i in range(1, size):
        for j in range(1, i + size):
            if j >= (i + 2):
                sums[i].append(float("inf"))
            else:
                print "i, j ", i, j
                print sums
                print sums[i-1][j-1], sums[i-1][j], sums[i-1][j+1]
                sums[i].append(min(sums[i-1][j-1], sums[i-1][j], sums[i-1][j+1]) + matrix_triangle[i][j-1])
            j += 1

        i += 1
    print sums
    return min(sums[size-1])
# end of function


def test_smallest_descent_sum(test_matrix):

    return smallest_descent_sum(test_matrix);
# end of function

example = [[2],[5, 4], [1, 4, 7], [8, 6, 9, 6]]
print test_smallest_descent_sum(example)
