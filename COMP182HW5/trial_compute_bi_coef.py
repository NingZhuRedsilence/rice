def compute_bi_coef(n, k):
    # check input validity
    if n <= k:
        print "Invalid input: n must be greater or equal to k!"
        return;

    # initialize a dp table c of size (k + 1) x (n + 1) as a list of lists
    c = []
    for i in range(0, k + 1):
        c.append([])

    # fill base cases where k = 0, c[k][j] = 1, i is in [0, n], i.e. the first row
    for j in range(0, n + 1):
        c[0].append(1)
        j += 1

    # calculate dp cells c[i][j] = c[i-1][j-1] + c[i][j-1], i is in [1, k], j is in [0, n]
    for i in range(1, k + 1):
        for j in range(0, n + 1):
            if i > j:
                c[i].append(float("-inf"))
            else:
                if i == j:
                    c[i].append(1)
                else:
                    c[i].append(c[i-1][j-1] + c[i][j-1])
            j += 1

        i += 1
    return c[k][n];
# end of function

def test_computer_bi_coef(n, k):

    return compute_bi_coef(n, k);

# n = 4, k = 3 4
# n = 7, k = 5 21
# n = 8, k = 3 56

# end of function

print "n = 4, k = 3", test_computer_bi_coef(4, 3)
print "n = 7, k = 5", test_computer_bi_coef(7, 5)
print "n = 8, k = 3", test_computer_bi_coef(8, 3)