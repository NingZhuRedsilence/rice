def compute_bi_coef_improved(n, k):
    """compute binomial coefficients using Pascal Identity"""
    # initialize 2 lists to store values for dynamic computing
    row_i_minus_1 = []
    row_i = []

    # validate input
    if n < k:
        # print "Invalid input: n must be greater or euqal to k!"
        return row_i

    # fill base case n = 1, k = 0 and k = 1
    row_i_minus_1 = [1, 1]

    # recursive case
    for i in range(2, n + 1):
        for j in range(0, k + 1):
            if j == 0:
                row_i.append(1)
                # print "row_i in base caes: ", row_i

            elif i == j:
                del row_i_minus_1[:]
                row_i.append(1)
                # watch out python reference
                for elem in row_i:
                    row_i_minus_1.append(elem)  # todo ask AT: the right way to move values between objects in Python?
                # print row_i_minus_1, row_i
                # what is this situation? need to clean the temporary storage if I'm using append
                del row_i[:]
                # print "row_i after deletion ", row_i
                break

            else:
                # print "i, j is ", i, j
                # print "row_i_minus_1 in recur", row_i_minus_1
                row_i.append(row_i_minus_1[j-1] + row_i_minus_1[j])
                # print "row_i in recur", row_i

            j += 1
        if i == n:
            return row_i[k]
        i += 1

def test_computer_bi_coef(n, k):

    return compute_bi_coef_improved(n, k);
# end of function

print "n = 4, k = 3", test_computer_bi_coef(4, 3)
print "n = 7, k = 5", test_computer_bi_coef(7, 5)
print "n = 8, k = 3", test_computer_bi_coef(8, 3)

