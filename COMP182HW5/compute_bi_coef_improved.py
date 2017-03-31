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
        for j in range(0, k + 2):
            if j == 0:
                row_i.append(1)

            elif i == j:
                row_i.append(1)
                break

            else:
                a = row_i_minus_1[j-1]
                b = row_i_minus_1[j]
                row_i.append(a + b)

                if j == k + 1:
                    break

            j += 1

        del row_i_minus_1[:]
        for elem in row_i:
            row_i_minus_1.append(elem)  # todo ask AT: the right way to move values between objects in Python?
        # what is this situation? need to clean the temporary storage if I'm using append
        del row_i[:]

        if i == n:
            return row_i_minus_1[k]
        i += 1
    # return row_i[k]
# end of function

def test_computer_bi_coef(n, k):

    return compute_bi_coef_improved(n, k);
# n = 4, k = 3 4
# n = 7, k = 5 21
# n = 8, k = 3 56
# end of function

print "n = 4, k = 3", test_computer_bi_coef(4, 3)
print "n = 7, k = 5", test_computer_bi_coef(7, 5)
print "n = 8, k = 3", test_computer_bi_coef(8, 3)

