def fibonacci(n):
    """
    Returns the nth Fibonacci number.

    Arguments:
    n -- desired Fibonacci number, must be >= 0

    Returns:
    nth Fibonacci number
    """
    # base case
    if n <= 1:
        return n

    # inductive case: f(n) = f(n-1) + f(n-2)
    return fibonacci(n - 1) + fibonacci(n - 2)


# test
print fibonacci(10)
print fibonacci(20)