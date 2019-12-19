import numpy as np

def find_med_backtrace(str1, str2, cutoff=-1):
    '''
    This function calculates the Minimum Edit Distance between 2 words using
    Dynamic Programming, and asserts the optimal transition path using backtracing.
    Input parameters: original word, target word
    Output: minimum edit distance, path
    Example: ('password', 'Passw0rd') -> 2.0, [('s', 'P', 0), ('s', '0', 5)]
    '''
    # op_arr_str = ["d", "i", "c", "s"]

    # Definitions:
    n = len(str1)
    m = len(str2)
    D = np.full((n + 1, m + 1), np.inf)
    trace = np.full((n + 1, m + 1), None)
    trace[1:, 0] = list(zip(range(n), np.zeros(n, dtype=int)))
    trace[0, 1:] = list(zip(np.zeros(m, dtype=int), range(m)))
    # Initialization:
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)

    # Fill the matrices:
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            delete = D[i - 1, j] + 1
            insert = D[i, j - 1] + 1
            if (str1[i - 1] == str2[j - 1]):
                sub = np.inf
                copy = D[i - 1, j - 1]
            else:
                sub = D[i - 1, j - 1] + 1
                copy = np.inf
            op_arr = [delete, insert, copy, sub]
            D[i, j] = np.min(op_arr)
            op = np.argmin(op_arr)
            if (op == 0):
                # delete, go down
                trace[i, j] = (i - 1, j)
            elif (op == 1):
                # insert, go left
                trace[i, j] = (i, j - 1)
            else:
                # copy or subsitute, go diag
                trace[i, j] = (i - 1, j - 1)
    # print(trace)
    # Find the path of transitions:
    i = n
    j = m
    cursor = trace[i, j]
    path = []
    while (cursor is not None):
        # 3 possible directions:
        #         print(cursor)
        if (cursor[0] == i - 1 and cursor[1] == j - 1):
            # diagonal - sub or copy
            if (str1[cursor[0]] != str2[cursor[1]]):
                # substitute
                path.append(("s", str2[cursor[1]], cursor[0]))
            i = i - 1
            j = j - 1
        elif (cursor[0] == i and cursor[1] == j - 1):
            # go left - insert
            path.append(("i", str2[cursor[1]], cursor[0]))
            j = j - 1
        else:
            # (cursor[0] == i - 1 and cursor[1] == j )
            # go down - delete
            path.append(("d", None, cursor[0]))
            i = i - 1
        cursor = trace[cursor[0], cursor[1]]
        # print(len(path), cursor)
    md = D[n, m]
    del D, trace
    return md, list(reversed(path))


if __name__ == "__main__":
    print(find_med_backtrace('a123456', 'a654321'))