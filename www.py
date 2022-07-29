import numpy as np
from itertools import permutations,combinations

def valid(A):
        bal = 0
        for c in A:
            if c == '(': bal += 1
            else: bal -= 1
            if bal < 0: return False
        return bal == 0


def generateParenthesis(n: int):
    s=""
    for i in range(n):
        s+="()"
    all=set(permutations(s))
    result=[]
    print(len(all))
    for exam in all:
        if valid(exam):
           result.append(("".join(exam)))
    # print(len(all))
    print(result)
    return result
generateParenthesis(6)
