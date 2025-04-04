A = [10, 36, 4, 5, -6]
m = A[0]
for i in range(1, len(A)):
    if A[i] < m:
        m = A[i]
print(m)