# lower triangle
n=int(input('enter n:'))
print("lower triangle:")
for i in range(1,n+1):
    print('*' * i)


# upper triangle
n=int(input('enter n:'))
print("upper triangle:")
for i in range(n,0,-1):
    print('*' * i)

# pyramid pattern
n=int(input('enter n'))
print("pyramid:")
for i in range(n):
    for j in range(n-i-1):
        print(" ",end="")
    for k in range(i+1):
        print('*',end=" ")
    print()