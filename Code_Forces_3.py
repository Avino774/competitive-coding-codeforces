#Om Namo Narayana
#Om shree ganapathaye namaha
''' Code Forces problemset Chapter 3 '''

'''1343b Balanced array '''
for n in [int(input()) for _ in ' '*int(input())]:
    if (n//2)%2 !=0:
        print('NO')
        continue
    even = [*range(2,2*n//2+1,2)]
    odd = [i-1 for i in even[:-1]] + [n//2-1 + even[-1]]
    print('YES')
    print(*(even+odd))

''' 1180a Alex and a rhombus ''' #I did not get it
n = int(input())
print( 2*n*(n-1) + 1 )

''' 1300a Non zero'''
for _ in range(int(input())):
    input()
    arr = [int(x) for x in input().split()]
    ch = 0
    for i,x in enumerate(arr):
        ch+=x==0
        arr[i]+=x==0
    print(ch+int(sum(arr)==0))

'''1453a Cancel the trains '''
i = lambda : [*map(int,input().split())]
for _ in range(int(input())):
    input()
    x = i()
    print(sum(1 for p in i() if p in x))

'''330a cakeminator'''
row,col = [*map(int,input().split())]
eat = [0,0]; mat = []
for _ in range(row):
    x = input()
    eat[0]+=int('S' not in x)
    mat.append(x)

for x in zip(*mat):
    eat[1]+=int('S' not in x)

print(eat[0]*col + eat[1]*row - eat[0]*eat[1])

'''119a epic game ''' #did not get it
from math import gcd
*a,n = map(int,input().split())
p=1
while n:
    p^=1
    n-=gcd(a[p],n)
print('01'[p])

''' 148a insomnia cure '''
i = lambda: int(input())
k = i()
l = i()
m = i()
n = i()
d = i()
esc = 0
for x in range(1,d+1):
    esc+=int(0 in [x%p for p in [k,l,m,n]])
print(esc)

''' 1108a 2 distinct points '''
for _ in range(int(input())):
    a,b,c,d = [int(x) for x in input().split()]
    print(a,[c,d][c==a])

###############################################################################
# RESTART CODEFORCES Aug 2022 and Continue
# Aim : TO become expert
# Do : A, B, C and learn all techniques
###############################################################################

#%%
''' 122A Lucky division '''
n = int(input())
possible = [4,7,47,74,444,447,477,474,744,774,777]
print("YES" if 0 in [n%p for p in possible] else "NO")

# %% 
''' 96A Football '''
s = input()
print('YNEOS'['0'*7 not in s and '1'*7 not in s :: 2])

# %% ''' 455A boredom ''' -- Very hard, very important. 
# How did they think of this?

input()
d = [0]*100001
for x in map(int,input().split()):
    d[x]+=x 
a=b=0 
for i in d:
    a,b = max(a,i+b),a 
print(a)

#%% 

for _ in '0'*int(input()):
    a,*b = map(int, input().split() )
    print(sum( x>a for x in b))

# %% ''' 405 A Gravity ''' 
input()
print(*sorted(map(int,input().split())))

# %% 1832A Palindrome 

for _ in '0'*int(input()):
    s = input()
    print('NO' if len(set(s[:len(s)//2])) == 1 else 'YES')

# %% 1832B Maximum sum 

for _ in '0'*int(input()):
    n,k = map(int, input().split())
    arr = sorted(map(int, input().split()))
    for _1 in range(k):
        if arr[0] + arr[1] > arr[-1]:
            arr = arr[-1]
        else:
            arr = arr[2:]
    print(sum(arr))

#%% 1832C Contrast value 

for _ in '0'*int(input()):
    n = int(input())
    a = [int(x) for x in input().split()]
    cnt = 0
    for x in a :
        cnt += a.count(x) == 1
    print(cnt)

# %% 