# Codeforces here, since the laptop is gone
# om namo narayana

# %% codecell
# 1459a Red blue shuffle

for _ in range(int(input())):
    n = int(input())
    r = [int(x) for x in input()]
    b = [int(x) for x in input()]
    count = sum( (red>blue) - (blue>red) for red,blue in zip(r,b) )
    print(['EQUAL','RED','BLUE', ][(count < 0) + (count!=0) ])


# %% codecell
# 1136a Natsya is reading a book
l = []; r = []
for _ in range(int(input())):
    a,b = map(int,input().split())
    l.append(a); r.append(b)
k = int(input()); flag=0
for a,b in zip(l,r):
    if k in range(a,b+1):
        print(len(l) - l.index(a))
        flag = 1
        break
if flag == 0:
    print(0)

# %% codecell
# Codeforces 1236a Stones

for _ in range(int(input())):
    a,b,c = map(int,input().split())
    # f,s = [[[a,b],[a,2*a]][a<=b//2],[[b,c],[b,2*b]][b<=c//2]][b<c or 0 in [a,b]]
    # print ( s - s%2 + s//2 + 3*(s%2 and a>=1 and c>=2))
    c = min(c,2*b)
    stone = 3*(c//2) * (b>0)
    b = b - c//2
    if b >= 2:
        stone+= min(a,b//2)*3
    print(stone)

    3*5//2

# %% codecell
# 1173a Nauuo and votes

x,y,z= map(int,input().split())
if min(x,y)+z < max(x,y) or z == 0:
    print(['0','-','+'][int(x>y) + int(x!=y)])
else:
    print('?')
