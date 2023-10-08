
###########################
''' Codeforces trials and codes
Part 2 '''
###########################

''' Phoenix and Coins cf 1348a'''
for _ in range(int(input())):
    n = int(input())
    a = sum(2**x for x in range(n//2,n))
    b = sum(2**x for x in range(1,n//2)) + 2**n
    print(b-a)

''' Bachgold problem cf 794a'''
n = int(input())
max_primes = [[2]*(n//2),[2]*(n//2 - 1) + [3]][n%2]
print(len(max_primes)); print(*max_primes)

''' Most unstable array 1353a'''
for n,m in [[int(x) for x in input().split()] for _ in ' '*int(input())]:
    print([0,((n>2)+1)*m][n!=1])

''' Find the gcd cf 1370a'''
for n in [int(input()) for _ in range(int(input()))]:#heavy program
    a,b = n-1,n
    gcd = -1
    for b in range(n,1,-1):
        for a in range(b-1,0,-1):
            gcd = [max(gcd,a),gcd][b%a>0]
    print(gcd)

for n in [int(input()) for _ in range(int(input()))]:
    print(n//2)

''' Restore the permutations by merger cf 1385b'''
for _,arr in [[input(),[int(x) for x in input().split()]] for _ in ' '*int(input())]:
    x = []
    for c in arr:
        if c not in x:
            x.append(c)
    print(*x)

''' Magical Sticks cf 1371a'''
for n in [int(input()) for _ in range(int(input()))]:
    print(n//2 + [0,1][n%2])

''' Boring apartment cf 1433a'''
for n in [input() for _ in range(int(input()))]:
    print(10*(int(n[0])-1) + len(n)*(len(n)+1)//2)

''' Maximum in table cf 509a '''
n = int(input())
a = [[1]*n]*n
for i in range(1,n):
    for j in range(1,n):
        a[i][j] = a[i][j-1] + a[i-1][j]
print(a[n-1][n-1])

''' add odd or subtract even cf '''
for a,b in [[int(x) for x in input().split()] for _ in range(int(input()))]:
    print([0,[2,1][(a<b and (b-a)%2) or (a>b and (a-b)%2==0)]][a!=b])

''' Brain's photos '''
colors = 'CMY'
r,c = [int(x) for x in input().split()]
cols = ''
for i in range(r):
    cols += input()
print('#'+['Black&White','Color'][sum(x in cols for x in colors)>0])

''' Three pairwise Maximums cf 1385a'''
for x,y,z in [[int(x) for x in input().split()] for _ in range(int(input()))]:
    x,y,z = sorted([x,y,z])[::-1]
    if x == y:
        print('YES')
        print(x,z,z)
    else:
        print('NO')

'''Collecting Coins 1294a'''
for _ in range(int(input())):
    a,b,c,n = [int(x) for x in input().split()]
    a,b,c = sorted([a,b,c])
    n = n - (2*c-a-b)
    print('YNEOS'[(n<0 or n%3!=0)::2])

''' Fashionablee 1369a'''
for _ in range(int(input())):
    print('YNEOS'[int(input())%4!=0::2])

''' Black Square 431a '''
calorie = {i:x for i,x in enumerate([int(x) for x in input().split()],1)}
print(sum(calorie[int(i)] for i in input()))

''' Gennady and a card game 1097a'''
table = input()
hand = input().split(' '); flag = 0
for i in [0,1]:
    if table[i] in list(zip(*hand))[i]:
        print('YES'); flag = 1
        break
if flag ==0:
    print('NO')

''' Vanya and cubes '''
n = int(input()); count = 1
while sum(i*(i+1)//2 for i in range(1,count+1))<=n:
    count +=1
print(count-1)

''' special permutations 1454a'''
for _ in range(int(input())):
    p = list(range(1,int(input())+1))
    print(*(p[1:] + [p[0]]))

''' sereja nad dima 381a '''
input()
cards = [int(x) for x in input().split()]
i=1; sereja = 0; dima = 0
while cards!=[]:
    if i%2:
        sereja+=max(cards[0],cards[-1])
    else:
        dima+=max(cards[0],cards[-1])
    del cards[cards.index(max(cards[0],cards[-1]))]
    i+=1
print(sereja,dima)

''' cards for friends 1472a '''
for w,h,n in [[int(x) for x in input().split()] for _ in ' '*int(input())]:
    w,h,cnt = w*h,1,0
    while w%2==0:
        w//=2
        cnt+=1
    print('YNEOS'[2**cnt<n::2])

for s in[*open(0)][1:]:
    w,h,n=map(int,s.split())
    print('YNEOS'[w*h&-w*h<n::2])

''' Fafa and his employees '''
n = int(input())
cnt = 0
for f in range(1,n//2+1):
    if (n-f)%f == 0:
        cnt+=1
print(cnt)

n = int(input())
print(sum((n-r)%r==0 for r in range(1,n//2+1)))

''' Ichihime and the triangle '''
tri = lambda a,b,c : a+b>c and a+c>b and b+c>a
for val in [[int(x) for x in input().split()] for _ in range(int(input()))]:
    val[1] = val[2]
    while not tri(*val[0:3]):
        val[1] = val[1]-1
    print(*val[0:3])

#spent too long on this question

''' Soft drinking '''
#lots of variables
n,k,l,c,d,p,nl,np = [int(x) for x in input().split()]
print(min(k*l//nl,c*d,p//np)//n)

''' array with odd sum 1296a'''
for n,arr in [[input(),[int(x) for x in input().split()]] for _ in ' '*int(input())]:
    condition = [len(arr)%2==0 and sum(x%2 for x in arr)%2==len(arr),sum(x%2 for x in arr)==0]
    print('YNEOS'[True in condition::2])

''' Omkar and completion 1372a'''
for n in [int(input()) for _ in ' '*int(input())]:
    print(*[i*(i+1)//2 + 1 for i in range(n+1)])

''' Mahmoud, Ehab and the even odd game 959a'''
print(['Mahmoud','Ehab'][int(input())%2])
#this is a fluke. How did this work?

''' Minutes before the new year 1283a '''
for _ in ' '*int(input()):
    h,m = [int(x) for x in input().split()]
    print(1440 - (h*60+m))

''' Again 25 630a Noicee!!'''
print(25)

''' CopyCopyCopyCopyCopy 1325b '''
for _ in ' '*int(input()):
    n = int(input())
    print(len(set([int(x) for x in input().split()])))

''' Ehab and Gcd 1325a'''
for x in [int(input()) for _ in ' '*int(input())]:
    print(*[[x//2,x//2],[1,x-1]][x%2 and x>1])

''' LCM problem 1389a'''
def factors(num):
    f = []
    for i in range(1,num+1):
        if num%i == 0:
            f.append(i)
    return f

for _ in range(int(input())):
    l,r = list(map(int,input().split()))
    f = list(set([i for i in factors(r - [0,1][r%2]) if i >=l]))
    if f == [] or len(f)==1:
        print(-1,-1)
    else:
        print(f[0],f[-1])


###

for s in[*open(0)][1:]:
    l,r=map(int,s.split())
    print(*([l,2*l],[-1]*2)[r<2*l])

''' Vus the cussack and the contest 1186a '''
n,m,k = list(map(int,input().split()))
print('YNEOS'[n>min(m,k)::2])

########### PAGE 2 CF ###########

''' Fair division 1472b '''
for _ in range(int(input())):
    n = int(input()); flag = 0
    arr = [int(x) for x in input().split()]
    if sum(arr)%2:
        flag = 0
    else:
        if (sum(arr)//2)%2:
            if arr.count(1)>0 and arr.count(1)%2==0:
                flag = 1
        else:
            flag = 1
    if flag ==1:
        print('YES')
    else:
        print('NO')


for s in[*open(0)][2::2]:
    a=s.count('1')
    print('YNEOS'[a%2 or a<len(s)%4::2])

''' Night at the museum 713a '''
name = input()
pos = 'a'; turn = 0
for i in range(len(name)):
    turn+=min(abs(abs(ord(name[i]) - ord(pos))-26), abs(ord(name[i]) - ord(pos)))
    pos = name[i]
print(turn)

''' combination lock 540a '''
input()
f = lambda : [int(x) for x in input()]
org = f(); key = f(); turn = 0
for o,k in zip(org,key):
    turn+=min(abs(o-k),abs(10-abs(o-k)))
print(turn)

''' Floor number 1426a '''
import math
for n,x in [[int(x) for x in input().split()] for _ in ' '*int(input())]:
    f = 1
    while 2+(f-1)*x < n:
        f+=1
    print(f)
for s in [*open(0)][1:]:a,b=map(int,s.split());print((a-3+b)//b+1 if a>2 else 1)

''' favourite sequence 1462a '''
for _ in range(int(input())):
    n = int(input()); final = []
    arr = [int(x) for x in input().split()]

for _ in range(int(input())):
	n = int(input())
	a = list(map(int, input().split()))
	for i in range(n//2):
		print(a[i],a[-1-i],end=" ")
	if(n%2):
		print(a[n//2],end="")
	print()

#the worst program I'd ever written

''' Die roll 9a'''
m = max(int(x) for x in input().split())
s = sum(x>=m for x in range(1,7))
y = [[1,6],[1,3],[1,2],[2,3],[5,6],[1,1]][s-1]
print('/'.join(str(x) for x in y))

''' Bus to udayland 711a '''
seats = []; flag = 0
for _ in range(int(input())):
    s = input()
    if 'OO' in s and flag == 0:
        s = s.replace('OO','++',1)
        flag = 1
    seats.append(s)

if flag == 0:
    print('NO')
else:
    print('YES')
    print('\n'.join(seats))

''' Replacing elements 1473a '''
for _ in range(int(input())):
    n,d = [int(x) for x in input().split()]
    arr = sorted([int(x) for x in input().split()])
    print('NYOE S'[sum(arr[:2])<=d or sum(i>d for i in arr)==0::2])

''' C+= 1368a '''
for a,b,n in [[int(x) for x in input().split()] for _ in ' '*int(input())]:
    moves= 0
    while a<=n and b<=n:
        a,b = sorted([a,b])
        a+,moves+=b,1
    print(moves)

''' Common subsequence 1382a '''
f = lambda : list(map(int,input().split()))
for _ in range(int(input())):
    n,m = f()
    a = f(); b = f()
    s = set(a).intersection(set(b))
    if len(s) == 0:
        print('NO')
    else:
        print('YES')
        print(1,list(s)[0])

''' Similar strings 1400a '''
for _ in range(int(input())):
    n = int(input())
    s = list(input()); w= []
    for i in range(n):
        w.append(s[i:n+i][i])
    print(''.join(w))

for s in[*open(0)][2::2]:print(s[::2])

'''Juggling letters 1397a '''
import string
for _ in range(int(input())):
    n = int(input())
    d = {i:0 for i in string.ascii_lowercase}
    for arr in [input() for _ in ' '*n]:
        for elem in arr:
            d[elem]+=1
    print('YNEOS'[sum(i%n for i in d.values())>0::2])

''' Maximum increase 702a'''
input(); maxlen = 0; last = 0; s = 0
for elem in [int(x) for x in input().split()]:
    s = s+1 if elem>last else 1
    last=elem
    maxlen = max(maxlen,s)
print(maxlen)

input();v=s=r=0
for x in map(int,input().split()):
    s=s+1 if x>v else 1;v=x;r=max(r,s)
print(r)

#####################################################
#I have forgotten much of what I was supposed to know by now from HR
####################################################
''' Bad triagle 1398a '''
bad_tri = lambda a,b,c: a+b<c or a+c<b and b+c<a
for _ in range(int(input())):
    n = int(input()); flag = 0
    arr = [int(x) for x in input().split()]
    for i in range(n-2):
        for j in range(i+1,n-1):
            for k in arr[j:]:
                if bad_tri(arr[i],arr[j],k):
                    req = [i+1,j+1,arr.index(k)+1]
                    flag = 1
                    break
    if flag == 0:
        print(-1)
    else:
        print(*req)

for _ in range(int(input())):
	input()
	a = list(map(int,input().split()))
	if a[0]+a[1]<=a[-1]:
        print(1,2,len(a))
	else:
        print(-1)

for _ in range(int(input())):
    input()
    arr = [int(x) for x in input().split()]
    if arr[0]+arr[1]>arr[-1]: # no non degenrate tri possible
        print(-1)
    else:
        print(1,2,len(arr))

''' 119a epic game '''
a,b,n = list(map(int,input().split()))
def gcd(a,b):
    g = 1
    for i in range(a,0,-1):
        if b%i == 0:
            g = i
    return g

move = 1
while n>0:
    n-=[gcd(min(a,n),max(a,n)),gcd(min(b,n),max(b,n))][move%2==0]
    move+=1
    print(move,n,a,b)
print([0,1][move%2])

# Did not get test 4, dont know what is wrong

''' 1196a 3 piles of candies '''
for arr in [[int(x) for x in input().split()] for _ in ' '*int(input())]:
    print(sum(arr)//2)

''' 1285a Menzo playing zoma '''
input()
s = input()
print(s.count('L')+s.count('R')+1)

'''1433B Yet another bookshelf '''
for _ in range(int(input())):
    n = int(input())
    arr = [int(x) for x in input().split(' ')]
    indices = []
    for i,x in enumerate(arr,1):
        if x == 1 and i+1 not in indices:
            indices.append(i)
    print(sum(j - (i+1) for i,j in zip(indices[:-1],indices[1:]) if j-i>=2))


'''1388a Captain flint and crew recruitment '''
## SUCCESS!!!! ###
primes = [2,3,5,7,11,13,17,19,23,29,31]
nearest = sorted([i*j for i in primes for j in primes[primes.index(i)+1:]])
for _ in range(int(input())):
    num = int(input())
    one = nearest[0]
    two = nearest[1]
    three = [i for i in nearest[2:] if i<num-one-two]
    if three == []:
        print('NO')
        continue
    diff = 6; i = 0
    while diff in nearest:
        diff = num-(one+two+three[i])
        i+=1
    print('YES')
    print(one,two,three[i-1],diff)

''' 1234a equalize prizes again '''
for _ in range(int(input())):
    n = int(input())
    prices = list(map(int,input().split(' ')))
    print(sum(prices)//n + [0,1][sum(prices)%n>0])

''' 599a Patrick and shopping '''
d1,d2,d3 = [int(x) for x in input().split()]
print(min(2*(d1+d2),d1+d2+d3,2*(d2+d3),2*(d1+d3)))

'''734b Anton and digits '''
arr = list(map(int,input().split()))
bigs = min(arr[i] for i in [0,2,3])
smalls = min(arr[0]-bigs,arr[1])
print(bigs*256+smalls*32)

''' 1391a suborrays '''
for _ in range(int(input())):
    print(*range(int(input()),0,-1))

''' 978b file name '''
input();s = input(); cnt = 0
for i in range(len(s)):
    if s[i:i+3] == 'xxx':
        cnt+=1
print(cnt)

'''1354a puzzle pieces '''
for n,m in [[int(x) for x in input().split()] for _ in range(int(input()))]:
    if True in [m==2 and n==2,min(m,n)==1]:
        print('YES')
    else:
        print('NO')

''' 32b borze '''
i=0; s = input()+'A'; num = []
while s[i:]!='A':
    if s[i:i+2] == '-.':
        num.append(1)
        i+=2
    elif s[i:i+2] == '--':
        num.append(2)
        i+=2
    else:
        num.append(0)
        i+=1
print(''.join(str(x) for x in num))

R=str.replace
print(R(R(R(input(),'--','2'),'-.','1'),'.','0'))

# 13 March 2021
'''822a I'm bored with life '''
def fact(num):
    if num == 1 or num == 0:
        return 1
    else:
        return fact(num-1)*num

print(fact(min(map(int,input().split()))))

'''1220a Cards '''
input()
s = input()
print(' '.join(['1']*(s.count('o') - s.count('z')) + ['0']*s.count('z')))

''' 1454b unique bid auction '''
for n,arr in [ [int(input()),[int(x) for x in input().split()]] for _ in ' '*int(input())]:
    z = [0]*(n+1)
    for elem in arr:
        z[elem]+=1
    print(arr.index(z.index(1))+1) if 1 in z else print(-1)

### fastest
for s in[*open(0)][2::2]:
 d={};i=0
 for x in s.split():i+=1;d[x]=i-i*(x in d)
 print(min([(int(x),d[x])for x in d if d[x]]+[(3e5,-1)])[1])

for s in[*open(0)][2::2]:
 d={};i=0
 for x in map(int,s.split()):i+=1;d[x]=(i,-1)[d.get(x,0)!=0]
 a=[(x,d[x])for x in d if d[x]>0];print(a and min(a)[1]or-1)

for _ in range(int(input())):
    n = int(input())
    lst = [0] * n
    a = [int(i) for i in input().split()]
    for i in a:
        lst[i-1] += 1
    print(a.index(lst.index(1)+1)+1) if 1 in lst else print(-1)

from collections import Counter
for _ in range(int(input())):
    input()
    k = list(map(int,input().split()))
    a = sorted({i:v for i,v in Counter(k).items() if v==1})
    print(k.index(a[0])+1 if len(a)>0 else -1)

''' 80a paranomix' prediction '''
n,m = [*map(int,input().split())]
flag = 0
for p in range(n+1,m+1):
    if 0 not in [p%i for i in range(2,p//2+1)] and flag == 0:
        print('YNEOS'[p!=m::2])
        flag = 1
if flag == 0:
    print('NO')

l=[2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,51]
n,m=map(int,input().split())
print(l[l.index(n)+1]==m and "YES" or "NO")

f=lambda x:all(x%i for i in range(2,x))
n,m=map(int,input().split())
print(['NO','YES'][f(m)and all(not f(i)for i in range(n+1,m))])

''' 1398b substring removal game '''
for s in[*open(0)][1:]:
    print(sum(sorted(map(len,s[:-1].split('0')))[-1::-2]))

''' 894a QAQ '''
cnt = 0
s = [x for x in input() if x in 'QAQ']
for i in range(len(s)-1):
    if s[i] == 'Q':
        for j in range(i+1,len(s)):
            if s[j] == 'A':
                cnt+= s[j+1:].count('Q')
print(cnt)

''' 1478a Nezzar and colorful balls '''
for s in [*open(0)][2::2]:
    arr = s.split()
    print(max([arr.count(i) for i in arr]))

''' 1501a Alexey and trains ''' #Did not get
import math
for _ in range(int(input())):
    a = [0]; b = [0]; depart = 0
    for _1_ in ' '*int(input()):
        _1,_2 = [*map(int,input().split())]
        a = a + [_1]; b = b+ [_2]
    tm = [0]+[int(x) for x in input().split()]
    for i in range(1,len(tm)):
        temp = a[i]
        a[i] = a[i] + tm[i] + depart - b[i-1]
        depart = max(a[i] + math.ceil((b[i]-a[i])/2),b[i])
    print(a[-1])

R=lambda:map(int,input().split())
t,=R()
while t:
 t-=1;n,=R();a,b=zip(*(R()for _ in[0]*n));s=0
 for x,y,z,w in zip(a,b,(0,*b),R()):p=s+x-z+w;s=p+max(y-x+1>>1,y-p)
 print(p)

'''1478b Nezzar and lucky numbers ''' #time limit exceeded

def solve(x,y,num):
    for a in range(num//x+1):
         for b in range(num//y+1):
             if a*x + b*y == num:
                 return True
    return False

for _ in range(int(input())):
    q,d = [*map(int,input().split())]
    for num in [*map(int,input().split())]:
        lucky = [i for i in range(1,num+1) if str(d) in str(i)][::-1]
        if True in [solve(x,y,num) for x in lucky[:-1] for y in lucky[lucky.index(x)+1:]]:
            print('YES')
        else:
            print('NO')

''' 1312a 2 regular polygons '''
for _ in range(int(input())):
    m,n = [int(x) for x in input().split()]
    print('YNEOS'[m%n>0::2])

''' 1077a Frog Jumping '''
for _ in range(int(input())):
    a,b,k = [int(x) for x in input().split()]
    print(k//2*(a-b) + [0,a][k%2])

''' 233a Perfect Permutation '''
n = int(input())
if n%2:
    print(-1)
else:
    print(*range(n,0,-1))

''' 1455a strange functions '''
for _ in range(int(input())):
    num = list(input()); temp = num
    print(len(num))

''' 686a Free icecreams '''
n,x = [int(x) for x in input().split()]
distress = 0
for _ in range(n):
    temp = x
    x+=eval(input())
    #print(x)
    if x<0:
        distress+=1
        x = temp
print(x,distress)

''' 1391b Fix you '''
R = lambda : list(map(int,input().split()))
for _ in range(int(input())):
    n,m = R(); lug = []; ch = 0
    for _1 in range(n):
        temp = [*input()]
        ch+ = temp[-1] == 'R'
        lug.append(temp)
    ch+=lug[-1].count('D')
    print(ch)

''' 1481a Space navigation ''' #genius not me
for _ in range(int(input())):
    x0,y0 = list(map(int,input().split())); x = x0; y = y0
    u,d,l,r = map(input().count,'UDLR')
    print('NYOE S'[-d<=y<=u and -l<=x<=r])

''' 1304a 2 rabbits '''
for _ in range(int(input())):
    x,y,a,b = map(int,input().split())
    if (y-x)%(a+b):
        print(-1)
        continue
    print((y-x)//(a+b))

''' 1303a Erasing 0s '''
for arr in [list(map(int,input())) for _ in ' '*int(input())]:
    ind = [i for i,x in enumerate(arr,1) if x == 1]
    print(sum([j-i-1 for i,j in zip(ind[:-1],ind[1:]) if j-i>1]))

''' 1490a Dense array '''
for _ in range(int(input())):
    input()
    arr = [*map(int,input().split())];
    ins = 0
    for i,j in zip(arr[:-1],arr[1:]):
        i,j = sorted([i,j])
        pow = 0
        while j>i*(2**(pow+1)):
            pow+=1
        ins+=pow
    print(ins)

#How?
print(sum(len(bin(-1-y//-x))-3 for x,y in map(sorted,zip(a,a[1:]))))

''' 448a Rewards '''
cups = sum(map(int,input().split()))
medals = sum(map(int,input().split()))
n = int(input())

I = lambda : sum(map(int,input().split()))
num = lambda n,m: n//m +[0,1][n%m>0]
print('YNEOS'[num(I(),5)+num(I(),10)>int(input())::2])

'''1095a Repeating ciphers '''
n = int(input())
s = input(); i = 1
while i*(i+1)//2 <=n:
    print(s[i*(i+1)//2-1],end = '')
    i+=1
print()

''' 1487a Arena '''
for _,a1 in [[input(),input().split()] for a in ' '*int(input())]:
    arr = list(map(int,a1))
    if arr.count(max(arr)) == 1:
        print(arr.index(max(arr))+1)
    else:
        print(0)

for s in[*open(0)][2::2]:
    a=*map(int,s.split()),
    print(len(a)-a.count(min(a)))

''' 1480a Yet another string game '''
for _ in range(int(input())):
    move = 1;
    for c in input():
        if move%2:
            print(['a','b'][c=='a'],end='')
        else:
            print(['z','y'][c =='z'],end='')
        move+=1
    print()

''' 1462b Last year's substring '''
for _ in range(int(input())):
    input()
    s = input()
    if s[0]!='2':
        print('YNEOS'['2020' not in s[-4:]::2])
    else:
        if s[:4] == '2020':
            print('YES')
        elif s[:3] == '202' and s[-1] == '0':
            print('YES')
        elif s[:2] == '20' and s[-2:] == '20':
            print('YES')
        elif s[:1] == '2' and s[-3:] == '020':
            print('YES')
        else:
            print('NO')

''' 1466a Bovine Dilemma '''
for _ in range(int(input())):
    _,x = input(),[*map(int,input().split())]
    print(len(set(j - i for i in x[:-1] for j in x[x.index(i)+1:])))

''' 1474a Puzzle form the future '''
for _ in range(int(input())):
    a = [1]*int(input())
    b = [int(x) for x in input()]
    for i in range(1,len(a)):
        if a[i]+b[i] == a[i-1] + b[i-1]:
            a[i] = 0
    print(''.join(str(x) for x in a))

for s in[*open(0)][2::2]:
 r='';p=0
 for x in map(int,s[:-1]):
     p=x+(x+1!=p);r+=str(p-x)
 print(r)

for i in range(int(input())):
    l=int(input())
    a="1"
    b=input()
    for j in range(1,l):
        if(int(b[j])+1!=int(a[j-1])+int(b[j-1])):
            a+="1"
        else:
            a+="0"
    print(a)

''' 1405a Permutation Forgery ''' #Bullshit : never thought well enough
for _ in range(int(input())):
    input()
    print(*[*map(int,input().split())][::-1])

''' 1183a Nearest Interesting number '''
a = int(input())
while sum([int(x) for x in str(a)])%4>0:
    a+=1
print(a)

''' 1092b Teams forming '''
n = int(input())
skills = sorted([int(x) for x in input().split()])
print( sum(b-a for a,b in zip(skills[::2],skills[1::2]) ) )

'''1493a Anti Knapsack ''' #time limit exceeded
for _ in range(int(input())):
    n,k = [int(x) for x in input().split()]
    after = [*range(k+1,n+1)]; before = [*range(1,k)]
    if len(before) == 1:
        after.append(before[0])
    for i in range(len(before)-1):
        if k not in [sum(before[i:j]) for j in range(i+1,len(before))]:
            after.append(before[i])
    print(len(after))
    print(*after)

for _ in range(int(input())):
    n, k = map(int, input().split())
    print(n - k + k // 2)
    print(*range(k + 1, n + 1), *range((k + 1) // 2, k))

''' 1393a Rainbow Dash, Fluttershy and Chess coloring '''
for _ in range(int(input())):
    print(int(input())//2+1)

''' 1003a Polycarp's pockets '''
n = int(input()); l = [*map(int,input().split())]
print(max(map(l.count,set(l))))

'''214a System of equations '''
n,m = [int(x) for x in input().split()]
s = lambda a : a*(a+1)//2
count = 0
for a in range(min(m,n)+1):
    b = n+a - 2*s(a)
    if a - b + 2*s(b) == m and b>=0:
        count+=1
print(count)

''' 1451a subtract/divide ''' # some grand observation unforeseen
for _ in range(int(input())):
    n = int(input())
    if n>3:
        print([2,3][n%2])
    else:
        print(n-1)

for s in[*open(0)][1:]:n=int(s);print(min(n-1,2+n%2))

for _ in range(int(input())):
	n=int(input())
	if n<=3:print(n-1)
	else:print(["3","2"][n%2==0])

'''1223a CME '''
import math
for _ in range(int(input())):
    n = int(input())
    if n == 1:s
        print(1)
    if n == 2:
        print(2)
    else:
        print(n%2)

''' 2a winner ''' #did not work
d = {}; winner = 0; wins = 0
for _ in range(int(input())):
    ans = [*input().split()]
    d[ans[0]] = int(ans[1]) if ans[0] not in d.keys() else d[ans[0]]+int(ans[1])
    if wins<max(d.values()):
        wins = max(d.values())
        winner = [*d.values()].index(wins)
print([*d.keys()][winner])

''' 1207a There are 2 types of burgers ''' #why is my thing not working ?
for _ in range(int(input())):
    b,p,f = [*map(int,input().split())]
    h,c = [*map(int,input().split())]
    z = lambda x,y : h*x + c*y
    print(max(z(p,0),z(0,f),z(b-f,f),z(p,b-p)))
z(b-f,f)
z(p,b-p)

for k in range(int(input())):
    b,p,f=map(int,input().split())
    h,c=map(int,input().split())
    a=min(b//2,p)
    bd=min(b//2,f)
    print(max(a*h+min((b-2*a)//2,f)*c,bd*c+min((b-2*bd)//2,p)*h))

''' 1436a reorder '''
i = lambda : [*map(int,input().split())]
for _ in range(int(input())):
    n,m = i()
    print('YNEOS'[sum(i())!=m::2])

''' 1249a Yet another dividing into teams '''
for _ in range(int(input())):
    input()
    arr = sorted([*map(int,input().split())])
    print(1+ int(1 in [j-i for i,j in zip(arr[:-1],arr[1:])]))

R=lambda:map(int,input().split())
q,=R()
for _ in[0]*q:
    R()
    a=sorted(R())
    print(3-min(y-x for x,y in zip([a[0]-2]+a,a)))

''' 1206a choose 2 numbers ''' # why didnt max(a) max(b) occur to me at all?
input()
a = sorted([*map(int,input().split())])
input()
b = sorted([*map(int,input().split())])
flag = 1
for x in a:
    for y in b:
        if x+y not in a and x+y not in b and flag == 1:
            print(x,y)
            flag = 0
            break

'Wow!'
print(*(max(map(int,input() and input().split())) for _ in'00'))

I=lambda:max(map(int,input() and input().split()))
print(I(),I())

''' 1466b Last minute enhancements '''
for _ in range(int(input())):
    input(); s = set()
    for x in [int(x) for x in input().split()]:
        s.add(x) if x not in s else s.add(x+1)
    print(len(s))

    # what the hell man, could not get the sum at all. Look at this solution!
for _ in range(int(input())):
	n=int(input())
	l=sorted(list(map(int,input().split())))
	s=set()
	for x in l:
		if x not in s:
			s.add(x)
		else:
			s.add(x+1)
	print(len(s))

''' 1500a going home ''' #beauty
n = int(input())
a = [int(x) for x in input().split()]
d ={}
for x in range(n):
    for y in range(x):
        if a[x]+a[y] not in d:
            d[a[x]+a[y]] = [x,y]
        else:
            z,w = d[a[x]+a[y]]
            if x not in [z,w] and y not in [z,w]:
                print('YES')
                print(*[1+i for i in (x,y,z,w)])
                exit()
print('NO')

''' 1452a Robot motion ''' #IGotIt
for x,y in [[*map(int,input().split())] for _ in " "*int(input())]:
    y,x = min(abs(x),abs(y)), max(abs(x),abs(y))
    x = x - y
    print( 2*y + min(4*(x//2) + x%2, 2*x - [0,1][x!=0]) )

''' 1367b Even array ''' #IGotIt
for _ in '0'*int(input()):
    n = int(input())
    arr = [int(x) for x in input().split()]
    even = sum(1 for a in arr if a%2==0)
    odd = sum(1 for a in arr if a%2==1)
    if even - odd != n%2:
        print(-1)
        continue
    ans = sum([abs(i%2 - a%2) for i,a in enumerate(arr)])
    print([ans//2,-1][ans%2])

#END OF FILE: 1000 Lines of Code, exactly!!!
