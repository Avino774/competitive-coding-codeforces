
nums = [1,1,1,1]
count = 0

for i,one in enumerate(nums):
    for j,two in enumerate(nums):
        if one == two and i<j : count = count + 1
count

#num in jewels
jewels = 'aA'
stones = 'aAAbbb'
count = 0
for i,letter in enumerate(stones):
    if letter in jewels:
        count = count + 1
count

nums = []
for i in range(2000,3200):
    if i%7 == 0 and i%5!=0:
        nums.append(i)
nums

x = int(input())
def fact(x):
    if x == 0:
        return 1
    else:
        return x*fact(x-1)
print(fact(x))

n = int(input())
dict = {i:i*i for i in range(1,n+1)}
print(dict)

l = input().split(',')
t = tuple(l); print(t)

l = [str(i) for i in range(2000,3200) if i%7==0 and i%5!=0]

#%%
class InputOutString(object):
    def __init__(self):
        self.s = ""
    def get_string(self):
        self.s = input()
    def put_string(self):
        print(self.s.upper())

strobj = InputOutString()
strobj.get_string()
strobj.put_string()

#%%
import math
c = 50
h = 30
value = []
items  = [x for x in input().split(',')]

for d in items:
    value.append(str(int(round(math.sqrt(2*c*float(d)/h)))))

print(','.join(value))

#%%
x = int(input())
y = int(input())
matrix = [[i*j for j in range(y)] for i in range(x) ]
matrix

#%%
words = [x for x in input().split(',')]
words.sort()
print(words)

#%%
s = [word for word in input().split(' ')]
print(" ".join(sorted(list(set(s)))))

#%%
import math
binary = [x for x in input().split(',')]
fives = []
for i,b in enumerate(binary):
    intb = int(b,2)
    if not intb%5: fives.append(b)
fives

#%%
evens = []
for num in range(1000,3000):
    s = str(num)
    if (int(s[0]) %2 != 0 or
        int(s[1]) %2 !=0 or
        int(s[2]) %2 !=0 or
        int(s[3]) %2 !=0 ):
        continue
    evens.append(num)
print(evens)

#%%
s = input()
d = {'digits': 0 , 'letter':0}

for c in s:
    if c.isdigit(): d['digits'] +=1
    elif c.isalpha(): d['letter'] +=1
    else: pass
print(f"There are {d['digits']} digits and {d['letter']} letters")

#%%
a = input()
print(sum([int(str(a)*i) for i in range(1,5)]))

#%%
values = input()
nums = [x for x in values.split(',') if int(x)%2]
nums

#%%
items = input().split(' ')
sign = 1
netam = 0

for item in items:
    if item.isalpha():
        if item == 'D': sign = 1
        else: sign = -1
    else:
        netam = netam + sign*int(item)
print(netam)

#%%

while True:
    s = input()
    if not s:
        break
    l.append(tuple(s.split(',')))
print(sorted(l,key = itemgetter(0,1,2)))

#%%
import math
pos = [0,0]

while True:
    s = input()
    if not s: break
    moves = s.split(" ")
    dir = moves[0]; steps = int(moves[1])

    if dir == 'UP': pos[1] += steps
    elif dir == 'DOWN': pos[1] -=steps
    elif dir == 'RIGHT': pos[0] +=steps
    elif dir == 'LEFT': pos[0] -=steps

print(math.sqrt(pos[0]**2 + pos[1]**2))

#%%
words = input().split(" ")
dict = {word:0 for word in set(words)}
for word in words:
     dict[word] += 1
print([dict[word] for word in set(words)])

#%%
class Person:
    #define class parameters
    def __init__(self,name = 'Mohamed Lee'):
        self.name = name

jeffrey = Person('jeffrey')
print('%s name is %s' %(jeffrey.name,jeffrey.name) )

nico = Person()
nico.name = 'Nico'
print(nico.name)

#%%
lambda num1,num2 : num1+ num2

#%%
n = int(input())
sum([float(float(x)/(x+1)) for x in range(1,n+1)])

#%%
import itertools
li = [1,2,3]
print(list(itertools.permutations(li)))

print(input()[::2])
print(input()[::-1])

#%%
s = input()
dict = {letter:0 for letter in set(s)}
for letter in s:
    dict[letter] += 1
print('\n'.join('%s%s'%(k,v) for k,v in dict.items()))

li = list(set(input().split(',')[::-1]))
li

li = [24,88,35,70,88,120,12]
#something like R?

subjects = ['I','You']; verbs = ['Play','Love']; objects = ['Hockey','Football']
print([col for col in list(zip(subjects,verbs,objects))])

#%%
"""Binary Search"""
li = [43,12,345,54,2,345]
def bin_search(which_elem):
    low = 0
    high = len(li)-1
    mid = int(math.floor((lo+high)/2))

    while high>=low:
        if li[mid]<which_elem:
            high = mid
        elif li[mid]>which_elem:
            low = mid
        elif li[mid] == which_elem:
            return mid
        else:
            pass
        mid = (lo+high)/2
    return -1

#%%
'''string compression'''

s = 'abbass'
buffer = str()
seen = str()

for i,letter in enumerate(s):
    if letter == s[i-1]:
        count =+ 1
    else:
        count = 1
        seen = seen + letter

print(seen)


#%%
steps = input()
level = 0
height = 0
mountains = 0
valleys = 0

for letter in steps:
    if height > 0: level = 1
    elif height == 0: level = 0
    else: level = -1

    if letter == 'U':
        height += 1
    else:
        height-= 1

    if height == 0 and level == 1: mountains +=1
    elif height == 0 and level == -1 : valleys +=1

print('%s %s'%(mountains,valleys))

#%%
n = int(input())
grades = [int(grade) for grade in input().split(' ')]
grades1 = [grade if (grade<40 or grade%5 <=2) else grade+(5 - grade%5) for grade in grades]
for grade in grades1:
    print(grade)

n = int(input().strip())
for i in range(n):
    grade = int(input().strip())
    if grade<37: print(grade)
    elif grade%5 > 2: print((grade//5+1)*5)
    else: print(grade)

a = [int(num) for num in input().split(',')]
b = [int(num) for num in input().split(',')]
aps = 0
bs = 0
for i,j in zip(a,b):
    if i>j: aps=aps+1
    elif i<j: bs = bs+1
    else: pass
print("%s %s"%(aps,bs))

i = 1000000001
5*i

#%%
n = int(input().strip())
mat = []
for i in range(n):
    alp = input().split(' ')
    mat.append([int(row) for row in alp])
mat
diag = abs(sum([mat[i][i] - mat[i][n-1-i] for i in range(n)]))
diag

#%%

n = int(input().strip())
nums=[int(num) for num in input().split(' ')]

for num in nums:
    if num>0:
        pf += 1/n
    elif num<0:
        nf += 1/n
    else:
        of +=1/n
print(pf,nf,of)

#%%
n = int(input().strip())
for i in range(n):
    for j in range(n):
        if j>=n-1-i:
            print(' #',end = ' ')
        else:
            print('  ', end = ' ')
    print()
#%%
'''minimax sums'''

n = int(input().strip())
arr = [int(num) for num in input().split(' ')]
print( min([sum(arr) - num for num in arr]), max([sum(arr) - num for num in arr]))


#%%
test= int(input().strip())
for i in range(test):
    threshold = int(input().strip())
    arr = [int(num) for num in input().split(' ')]
    if len([time for time in arr if time<=0]) < threshold:
        print('NO')
    else:
        print('YES')


#%%
s,t = [int(num) for num in input().split(' ')]
a,b = [int(num) for num in input().split(' ')]
m,n = [int(num) for num in input().split(' ')]
na = [a + int(num) for num in input().split(' ')]
nb = [b + int(num) for num in input().split(' ')]

nas = [apple for apple in na if apple>=7 and apple<=10]
nbs = [orange for orange in nb if orange>=7 and orange<=10]
print(len(nas)); print(len(nbs))

n = int(input().strip())

if not n%2:
    print('Wierd')
elif:
    if n>=2 and n<=5:
        print('Not Weird')
    elif n>=6 and n<=20:
        print('Wierd')
    else:
        print('Not Weird')

#%%
a = int(input().strip())
b = int(input().strip())
print(a+b)
print(a-b)
print(a*b)

#%%
x1,v1,x2,v2 = [int(num) for num in input().split(' ')]
v2 = v2 - v1
dist = x2 - x1
count = 1

while True:
    x2 = x2 + v2*count
    if x2 - x1 > dist:
        print('NO')
        break
    else:
        count +=1
        if x2 == x1:
            print('YES')

#%%

x,y,z,n = [int(input()) for _ in range(4)]
print([[a,b,c] for a in range(x+1) for b in range(y+1) for c in range(z+1) if a+b+c != n ])

n = int(input())
a = [int(num) for num in input().split(' ')]
print(max([x for x in a if x!=max(a)]))

#%%
n = int(input())
student = []
for i in range(n):
    name = input()
    marks = float(input())
    student.append(list(name,marks))

print(min([student[i][0] for i in range(n) if student[i][1]!= min(student[:][1])]))

student = [['been',5],['dee',4]]; n = 2
#n = int(input())
#student = []
for i in range(n):
    #name = input()
    #marks = float(input())
    #student.append([name,marks])
    pass
minas = min(student[i][1] for i in range(n))
print(min([student[i][0] for i in range(n) if student[i][1]!= minas]))
minas

#nearly the best
student = [[input(),float(input())] for i in range(int(input()))]
second_highest = sorted(list(set([marks for name,marks in student])))[1]
print('\n'.join(sorted([name for name,marks in student if marks == second_higest])))

#%%
student_marks = {}
for _ in range(int(input())):
    name,*marks = input().split(' ')
    student_marks[name] = list(map(float,marks))
print('%.2f'%(sum(student_marks[input()])/len(student_marks)))

s = [1,2,3]
s.insert(2,4)
s

li = []
for _ in range(int(input())):
    command = input()
    if command.contains('insert'):
        name,pos,num = command.split(' ')
        li.insert(pos,num)
    elif command == 'print':
        print(li)
    elif command.contains('remove'):
        name,num = command.split(' ')
        li.remove(num)
    elif command.contains('append'):
        name,num = command.split(' ')
        li.append(num)
    elif command=='sort':
        li = sorted(li)
    elif command == 'pop':
        li.pop()
    elif command == 'reverse':
        li = li[::-1]

li = []
for _ in range(int(input())):
    comin = input().split()
    command = comin[0]
    args = s[1:]
    if command!='print':
        command += '(' + ','.join(args) + ')'
        eval('li.'+command)
    else:
        print(li)

#%%
n = int(input())
print(hash(tuple([int(num) for num in input().split(' ')])))

#%%
p = 1
for _ in range(int(input())):
    n,d = list(map(int,input().split(' ')))
    p *=n/d
print(p)

s = 'sasa'
print( ''.join(i.lower() if i.isupper() else i.upper() for i in s) )

count = 0; string = 'abcdcdc'; substr = 'cdc'
print(sum([1 for i in range(len(string) - len(substr) + 1) if string[i:i+len(substr)] == substr]))

funs = ['isalnum()','isalpha()','isdigit()','islower()','isupper()']
string = 'qA2'
print('\n'.join([str(any([eval('c.'+command) for c in string])) for command in funs]))

#%% text wrap
substr = []; rem = []; parah = []; s=input(); max_width = int(input())
if len(s)%max_width!=0:
    rem = s[len(s)-(len(s) % max_width):]
for i,char in enumerate(list(s)):
    substr.append(char)
    if (i+1)%max_width == 0:
        parah.append(substr)
        substr = []
parah.append(rem)
print( '\n'.join([''.join(c) for c in parah]) )

#%%
s = []
dashes = [for i in range((n-1)/2) for j in range(3*(n-1)/2)]

#%%circshift
i,j,n = float(input().split(' '))
print(sum( [1 for num in range(i,j+1) if (abs(num - float(str(num)[::-1])) % k ) % 2 == 0 ] ))

n,k,q = [int(num) for num in input().split(' ')]
a = [int(num) for num in input().split(' ')]
queries = [int(num) for num in input().split(' ')]

k = k%n
for i in range(k):
    a = a[-1] + a[:]
    del a[-1]
print('\n'.join(str(num) for num in a[queries]))

a=[1,2,3]
a[-2:]

temp = a[-1]
del a[-1]
a[1:] = a[:]
a[0] = temp

a = a[-k:] + a[:]
del a[-k:]

for each in range(q):
    print(a[int(input())])

n = int(input())
p = [int(num) for num in input().split()]
for x in range(n+1):
    p_y = [i for i in p if p[i] == x]
    y = [i for i in p_y if p[i] == y]

n,k = map(int,input().split(' '))
c = [int(num) for num in input().split()]
for i in range(n):
    e = e - k -2*c[(i+k)%n]
    if (i+k) %n == 0:
        break
print(e)

#%%

t = int(input())
for _ in range(t):
    n = list(input())
    print( sum([1 if int(n)%int(c)==0 and int(c)! = 0 for c in n]) )

commands = ['remove','discard','pop']
n = int(input())
s = set([x for x in input().split(' ')])
N = int(input())

for _ in range(N):
    enter, num = input(), int(input())
        eval('s.'+enter+'('+num+')')
print(sum(s))

n = int(input())
english = set(input().split(' '))
n = int(input())
french = set(input().split(' '))
print(len(english | french))

n = int(input())
s = set(map(int,input().split(' ')))
for _ in range(int(input())):
    command = input().split(' '); s2 = set(map(int,input().split(' ')))
    eval('s.{0}({1})'.format(command[0],s2))
print(sum(A))

arr = [1,2,3,4]
A = [2,3]; B = [3,4];
sum([(i in A) - (i in B) for i in arr])

l = [int(x) for x in '1 2 3 6 5 4 4 2 5 3 6 1 6 5 3 2 4 1 2 5 1 4 3 6 8 4 3 1 5 6 2'.split(' ')]
i = 1
sum([i in l for i in set(l)])

#if B.intersection(A) == A

''' append and delete '''

s = list(input()); t = list(input()); k = int(input())
changed_s = [x for x in s if x not in t]
changed_t = [x for x in t if x not in s]

if k == len(changed_s) + len(changed_t):
    print('Yes')
elif len(s) == len(changed_s) and k >= len(s) + len(t):
    print('Yes')
else:
    print('No')

import cmath
print(*cmath.polar(complex(input())),sep = '\n')

''' Cut the sticks '''
n = int(input())
sticks = [int(x) fox x in input().split(' ')]

while sticks! = []:
    min_stick = min(sticks)
    sticks = [i-min_stick for i in sticks]
    sticks = [i if i!=0 for i in sticks]
    print(len(sticks))

''' infinite string '''

string = input()
n = int(input())
count = n//len(string)*sum([1 for i in string if i == 'a'])
for i in string[:n % len(string)]:
    if i == 'a':
        count += 1
print(count)

''' jumping clouds '''
n = int(input())
c = [int(x) for x in input().split(' ')]
count = 0

while i<n:
    if c[i+2] == 0 and i+2 <=n-1:
        i = i+2; count = count+1
    elif c[i+2] == 1 and c[i+1] == 0 and i+1<=n-1:
        i = i+1; count = count+1
    else:
        pass
print(count)

''' Bitwise manipulation; ACMICPC teams '''
persons = []
for _ in range(int(input())):
    persons.append([int(x) for x in input().split()])

def know_union(ar1,ar2):
    return sum((m or n) for m,n in zip(ar1,ar2))

vast = []
num_vast = 0

for i in range(n-1):
    for j in range(i+1,n):
        vast.append(know_union(persons[i],persons[j]))
print(max(vast))
print(vast.count(max(vast)))

''' Taum and b day '''
for _ in range(int(input())):
    b,x = [int(x) for x in input().split()]
    bc,wc,z = [int(x) for x in input().split()]
    print( b*min(bc,wc+z) + w*min(wc,bc+z))

''' Row sum column sumswap elements '''
# what is the condition for balls to be swapped into containers?
for _ in range(int(input()):
    M = []
    n = int(input())
    for i in range(n):
        M.append([int(x) for x in input().split()])

    box_totals = sorted([sum(x) for x in M])
    ball_totals = sorted([sum(x) for x in zip(*M)])

    if box_totals == ball_totals:
        print('Possible')
    else:
        print('Impossible')

''' Kaprekar numbers '''
kaps = []
for num in range(map(int,input().split())):
    sq = str(num**2)
    if len(sq)%2 !=0:
        sq = '0'+sq
    l,r = sq[:len(sq)/2], sq[len(sq)/2:]
    if int(l) + int(r) == num:
        k.append(num)
print(' '.join([str(x) for x in num]))

''' Magic Square '''


''' Encryption '''
s = [c for c in input() if c! = ' ']
import math
for num in range( floor(math.sqrt(len(s))), ceil(math.sqrt(len(s))) + 1)
    row,column = num, ceil(math.sqrt(len(s))) + 1 - num
    if row * column < min:
        best_row, best_column = row,column
        min = row*column
new_s = []
for i in range(best_row):
    buffer = ''
    for j in range(best_column):
        buffer = buffer + s[i*best_row + j]
    new_s.append(buffer)
print(' '.join(new_s))

''' find subsequence in string '''

for _ in range(int(input())):
    string = input()
    hk = 'hackerrank'
        for pos,c_dash in enumerate(hk):
            if c == c_dash and pos<=i:
                string = string[i:]; hk = hk[pos:]
                i = 0; c = ''
            if hk = '':
                print('YES')
                flag = 1
                break
    if flag == 0:
        print('NO')

for _ in range(int(input())):
    string = input()
    hk = 'hackerrank'

    for char in string:
        if char == hk[i]:
            i = i+1

''' find min distances between array '''

n = int(input())
arr = [int(x) for x in input().split()]
dist = [abs(i-j) for i in range(len(arr)-1) for j in range(i+1,len(arr)) if arr[i] == arr[j]]
print(min(dist))

''' how many games '''
p,d,m,s = (int(x) for x in input().split())
count = 0
while s>=0:
    s = s-p; count = count + 1
    p = max([p-d,m])
print(count)

''' time in words '''

time = 1
t = int(input());

count = 0
while True:
    time = time + 3*2**count
    count = count + 1
    if time>t:
        print(time - t)
        break


''' cavity search '''

grid = [[int(elem) for elem in input()] for _ in range(int(input()))]
for i in range( 1, len(grid) -1 ):
    for j in range( 1, len(grid[i]-1) ):
        cavity = max( grid[a][b] for a,b in range( zip([i-1,i+2],[j-1,j+2]) ))
    print(''.join([ str(elem) if elem != cavity else 'X' for elem in grid[i] ] ))
#didnt get that


''' delete adjacent characters '''

s = input(); flag = 0
while i < len(s):
    if flag ==  1:
        i = 0
        flag = 0
    else:
        i = i+1
    if s[i] == s[i+1]:
        del s[i]
        del s[i]
        flag = 1
        if s == '':
            print('Empty String')
            break
print(s if s!='')

''' camel case
s = input()
l = []; prev = 0
for char in s:
    if char.isupper():
        if s.index(char) == len - 1:
            l.append(s[prev:])
        else:
            l.append(s[prev:s.index(char)])
            prev = s.index(char)
print('\n'.join(l)) '''

" Password "
numbers = "0123456789"
lower_case = "abcdefghijklmnopqrstuvwxyz"
upper_case = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
special_characters = "!@#$%^&*()-+"

n = int(input())
password = input()
criteria = [numbers, lower_case, upper_case, special_characters]
count = 0
for crit in criteria:
    if any(password) not in crit:
        count +=1
print(max(count, 6 - n ))

''' two characters '''

''' rotations '''
alphabet = 'abcdefghijklmnopqrstuvwxyz'
n = int(input())
s = input()
k = int(input()) % 26
r = alphabet[k:] + alphabet[0:k]

buffer = ''
for char in s:
    if char.isalpha():
        if char.isupper():
            buffer = r[alphabet.index(char.lower())].upper()
        else:
            buffer = r[alphabet.index(char)]
    else:
        buffer = char
    print(buffer,end = '')

''' changing sos '''
altered_sos = list(input())
sos = list('SOS');
count = 0
while i<len(altered_sos)-3:
    for each,real in zip(altered_sos[i:i+3],sos):
        count += 1 if each!=real else 0
    i+=3
print(count)

''' subsequence '''

for _ in range(int(input())):
    subseq = list('hackerrank'); flag = 0
    string = list(input()); x = [-1]*len(subseq)

    for i,char in enumerate(string):
        for pos,char1 in enumerate(subseq):
             if char == char1 and pos<=i:
                x[pos] = 0

    print('NO' if any(i==-1 for i in x) == True else 'YES')

''' pangrams '''

s = list(input()).lower()
alpha = list('abcdefghijklmnopqrstuvwxyz'); flag = 0
for char in alpha:
    if char not in s:
        print('not pangram')
        flag = 1
        break
if flag == 0:
    print('pangram')

''' weight of characters '''
alphabet = 'abcdefghijklmnopqrstuvwxyz'
weight = {i:a for i,a in zip(range(1,27),alphabet) }
string = input()

string_wt = set()
for i in range(len(string)):
    cnt = cnt+1 if (i+1!=len(string) and string[i] == string[i+1]) else 1
    string_wt.add(weight[string[i]]*cnt)
print( 'Yes' if int(input()) in string_wt else 'No' for _ in range(int(input)) )

''' Ladybugs '''
for _ in range(int(input())):
    n = int(input())
    b = input()
    flag = 0
    for char in sorted(set(b)):
        if char.isalpha() and b.count(char)==1:
            flag = 1
            break
    if b.count('_') == 0 and flag == 0:
        for i in range(1,n-1):
            if b[i] != b[i+1] and b[i] != b[i-1]:
                flag = 1
                break
    print( 'YES' if not flag else 'NO')

b = ['b','b','r','b','r','b','_','_']
sorted(b)

''' Manasa and stones '''
n,a,b = [int(input())]*3 #why does this not work
count = 0; last_stone = set()
while count<n+1:
    last_stone.add((n-count)*a + count*b)
    count += 1
print(' '.join( str(x) for x in last_stone) )

''' cavity maps: good for contours '''
n = int(input());
grid = []
for _ in range(n):
    grid.append(input())

for i in range(1,n-1):
    for j in range(1,n-1):
        cavity = max( [grid[i][j] for i,j in zip( range(i-1,i+2), range(j-1,j+2) )])
        print('X' if grid[i][j] == cavity else grid[i][j], end = '')
    print()

a = [1]
a[-1]

''' even loaves of bread '''
n = int(input())
bread = [int(x) for x in input().split()]
if sum(bread) % 2 !=0 :
    print('No')
else:
    idx = [i for i,x in enumerate(bread) if x%2 == 1]
    print(sum(idx[i+1] - idx[i])*2 for i in range(0,len(idx),2))

''' beautiful triplets '''
n,d = [int(x) for x in input().split()]
arr = [int(x) for x in input().split()]
print( sum([1 if arr[j]-arr[i] == d and arr[k] - arr[j] == d for k in range(j+1,n) for j in range(i+1,n-1) for i in range(n-2)]) )
print( sum([i-d in arr and i+d in arr for i in arr]))

''' chocolates '''
for _ in range(int(input())):

    n,c,m = list(map(int,input().split()))
    eaten = n//c; chocs = 0
    wraps = n//c
    while wraps >= m:
        eaten = eaten + chocs
        wraps = chocs + wraps
        chocs = wraps//m
        wraps = wraps - chocs
    print(eaten)


''' entries and exits '''
n,t = [int(x) for x in input().split()]
widths = [int(x) for x in input().split()]

for _ in range(t):
    entry,exit = [int(x) for x in input().split()]
    print(min(widths[entry:exits+1]))

''' Math workbook '''

n, k = [int(x) for x in input().split()]
arr = [int(x) for x in input().split()]
pages = 1; index = {}

for i in arr:
    temp = pages
    pages = pages + i//k + int(i%k>0)
    index[temp] = pages

for start,finish in index.items():
    count = sum([ page in list(range(1,i+1)) for i in arr for page in range(start,finish+1)])
print(count)

''' flatland iss '''

n_city,n_ss = list(map(int,input().split()))
ss = [int(x) for x in input().split()]
not_ss = [i for i in range(n) if i not in ss]
max([ min([abs(not_s - s) for s in ss]) for not_s in not_ss ])

''' math workbook '''
n, k = [int(x) for x in input().split()]
arr = [int(x) for x in input().split()]
start_page = 1; count = 0

for no_of_problems in arr:
    count += sum([1 for prob in range(start_page,no_of_problems+1,1) if prob == prob//k + start_page ])
    start_page = start_page + no_of_problems//k + int(no_of_problems%k>0)
print(count)

list(range(10,5))

'descent solution'
n,k = [int(x) for x in input().strip().split()]
problems_in_chapters = [int(x) for x in input().strip().split()]
count = 0
page = 1
for chapter_problem in problems_in_chapters:
    for current_problem in range(1,chapter_problem + 1):
        if(page == current_problem):
            count = count + 1
        if ((current_problem % k == 0 ) or current_problem == chapter_problem):
            page = page + 1
print(count)

''' append and delete '''

s = list(input())
t = list(input())
k = int(input())

while s!=t[:len(s)]:
    del s[-1]; k -= 1
while s!=t[:len(s)]:
    s.append(t[len(s)-1]); k-=1

print('Yes' if k>=0 else 'No')

    s = list(input())
    t = list(input())
    k = int(input())

    while s!=t[:len(s)]:
        del s[-1]; k -= 1
    while s!=t[:len(s)]:
        s.append(t[len(s)-1]); k-=1

    print('Yes' if k>=0 else 'No')

print('Yes' if k == abs(len(s) - len(t) or k>= len(s)+len(t) else 'No')


import sys

s = input().strip()
t = input().strip()
k = int(input().strip())

lead = 0
for i in range(min(len(s),len(t))):
    if s[i] != t[i]:
        lead = i
        break
    else:
        lead = i + 1

d = len(s) - lead + len(t) - lead

if k >= len(s) + len(t):
    print("Yes")
elif d <= k and (d % 2) == (k % 2):
    print("Yes")
else:
    print("No")

s = input().strip()
t = input().strip()
k = int(input().strip())

ls = len(s)
lt = len(t)

lcp = 0 # Length of common prefix
while lcp <= ls-1 and lcp <= lt-1 and s[lcp] == t[lcp]:
    lcp += 1

if k >= ls + lt:
    print ("Yes")
elif k >= ls + lt - 2*lcp and (k - ls - lt + 2*lcp)%2 == 0:
    print ("Yes")
else:
    print ("No")

''' grading students '''
for _ in range(int(input())):
    gr = int(input())
    print( gr//5 +5 if gr%5>=3 and gr>=38 else gr)

''' beautiful strings '''
for _ in range(int(input())):
    a = input(); n = len(a)

''' 2 alternating characters '''
n = int(input())
s = list(input())
s = [char for char in s if s.count(char)!=1]; prev = s[:]
while True:
    prev = s[:]
    for i in range(len(s)-1):
        if s[i] == s[i+1]:
            del s[i]; del s[i+1]
            break
    if prev == s:
        break
print(len(s))

''' 1 '''
def char_range(c1, c2):
    for c in range(ord(c1), ord(c2)+1):
        yield chr(c)

def is_alternating(s):
    for i in range(len(s) - 1):
        if s[i] == s[i+1]:
            return False
    return True
len_s = int(input())
s = input()
max_len = 0
for c1 in char_range('a', 'z'):
    for c2 in char_range('a', 'z'):
        if c1 == c2:
            continue
        new_string = [c for c in s if c == c1 or c == c2]
        new_string_len = len(new_string)
        if is_alternating(new_string) and new_string_len > 1:
            if new_string_len > max_len:
                max_len = new_string_len
print(max_len)

''' 2 '''

import string
n = int(input())
s = input()
ans = 0
for x in string.ascii_lowercase:
    for y in string.ascii_lowercase:
        if x not in s or y not in s:
            continue
        filtered = [c for c in s if c == x or c == y]
        filt_len = len(filtered)
        for i in range(1, filt_len):
            if filtered[i] == filtered[i - 1]:
                filtered = []
                break

        filt_len = len(filtered)
        if filt_len >= 2:
            ans = max(ans, filt_len)
print(ans)

''' common substrings '''
for _ in range(int(input())):
    s1 = input()
    s2 = input()
    if any([char in s2 for char in s1]) == False :
        print('NO')
    else:
        print('YES')

''' string construction '''
for _ in range(int(input())):
    s = input()
    print(len(set(s)))

''' funny strings '''

for _ in range(int(input()):
    s = list(map(ord,list(input())))
    sd = s[1:] - s[:-1]
    rd = reversed(s)[1:] - reversed(s)[:-1]
    print('Funny' if sd == rd else 'Not Funny')

''' anagrams '''

def anagram(s):
    if len(s)%2 !=0:
        return -1
    first = sorted(map(ord,s[:len/2+1])); second = sorted(map(ord,s[len/2+1:]))
    return sum([1 for char1,char2 in zip(first,second) if char1!=char2])

for _ in range(int(input())):
    print(anagram(input()))

first = 'xaxb'; second = 'bbxx'
sum( [first.count(char) - second.count(char) for char in set(first)] )

' working '
def initDict (str, d):
    for c in str:
        d[c] = d[c] + 1
    return d

T = int(raw_input())
t = 0
while t < T:
    inp = raw_input()
    abLength = len(inp)
    if abLength%2 != 0:
        print -1
    else:
        str1 = inp[:abLength/2]
        str2 = inp[abLength/2:]
        d1 = dict.fromkeys(string.ascii_lowercase, 0)
        d2 = dict.fromkeys(string.ascii_lowercase, 0)
        d1 = initDict(str1, d1)
        d2 = initDict(str2, d2)
        count = 0
        for c in d2.iterkeys():
            count = count + max(d2[c] - d1[c], 0)
        print count
    t = t + 1

''' anagrams '''

def init_dict(d,s):
    for c in s:
        d[c] += 1
    return d

for _ in range(int(input())):
    s = input()
    str1 = s[:len(s)//2]
    str2 = s[len(s)//2:]
    d1 = dict.fromkeys(string.ascii_lowercase,0)
    d2 = dict.fromkeys(string.ascii_lowercase,0)
    d1 = init_dict(d1,str1)
    d2 = init_dict(d2,str2)
    count = 0
    for c in d2.iterkeys():
        count  = coun + max(d2[c] - d1[c],0)
    print(count)

''' palindromes, finally '''
for _ in range(int(input())):
    s = list(map(ord,list(input())))
    diff = 0
    for char1,char2 in zip(s,s[::-1]):
        diff = diff + abs(char1 - char2)
    print(diff)

''' palindrome index '''

def is_palin(s):
    for char1,char2 in zip(s,s[::-1]):
        if char1!=char2:
            return False
    return True

for _ in range(int(input())):
    s = list(map(ord,list(input())))
    flag = 0
    for i,char in s:
        if is_palin( [c for c in s if c!=char] ):
            print(i); flag = 1
            break
    if flag == 0:
        print(-1)

t = int(input())

def is_palindrome(s):
	return s == s[::-1]

def solve(s):
	i, j = 0, len(s) - 1
	while (i < j) and (s[i] == s[j]):
		i += 1
		j -= 1
	if i == j:
		return 0
	if is_palindrome(s[i + 1 : j + 1]):
		return i
	if is_palindrome(s[i:j]):
		return j
	raise AssertionError

for _ in range(t):
	print(solve(input()))

''' my own code for this one '''
def is_palin(s):
    return s == s[::-1]

for_ in range(int(input())):
    s = input()
    i,j, flag = 0,len(s) - 1, 0
    while i<j and s[i]==s[j]:
        i+=1; j-=1
    if i==j:
        print(-1)
    if is_palin(s[i+1:j+1]):
        print(i); flag = 1
    if is_palin(s[i:j]):
        print(j); flag = 1
    if flag == 1:
        print(-1)

''' palindrome GOT'''
for _ in range(int(input())):
    s = list(input())
    if len(s)%2!=0:
        s = s[:(len-1)//2] + s[(len+1)//2:]
    if not any([s.count(char)%2==0  for char in set(s)]):
        print('NO')
    else:
        print('YES')

s = input(); cnt = {}
for char in set(s):
    cnt[c] = s.count(char)
nodd = len(filter(lambda x:x%2==1,cnt.values()))
print('NO' if nodd<=1 else 'YES')

''' gemstones '''
import string
stone = string.ascii_lowercase
for _ in range(int(input())):
    stone = stone.intersection(set(input()))
print(len(stone))

''' separate the numbers '''

for _ in range(int(input())):
    s = input(); n = len(s)
    x = ''
    for i in s[:-1]:
        x = x+i
        y = int(x)
        z = ''
        while len(z)<n:
            z = z+str(y)
            y+=1
        if n == len(z) and s == z:
            print('YES',x)
            break
        else:
            print('NO')

''' adjacent character deletions '''
for _ in range(int(input())):
    s,x = list(input()),s[0]
    for j in s:
        if x[-1] == j:
            continue:
        else:
            x.append(j)
    print(len(s) - len(x))

'2 liner'
for word in [input() for _ in range(int(input()))]:
    print( len( [i for i in range(len(s)-1) if s[i]== s[i+1]] )
    )

''' binary beautiful string '''
n,s = int(input()),input()
x = ''; change = 0
for char in s:
    x = x+char
    if '010' in x:
        change+=1
        x = ''
print(change)

s = '10101010'
print(input() * False + str(s.count('010')))

'''making anagrams '''
import string

print(
sum([abs(s1.count(c) - s2.count(c)) for c in string.ascii_lowercase])
)

''' watermelon '''
w = int(input())
print('YES' if w!=2 and w%2==0 else 'NO')

''' way too long words '''
for word in [input().strip() for _ in range(int(input()))]:
    if len(word) > 10:
        print(word[0] + str(len(word[1:-1])) + word[-1])
    else:
        print(word)

''' theatre square '''
m,n,a = [int(x) for x in input().split()]
print( (m//a+1)*(n//a+1) )
print(((m - 1) // a + 1) * ((n - 1) // a + 1))

''' winner '''
players = {}; count = {}
for _ in range(int(input())):
    name,score = input().split()
    if name not in players:
        players[name] = int(score)
        count[name] = 1
    else:
        players[name] += int(score)
        count[name] += 1

min_name = ''
min_count = 1000
for name in [p for p in list(players) if players[p] == max(players.values())]:
    if min_count > count[name]:
        min_count = count[name]
        min_name = name
print(min_name)

''' breaking the records going pro '''
scores = list( map(int,input().split()) )
max = scores[0]; change_max = 0
min = scores[0],change_min = 0
for score in scores:
    if max < score:
        change_max+=1
        max = score
    if min>score:
        change_min+=1
        min = score
print(change_max,change_min)

''' migratory birds '''
input(); arr = [int(x) for x in input().split()]
d = {bird:arr.count(bird) for bird in set(arr)}
for bird,freq in sorted(d):
    if freq == max(d.values()):
        print(bird)
        break

''' day of the programmer '''
year = int(input())
if year == 1918:
    print(str(11+14)+'.09.1918')
else:
    if True in [year%4 == 0 and year%100!=0, year%400 == 0]:
        print('12.09.'+str(year))
    else:
        print('11.09.'+str(year))

''' teams cf '''
l = 0
for _ in int(input()):
    l += 1 if sum([int(x) for x in input().split()]) >=2 else 0
print(l)

''' scores cf '''
n,k = [int(x) for x in input().split()]
scores = [int(x) for x in input().split()]
print(
len([ 1 for score in scores if score>0 and score in list(set(scores))[::-1][:k-1] ] )
)
list(set(scores))[::-1][:k-1]

f = lambda : map(int,input().split())
n,k =f()
scores = list(f())
print( sum(i>=max(1,scores[k-1]) for i in scores) )

''' maximum no. of dominoes to place '''
m,n = map(int,input().split())
m,n = max(m,n),min(m,n)
tot = 0
mby2 = m%2 == 0; nby2 = n%2 == 0
if True in [mby2, not mby2 and nby2]:
    tot = m*n/2
else:
    tot = (m-1)*(n-1)/2 + (m-1)/2 + (n-1)/2

''' bit++ cf '''
x = 0
for _ in range(int(input())):
    stmt = input();
    x = x+1 if '++' in stmt else x-1
print(x)
'''petya and strings cf'''
s1 = input(); s2 = input()
snums = lambda s: list(map(ord,list(s.lower())) )
for c1,c2 in zip(snums(s1),snums(s2)):
    if c1<c2:
        print(-1)
        break
    elif c1>c2:
        print(1)
        break
else:
    print(0)

''' beautiful matrix '''
for row in range(5):
    col = [int(x) for x in input().split()]
    ind = -1 if 1 not in col else col.index(1)
    if ind!=-1:
        moves = abs(2-row) + abs(2 - ind)
print(moves)

''' xenia learning the addition cf '''
print('+'.join(sorted([x for x in input().split('+')])))

''' capitalization cf'''
s = input()
print( s[0].upper() + s[1:] )

'''stones on the table cf'''
n,s = input(),input()
x = s[0]
for stone in s:
    if x[-1]!=stone:
        x += stone
print(len(s) - len(x))

''' chat cf'''
if len(set(input()))%2:
    print('IGNORE HIM!')
else:
    print('CHAT WITH HER!')

''' bananas cf'''
k,n,w = [int(x) for x in input().split()]
print( max(int(w/2*(w+1)*k - n),0 )

''' limak and bob bears cf '''
a,b = [int(x) for x in input().split()]
y = 0
while a<=b:
    a,b,y = 3*a,2*b,y+1
print(y)

''' hostile subtraction '''
n,k = [int(x) for x in input().split()]
for _ in range(k):
    n = n-1 if str(n)[-1]!='0' else n//10
    print(n)


''' elephant wants to visit his friend cf'''
final = int(input()); steps = 5; count = 0
while final>0:
    count += final//steps
    final = final%steps
    steps  = steps - 1
print(count)

''' tram cf '''
capacity,mx,b_prev = 0,0,0
for _ in range(int(input())):
    a,b = [int(x) for x in input().split()]
    capacity+=b_prev-a + b
    if capacity>mx:
        mx = capacity
    b_prev = b
print(mx)

''' tram cf '''
cap, mx_cap,bf_entry = 0,0,0
for stop in range(int(input())):
    exit, entry = [int(x) for x in input().split()]
    cap = entry + bf_entry - exit
    mx_cap = cap if cap>mx_cap else mx_cap
    bf_entry = cap
print(mx_cap)

''' replace with letters cf '''
s = input()
condition = sum([c.islower() for c in s]) - sum([c.isupper() for c in s])
if condition >= 0:
    print(s.lower())
else:
    print(s.upper())

l=[2,1,0,1,2]
for i in l:
 s=input()
 if "1" in s:print(i+l[s.find("1")//2])

''' lucky numbers '''
print( 'YES' if len([x for x in input() if x in ['4','7']] ) in [4,7] else 'NO' )
print('NYOE S'[sum(i in'47'for i in input())in(4,7)::2])

''' interchange girls and boys '''
n,t = [int(x) for x in input().split()]
q = list(input())
for _ in range(t):
    i = 0
    while i<len(q) - 1:
        if q[i] == 'B' and q[i+1] == 'G':
            q[i],q[i+1] = 'G','B'
            print(q)
        i+=1
print(''.join(q))

_,k = [int(x) for x in input().split()]
q = input()
for _ in range(k+1):
    q.replace('BG','GB')
print(q)

''' translation cf '''
print( 'YES' if input()[::-1] == input() else 'NO')
print('YNEOS'[input()!=input()[::-1]::2])

''' anton and danik cf'''
input()
s = input()
count = {c:s.count(c) for c in {'A','D'}}
if count['A']>count['D']:
    print('Anton')
elif count['A'] == count['D']:
    print('Friendship')
else:
    print('Danik')

######### IMPORTANT #########
d=int(input())-input().count('A')*2;print([['Friendship','Danik'][d>0],'Anton'][d<0])
s=input();print([s.lower(),s.upper()][sum(x<'['for x in s)*2>len(s)])
print('YNEOS'[input()!=input()[::-1]::2])
print('NYOE S'[sum(i in'47'for i in input())in(4,7)::2])
#############################

''' year with distinct digits cf '''

year = int(input()) + 1
while True:
    if len(set(str(year))) == 4:
        break
    else:
        year = year + 1
print(year)

year = int(input()) + 1
while len(set(str(year)))<4:
    year+=1
print(year)

''' rooms cf '''
count = 0
for _ in range(int(input())):
    p,q = [int(x) for x in input().split()]
    if p<=q-2:
        count +=1
print(count)
print(sum( eval( input().replace(' ','-'))<-1 for _ in ' '*int(input()) ))

''' fence cf '''
n,h = [int(x) for x in input().split()]
print( sum((int(c) > h)+1 for c in input().split()))

''' gifts cf '''
n = input()
g = {int(x):i+1 for i,x in enumerate(input().split())}
[g[elem] for elem in sorted(g.values()) ]
print(' '.join([str(elem) for elem in x]))
n=int(input());s=input().split();print(*[s.index(str(i+1))+1 for i in range(n)])

''' magnets '''
last = ''; gr = 0
for _ in range(int(input())):
    s = input()
    if s == last:
        continue
    gr += 1
    last = s
print(gr)

f = 0; p = ''
for _ in range(int(input())):
    s = input(); f+=p!=s; p=s
print(f)

''' In search of an easy problem cf ''' 'Excellent'
input()
print(['Easy','Hard'][max([int(x) for x in input().split()])])

''' Banner's deepest feelings '''
s = ['hate']
for _ in range(1,int(input())):
    if s[-1]=='hate':
        s.append('love')
    else:
        s.append('hate')

print('I ' + ' that I '.join(s) + ' it')

s = ['hate']
for _ in range(int(input())-1):
    s.append(['hate','love'][s[-1]=='hate'])
print('I ' + ' that I '.join(s) + ' it')

print(*(['I hate','I love']*50)[:int(input())],sep=' that ',end=' it')

''' sum of series cf'''
print( sum( [-i if i%2 else i for i in range(1,int(input())+1)] ))
'bloody shit, dont forget your math'
n = int(input())
print([n//2,-(1+n)//2][n%2])

''' shapur cf'''
print( ''.join([str(int(x1!=x2)) for x1,x2 in zip(list(input()),list(input()) )]) )

''' orange juice cf'''
n =int(input())
print('%.6f'%(sum([int(x)/100 for x in input().split()])/n*100))

''' all the levels cf '''
f = lambda: [int(x) for x in input().split()]
n = int(input())
s = set(f()[1:]).union(set(f()[1:]))
print( ['I become the guy.','Oh, my keyboard!'][any(x not in s for x in range(1,n+1))] )
a=input;print("IO hb,e cmoym ek etyhbeo agrudy!."[int(a())>len(set(a().split()[1:]+a().split()[1:]))::2])

''' a dance with dragons dance cf ''' ' a good problem'

''' easy problem cf'''
 print(4 - len(set([int(x) for x in input().split()])) )

 ''' heights cf'''
input()
heights = set([int(x) for x in input().split()])
print(heights.index(max(heights)) + len(heights) - 1 - heights.index(min(heights)))

 ''' input as a set directly cf'''
import string
print(len(set([x for x in input() if x in string.ascii_lowercase])))
print(len(set(input()+', '))-4)

''' make a divisible by b cf'''
for _ in range(int(input())):
    a,b = list(map(int,input().split()))
    if a<b:
        print(b-a)
        continue
    print([(a//b+1)*b - a,0][not (a%b)])

''' home and away team uniforms cf'''
home,away = [],[]
for _ in range(int(input())):
    h,a = [int(x) for x in input().split()]
    home.append(h); away.append(a)
print(sum(i==j for i in home for j in away))

 _=input
x,y=zip(*[_().split()for i in'_'*int(_())])
print(sum(x.count(i)for i in y))

''' pangrams cf '''
input()
print('YNEOS'[len(set(input().lower()))<26::2])

''' polyhedra cf '''
poly = {'tetrahedron':4,
        'cube':6,
        'octahedron':8,
        'dodecahedron':12,
        'icosahedron':20
        }
print(sum([poly[input().lower()] for _ in ' '*int(input())]))

''' distribute candies cf '''
print( '\n'.join([str((int(input())-1)//2) for _ in ' '*int(input()) ]))

''' New year and christmas men cf'''
n1 = input()
n2 = input()
jum = input()
d = lambda n:{x:n.count(x) for x in n}
print(('YNEOS'[d(jum)==d(n1).update(d(n2))::2]).strip())

print('YNEOS'[sorted(n1+n2) == sorted(jum)::2])

''' Hit the lottery cf '''
notes = [100,20,10,5,1]
count = 0
amt = int(input())
for note in notes:
    count+=amt//note
    amt = amt%note
print(count)

''' draw the snake '''
n,m = [int(x) for x in input().split()]
for i in range(1,n+1):
    if i%2!=0:
        print('#'*m)
    elif i%4 == 0:
        print('#' + '.'*(m-1))
    else:
        print('.'*(m-1)+'#')

''' Vasya the hipster '''
r,b = [int(x) for x in input().split()]
r,b = min(r,b),max(r,b)
print(r,(b-r)//2)

''' sum of round numbers cf '''
for _ in range(int(input())):
    num = [int(x) for x in input()]
    exp_for = filter(lambda x:x!=0, [num[i]*(10**(len(num[i:])-1)) for i in range(len(num)) ])
    print(len(list(exp_for)))
    print(' '.join([str(x) for x in exp_for]))

for s in[*open(0)][1:]:
    a=[c+i*'0'for i,c in enumerate(s[-2::-1]) if'0'<c ]
    print(len(a),*a)

for s in [*open(0)][1:]:
    a = [c+i*'0' for i,c in enumerate(s[-2::-1]) if c>'0']
    print(len(a),*a)

''' sum of composite numbers '''
num = int(input())
is_comp = lambda x: sum(x%i==0 for i in range(2,x//2+1)) != 0
a = num//2
while not is_comp(a) or not is_comp(num-a):
    a = a-1
print(a,num-a)

_=int(input());print(Z:=8+_%2,_-Z)

''' times cf '''
n,k = [int(x) for x in input().split()]; x = 1
while 5*x*(x+1)//2 <= 240 - k and x<n:
    x+=1
print(x)

print(sum(5*i*(i+1)//2 <= 240 - k for i in range(1,n+1)))

''' remove smallest '''
for _ in range(int(input())):
    input()
    arr = sorted([int(x) for x in input().split()])

for s in[*open(0)][2::2]:
    a={*map(int,s.split())}
    print('YNEOS'[max(a)-min(a)>=len(a)::2])

''' coder performance cf'''
input()
mx,mn = -1, 1000
cnt = 0
for score in [int(x) for x in input().split()]:
    if score>max:
        mx = scores
        cnt+=1
    if score<min:
        mn = score
        cnt+=1
print(cnt)

n = int(input())
scores = [int(x) for x in input().split()]
print(
sum([scores[i]<min(scores[:i]) or scores[i]>max(scores[:i]) for i in range(1,n)])
)

''' yet another 2 integer problem cf'''
for _ in range(int(input())):
    a,b = [int(x) for x in input().split()]
    ans = abs(b-a); k = 10; count = 0
    while ans>0 and k>=1:
        count+= ans//k
        ans = ans%k
        k-=1
    print(count)

for s in[*open(0)][1:]:
    a,b=map(int,s.split());
    print(0--abs(a-b)//10)

''' guessing the numbers cf '''
l = sorted([int(x) for x in input().split()])
print(l[3] - l[2],l[3] - l[1],l[3] - l[0])

''' buy a shovel cf'''
k,r = [int(x) for x in input().split()]
n = 1
while not (int(str(k*n)[-1]) == r or (k*n)%10 == 0 ) :
    n+=1
print(n)

''' short substring cf'''
for _ in range(int(input())):
    s = input()
    #s = s[0] +s+ s[-1]
    #print(s[::2])
    print((s[0]+s+s[-1])[::2])

''' police recruits cf '''
n = int(input())
r = [int(x) for x in input().split()]
hire = 0;crimes = 0
for c in r:
    if c>0:
        hire+=c
    if c<0:
        crimes = [crimes+1,crimes][hire>0]
        hire = max(0,hire - 1)
print(crimes)

''' balanced array cf '''
for _ in range(int(input())):
    n = int(input())
    arr = list(set(range(1,n+1)))
    if len(arr)<=1:
        print('NO')
        continue
    odd,even = arr[0::2],arr[1::2]
    even[-1] = sum(odd) - sum(even[:-1])
    while len(set(even)) != n//2:
        odd[-1]+=2
        even[-1] = sum(odd) - sum(even[:-1])
    print('YES')
    print(*odd,*even)

''' 3 friends cf ''' # even this was hard, dont know why i did not get it in the first try itself
arr = [int(x) for x in input().split()]
print(sum(abs(x - sorted(arr)[1]) for x in arr))

''' Holiday of Equality cf '''
input()
arr = [int(x) for x in input().split()]
print(sum(abs(max(arr) - x) for x in arr))

''' remove smallest cf '''
for _ in range(int(input())):
    n = int(input())
    arr = [int(x) for x in input().split()]
    arr1 = [c for c in arr if (c+1) in arr]
    print('YNEOS'[len(set(arr)) != len(set(arr1))])

''' choosing teams cf '''
n,k = [int(x) for x in input().split()]
y = [int(x) for x in input().split() if int(x)+k<=5]
print(len(y)//3)

P=lambda:map(int,input().split())
_,k=P()
print(sum(x<=5-k for x in P())//3)

''' Two arrays and two swaps cf '''
P = lambda : list(map(int,input().split()))
for _ in range(int(input())):
    n,k = P()
    a,b = sorted(P()),sorted(P())[::-1]
    print(sum([[a[i],b[i]][i<k and a[i]<b[i]] for i in range(n)]))

''' Required remainder cf'''
for _ in range(int(input())):
    req = -1
    x,y,n = [int(x) for x in input().split()]
    for k in range(n,-1,-1):
         if k%x==y:
             req = k
             break
    print(req)

''' minimal square cf '''
for _ in range(int(input())):
    a,b = [int(x) for x in input().split()]
    a,b = max(a,b),min(a,b)
    print(max(a,2*b)**2)

for _ in range(int(input())):
    n = int(input())
    a = [int(x) for x in input().split()]
    b = [int(x) for x in input().split()]
    print(sum(max(i - min(a),j-min(b)) for i,j in zip(a,b)))

''' team building cf '''
n = input()
arr = [int(x) for x in input().split()]
cnt = min([arr.count(i) for i in (1,2,3)])
print(cnt)
for i,x in enumerate([1,2,3]*cnt):
    print(arr.index(x)+1,end = ' ')
    arr[arr.index(x)] = 100
    if (i+1)%3==0:
        print()
#best lookign sol:
input();a=[[],[],[]]
for i,x in enumerate(input().split(),1):a[int(x)-1]+=[i]
print(min(map(len,a)))
for x in zip(*a):print(*x)

''' even array cf '''
for _ in range(int(input())):
    input()
    arr = [int(x) for x in input().split()]
    odd = [x for i,x in enumerate(arr) if i%2!=x%2]
    print([[len(odd)//2,-1][0 not in [x%2 for x in odd] and odd != []],-1][len(odd)%2])

''' mishka and game cf '''
mishka = 0; chris = 0
for _ in range(int(input())):
    a,b = [int(x) for x in input().split()]
    mishka+=a>b; chris+=b>a;
print(['Friendship is magic!^^',['Mishka','Chris'][mishka<chris]][mishka!=chris])

''' Honest coach cf'''
for _ in range(int(input())):
    n = int(input())
    sth = sorted([int(x) for x in input().split()])
    min = 1000
    for a,b in zip(sth[:-1],sth[1:]):
        if abs(a-b)<min:
            min = abs(a-b)
    print(min)

''' 2000 lines of code reached here '''
