old = []
true = []

for l in open('test3-pos/test3.pos'):
    old.append(l)

for l in open('pos/test'):
    true.append(l)

print len(old)
print len(true)
f = open('test3-pos/1/test3.pos', 'w')

for i in range(len(old)):
    if true[i] == '\n':
        f.write("\n")
        continue
    f.write("{0} {1}\n".format(true[i].strip(), old[i].split()[1].strip()))