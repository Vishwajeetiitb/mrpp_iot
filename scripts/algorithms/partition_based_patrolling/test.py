from tokenize import group


l = [['bot1',1, 3], ['bot2',2, 3], ['bot3',4, 5], ['bot4',6, 5], ['bot5',7, 5], ['bot6',8, 9]]
print(list(map(set, l)))
l = [set(ele) for ele in l]

pool = set(map(frozenset, l))
print(pool)
groups = []
while pool:
    groups.append(set(pool.pop()))
    while True:
        for idx,candidate in enumerate(pool):
            print(idx)
            if groups[-1] & candidate:
                groups[-1] |= candidate
                pool.remove(candidate)
                break
        else:
            break

print(groups)