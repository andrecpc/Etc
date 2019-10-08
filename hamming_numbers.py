def hamming(n):
    numb = 1
    all_numbers = set([numb])
    cash = dict()
    for i in range(n):
        all_numbers |= set([numb*2,numb*3,numb*5])
        res = min(all_numbers)
        cash[i+1] = res
        numb = res
        all_numbers.remove(res)
    return cash[n]
    
for i in range(5000):
    print (hamming(i+1))
