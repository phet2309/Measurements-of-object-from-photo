s=input('')
tp_set=set()
res=[]
count=0
fin=''
for i in range(len(s)):
    if s[i] not in tp_set:
        tp_set.add(s[i])
        if i!=0:
            # res.append(count)
            fin=fin+str(s[i-1])
            fin=fin+str(count)
        count=1
    else:
        count+=1

fin=fin+s[len(s)-1]+str(count)
print(fin)

        
