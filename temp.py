from collections import deque

temp = deque()
temp.append(1)
temp.append(2)
temp.append(3)
temp.append(4)
temp.append(5)
temp.append(5)
temp.append(6)
temp.append(7)
print(temp)
temp.popleft()
print(temp)
