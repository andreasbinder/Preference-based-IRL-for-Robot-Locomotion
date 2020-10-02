def add(x):
    return x + 1

dic = {
    "func" : (add, 5, 10)
}

result = dic["func"]

print(result[0](5))