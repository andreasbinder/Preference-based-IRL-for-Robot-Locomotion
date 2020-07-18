def test(**args):
    print(args)
    print(type(args))

#test(a="Mike", b="Francis", c="Claire")

def tuple_test():
    data = [[],[]]
    for i in range(3):
        data[0].append(i)
        data[1].append(i)

    return tuple(data)

print(tuple_test())
