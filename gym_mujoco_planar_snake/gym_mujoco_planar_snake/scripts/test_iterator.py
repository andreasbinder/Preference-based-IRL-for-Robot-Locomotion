

def test():
    for i in range(10):
        yield i

t = test()
print(t.__next__())
print(t.__next__())
print(t.__next__())
print(next(t))