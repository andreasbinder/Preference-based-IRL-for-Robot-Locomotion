class item:
    def __init__(self):
        variable = 50


l = [4] * 10

def yieldd():
    for i in l:
        yield(i)

print(list(yieldd()))