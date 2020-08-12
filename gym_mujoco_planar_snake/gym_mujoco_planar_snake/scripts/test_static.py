class Test:

    def __init__(self):
        self.var = 10



    def method(self):
        return self.var


print(Test.method(self))