class A:
    def __init__(self):
        pass
    def b(self):
        print('a')

class B(A):
    def __init__(self):
        super(B, self).__init__()
    def b(self):
        print('b')
        super(B, self).b()

B().b()