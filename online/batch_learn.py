from river import base


class BatchLearn:
    def __init__(self):
        self.curBatch = []

    def learn(self, X):
        self.curBatch = X

    def get_summary(self):
        return self.curBatch
