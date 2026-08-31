class Tracker:
    def __init__(self):
        self.default_store = None
        self.best_store = None

    def set_best_store(self, store: dict):
        self.best_store = store

    def get_best_store(self):
        return self.best_store

    def set_default_store(self, store: dict):
        self.default_store = store

    def get_default_store(self):
        return self.default_store