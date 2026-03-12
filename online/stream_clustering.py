class StreamClusterer:
    def __init__(self):
        pass

    """
        Updates Stream Clusterer with a single instance at a certain timestamp
        
        Parameters
        ----
        x: array-like, instance inserted
        t: int, timestamp when instance is inserted, optional, increment by 1 if None is provided
        
        Returns
        ----
        # assignment id (MC Id or equivalent)
    """
    def learn_one(self, x, t: int) -> int:
        pass

    """
        Updates Stream Clusterer with multiple instances each at a certain timestamp

        Parameters
        ----
        xs: list of array likes, instances inserted
        ts: list of ints, timestamps when instance is inserted, optional, increment by 1 if None is provided

        Returns
        ----
        # assignment id (MC Id or equivalent)
    """
    def learn(self, xs, ts) -> list:
        pass
