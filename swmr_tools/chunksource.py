class ChunkSource:
    def __init__(self, datasets):
        self._datasets = datasets
        self.min_n_chunks = min([ds.id.get_num_chunks() for ds in datasets])

    def __iter__(self):
        return self

    def __next__(self):
        pass
