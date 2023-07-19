from .datasource import SliceDict
import blosc
import numpy as np
import time
from . import utils

import logging

logger = logging.getLogger(__name__)


class ChunkSource:
    def __init__(self, datasets, timeout=10, finished_dataset=None):
        self.check_datasets(datasets.values())
        self._datasets = datasets
        self.finished_dataset = finished_dataset
        self.timeout = timeout
        self.finished_set = False
        self.max_size = min([ds.maxshape[0] for ds in self._datasets.values()])
        self.min_n_chunks = min(
            [ds.id.get_num_chunks() for ds in self._datasets.values()]
        )

        self.chunk_size = list(self._datasets.values())[0].chunks[0]

        self.current_index = 0

    def check_datasets(self, datasets):
        for d in datasets:
            s = d.shape
            c = d.chunks
            # ignore first dimension
            for i in range(1, len(s)):
                if s[i] != c[i]:
                    raise RuntimeError(f"Chunk {c} and shape {s} not compatible")

    def read_datasets(self, current_index, datasets, output):
        for n, d in datasets.items():
            s = list(d.chunks)
            s[0] = -1

            coffset = [0] * len(d.shape)

            flat_index = current_index * self.chunk_size
            coffset[0] = flat_index
            chunk = d.id.read_direct_chunk(coffset)
            ds = self.chunk2numpy(chunk[1], d.dtype, s)

            if self.max_size < (current_index * self.chunk_size + self.chunk_size):
                s[0] = self.max_size - current_index * self.chunk_size
                if ds.shape != s:
                    slices = [slice(0, None)] * len(ds.shape)
                    slices[0] = slice(0, s[0])
                    ds = ds[tuple(slices)]

            output[n] = ds

    def chunk2numpy(self, blob, dtype, shape):
        decom = blosc.decompress(blob)
        npa = np.frombuffer(decom, dtype=dtype, count=-1)
        return npa.reshape(shape)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index < self.min_n_chunks:
            return self._generate_output()
        else:
            start_time = time.time()
            while self.timeout > (time.time() - start_time):
                time.sleep(self.timeout / 20.0)
                self._check_finished_dataset()

                for ds in self._datasets.values():
                    utils.refresh_dataset(ds)
                tmp_min = min(
                    [ds.id.get_num_chunks() for ds in self._datasets.values()]
                )
                if tmp_min > self.min_n_chunks:
                    self.min_n_chunks = tmp_min
                    return self._generate_output()

                if self.finished_set:
                    raise StopIteration

            raise StopIteration

    def _generate_output(self):
        output = SliceDict()
        output.index = self.current_index * self.chunk_size
        output.slice_metadata = [
            slice(
                self.current_index * self.chunk_size,
                self.current_index + 1 * self.chunk_size,
            )
        ]
        output.maxshape = [self.max_size]

        self.read_datasets(self.current_index, self._datasets, output)

        self.current_index += 1

        return output

    def _check_finished_dataset(self):
        if self.finished_dataset is None:
            return

        utils.refresh_dataset(self.finished_dataset)

        if self.finished_dataset.size != 1:
            logger.warning(
                f"finished dataset ({self.finished_dataset}) is non-singular"
            )
            return

        # set on a attribute so the timeout loop runs once after
        # finished is set.
        # this is important due to race conditions between the final
        # keys being readable and the finished flag being set
        self.finished_set = not self.finished_dataset[0] == 0

        if self.finished_set:
            logger.debug("Finish flag set after finished dataset check")
