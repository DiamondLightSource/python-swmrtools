from .datasource import SliceDict
import blosc
import numpy as np
import time
from . import utils

import logging

logger = logging.getLogger(__name__)


class ChunkSource:
    def __init__(self, datasets, timeout=10, finished_dataset=None):
        self._check_datasets(datasets.values())
        self._datasets = datasets
        self.finished_dataset = finished_dataset
        self.timeout = timeout
        self.finished_set = False
        self.max_size = None
        for ds in self._datasets.values():
            ms = ds.maxshape[0]

            if ms is not None and (self.max_size is None or self.max_size > ms):
                self.max_size = ms

        self.chunk_size = list(self._datasets.values())[0].chunks[0]

        self.current_index = 0

    def _check_datasets(self, datasets):
        for d in datasets:
            s = d.shape
            c = d.chunks
            # ignore first dimension
            for i in range(1, len(s)):
                if s[i] != c[i]:
                    raise RuntimeError(f"Chunk {c} and shape {s} not compatible")

    def _read_datasets(self, current_index, datasets, output):
        for n, d in datasets.items():
            s = list(d.chunks)
            s[0] = -1

            coffset = [0] * len(d.shape)

            flat_index = current_index * self.chunk_size
            coffset[0] = flat_index

            prop_dcid = d.id.get_create_plist()
            use_blosc = False
            if prop_dcid.get_nfilters() == 1 and prop_dcid.get_filter(0)[0] == 32001:
                use_blosc = True
            elif prop_dcid.get_nfilters() == 0:
                pass
            else:
                raise RuntimeError(
                    "Dataset filters not supported for direct chunk read"
                )

            # since we have checked the index and shape this should always work...
            chunk = d.id.read_direct_chunk(coffset)

            ds = self._chunk2numpy(chunk[1], d.dtype, s, use_blosc)

            if self.max_size < (current_index * self.chunk_size + self.chunk_size):
                s[0] = self.max_size - current_index * self.chunk_size
                if ds.shape != s:
                    slices = [slice(0, None)] * len(ds.shape)
                    slices[0] = slice(0, s[0])
                    ds = ds[tuple(slices)]

            output[n] = ds

    def _chunk2numpy(self, blob, dtype, shape, use_blosc):
        if use_blosc:
            blob = blosc.decompress(blob)
        npa = np.frombuffer(blob, dtype=dtype, count=-1)
        return npa.reshape(shape)

    def _check_index(self, datasets, current_index):
        for n, d in datasets.items():
            s = list(d.chunks)
            s[0] = -1

            coffset = [0] * len(d.shape)

            flat_index = current_index * self.chunk_size
            coffset[0] = flat_index

            si = d.id.get_chunk_info_by_coord(tuple(coffset))

            # if offset is None chunk is not written
            if si.byte_offset is None:
                return False

            # if shape is less than (or equal) to current index
            # chunk is flushed (offset not None), but metadata not updated
            if d.shape[0] <= flat_index:
                return False

        return True

    def __iter__(self):
        return self

    def __next__(self):
        if self._check_index(self._datasets, self.current_index):
            return self._generate_output()

        start_time = time.time()
        while self.timeout > (time.time() - start_time):
            time.sleep(self.timeout / 20.0)
            self._check_finished_dataset()

            for ds in self._datasets.values():
                utils.refresh_dataset(ds)

            if self._check_index(self._datasets, self.current_index):
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

        self._read_datasets(self.current_index, self._datasets, output)

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
