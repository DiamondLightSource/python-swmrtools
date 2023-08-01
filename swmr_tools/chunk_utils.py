import numpy as np
from . import utils
import math


class SliceInOut:
    def __init__(self, input, output):
        self.input = input
        self.output = output


class SliceSize:
    def __init__(self, slice, size):
        self.slice = slice
        self.size = size


class ChunkSliceCollection:
    types = ("current", "last", "combined")

    def __init__(self, type, position):
        if type not in ChunkSliceCollection.types:
            raise RuntimeError(f"{type} not in {ChunkSliceCollection.types}")

        self.type = type
        self.current = None
        self.last = None
        self.intermediate = None
        self.position = position

    def __repr__(self):
        out = f"Type {self.type}, position {self.position}"

        if self.last:
            out += (
                " Last input: "
                + str(self.last.input)
                + " Output: "
                + str(self.last.output)
            )

        if self.current:
            out += (
                " Current input: "
                + str(self.current.input)
                + " Output: "
                + str(self.current.output)
            )

        if self.intermediate:
            out += (
                " Intermed: "
                + str(self.intermediate.slice)
                + " size: "
                + str(self.intermediate.size)
            )

        return out


def _build_collection(
    pos, in_start, in_stop, in_step, out_start, out_stop, out_step, type="current"
):
    input = slice(in_start, in_stop, in_step)
    output = slice(out_start, out_stop, out_step)
    current = SliceInOut(input, output)
    cc = ChunkSliceCollection(type, pos)
    if type == "current":
        cc.current = current
    else:
        cc.last = current
    return cc


def write_data(slice_structure, data, last_data, output):
    for s in slice_structure:
        spos = s.position

        output_slice = [slice(0, None)] * len(output.shape)

        for i in range(len(spos)):
            output_slice[i] = slice(spos[i], spos[i] + 1)

        if s.type == "current":
            si = s.current.input
            so = s.current.output

            input_slice = [slice(0, 1)] * len(data.shape)
            input_slice[0] = si
            flush_data = data[tuple(input_slice)]
        elif s.type == "last":
            si = s.last.input
            so = s.last.output
            input_slice = [slice(0, 1)] * len(last_data.shape)
            input_slice[0] = si

            flush_data = last_data[tuple(input_slice)]
        else:
            # Combined
            inter_shape = list(data.shape)
            inter_shape[0] = s.intermediate.size
            intermediate = np.zeros(inter_shape, dtype=data.dtype)

            ls = s.last.input
            li = s.last.output
            last_slice = [slice(0, 1)] * len(last_data.shape)
            last_slice[0] = ls
            intermediate[li] = last_data[tuple(last_slice)]

            cs = s.current.input
            ci = s.current.output
            input_slice = [slice(0, 1)] * len(data.shape)
            input_slice[0] = cs
            intermediate[ci] = data[tuple(input_slice)]

            so = s.intermediate.slice

            flush_data = intermediate

        rank = len(data.shape)
        output_slice[-1 * rank] = so

        output[tuple(output_slice)] = flush_data


def _raster_routine(index, poff, spos, n_points_chunk, scan_shape):
    remaining = n_points_chunk - poff
    last_chunk_width = scan_shape[-1] % n_points_chunk
    ss = []
    if poff == 0:
        remaining = 0

    start_write = index - remaining
    start = utils.get_position(start_write, scan_shape, len(scan_shape))

    if start[-1] + last_chunk_width == scan_shape[-1]:
        end_chunk = True
        end_write_offset = last_chunk_width
    else:
        end_chunk = False
        end_write_offset = n_points_chunk

    end = list(start)
    end[-1] = end[-1] + end_write_offset
    if poff == 0:
        cc = _build_collection(
            spos, 0, end_write_offset, 1, start[-1], end[-1], 1, type="current"
        )
        ss.append(cc)
    else:
        last_input_start = poff
        last_input_stop = n_points_chunk
        last_output_start = 0
        last_output_stop = remaining

        if end_chunk:
            current_input_stop = last_chunk_width - remaining
        else:
            current_input_stop = poff

        current_input_start = 0
        current_output_start = remaining
        current_output_stop = n_points_chunk

        last = SliceInOut(
            slice(last_input_start, last_input_stop),
            slice(last_output_start, last_output_stop),
        )
        current = SliceInOut(
            slice(current_input_start, current_input_stop),
            slice(current_output_start, current_output_stop),
        )
        cc = ChunkSliceCollection("combined", spos)
        cc.current = current
        cc.last = last
        intermediate = SliceSize(
            slice(start[-1], start[-1] + end_write_offset), end_write_offset
        )
        cc.intermediate = intermediate
        ss.append(cc)

    if remaining >= last_chunk_width and end[-1] + last_chunk_width == scan_shape[-1]:
        new_start = list(end)
        new_end = list(new_start)
        new_end[-1] = new_end[-1] + last_chunk_width
        cc = _build_collection(
            spos,
            poff,
            poff + last_chunk_width,
            1,
            new_start[-1],
            new_end[-1],
            1,
            type="current",
        )
        ss.append(cc)

    return ss


def _snake_routine(poff, spos, n_points_chunk, scan_shape):
    ss = []
    remaining = n_points_chunk - poff
    last_chunk_width = scan_shape[-1] % n_points_chunk

    distance_from_end = scan_shape[-1] - spos[-1]

    if distance_from_end < remaining + n_points_chunk:
        if poff == 0 and last_chunk_width != 0 and distance_from_end == 1:
            # fill end from current chunk
            cc = _build_collection(
                spos,
                last_chunk_width,
                0,
                -1,
                scan_shape[-1] - last_chunk_width,
                scan_shape[-1],
                1,
                type="current",
            )
            ss.append(cc)
            # can do no more
            return ss

        elif remaining >= last_chunk_width and last_chunk_width != 0:
            start_input = poff + last_chunk_width - 1

            if poff == 0:
                stop_input = None
            else:
                stop_input = poff - 1

            cc = _build_collection(
                spos,
                start_input,
                stop_input,
                -1,
                scan_shape[-1] - last_chunk_width,
                scan_shape[-1],
                1,
                type="last",
            )
            ss.append(cc)

        elif last_chunk_width != 0:
            # combined end fill
            last_input_start = poff
            last_input_stop = n_points_chunk
            last_output_start = last_chunk_width - 1
            last_output_stop = last_chunk_width - remaining - 1

            current_input_start = 0
            current_input_stop = last_chunk_width - remaining
            current_output_start = last_chunk_width - remaining - 1
            current_output_stop = None

            last = SliceInOut(
                slice(last_input_start, last_input_stop),
                slice(last_output_start, last_output_stop, -1),
            )

            current = SliceInOut(
                slice(current_input_start, current_input_stop),
                slice(current_output_start, current_output_stop, -1),
            )

            cc = ChunkSliceCollection("combined", spos)
            cc.current = current
            cc.last = last
            intermediate = SliceSize(
                slice(scan_shape[-1] - last_chunk_width, scan_shape[-1], 1),
                last_chunk_width,
            )
            cc.intermediate = intermediate
            ss.append(cc)
            return ss

    if last_chunk_width == 0:
        # chunks align
        poff = n_points_chunk
    elif remaining < last_chunk_width:
        poff = last_chunk_width - remaining
    else:
        poff = poff + last_chunk_width

    remaining = n_points_chunk - poff
    if remaining == 0:
        start = spos[-1] - n_points_chunk + 1
        # if stop == -1:
        #     stop = None

        cc = _build_collection(
            spos, n_points_chunk, None, -1, start, spos[-1] + 1, 1, type="current"
        )
        ss.append(cc)
    else:
        # can we write a combined?
        last_input_start = poff
        last_input_stop = n_points_chunk
        last_output_start = n_points_chunk
        last_output_stop = poff - 1

        current_input_start = 0
        current_input_stop = n_points_chunk - remaining
        current_output_start = poff - 1
        current_output_stop = None

        start = spos[-1] - poff + 1

        last = SliceInOut(
            slice(last_input_start, last_input_stop),
            slice(last_output_start, last_output_stop, -1),
        )

        current = SliceInOut(
            slice(current_input_start, current_input_stop),
            slice(current_output_start, current_output_stop, -1),
        )

        cc = ChunkSliceCollection("combined", spos)
        cc.current = current
        cc.last = last
        intermediate = SliceSize(
            slice(start, spos[-1] + remaining + 1, 1), n_points_chunk
        )
        cc.intermediate = intermediate
        ss.append(cc)

    return ss


def _build_small_raster_complete(pos, i, scan_shape):
    in_start = i * scan_shape[-1]
    in_stop = scan_shape[-1] * (i + 1)
    in_step = None
    out_start = pos[-1]
    out_stop = pos[-1] + scan_shape[-1]
    out_step = None
    return _build_collection(
        pos, in_start, in_stop, in_step, out_start, out_stop, out_step
    )


def _build_small_snake_complete(pos, i, scan_shape):
    if i == 0:
        in_stop = None
    else:
        in_stop = i * (scan_shape[-1]) - 1

    in_start = scan_shape[-1] * (i + 1) - 1
    in_step = -1
    out_start = 0
    out_stop = pos[-1] + scan_shape[-1]
    out_step = 1

    return _build_collection(
        pos, in_start, in_stop, in_step, out_start, out_stop, out_step
    )


def _build_small_raster_offset(pos, i, scan_shape, offset):
    in_start = i * scan_shape[-1] + offset
    in_stop = scan_shape[-1] * (i + 1) + offset
    in_step = None
    out_start = 0
    out_stop = scan_shape[-1]
    out_step = None
    return _build_collection(
        pos, in_start, in_stop, in_step, out_start, out_stop, out_step
    )


def _build_small_snake_offset(pos, i, scan_shape, offset):
    in_stop = i * scan_shape[-1] + offset - 1
    in_start = scan_shape[-1] * (i + 1) + offset - 1
    in_step = -1
    out_stop = scan_shape[-1]
    out_start = 0
    out_step = 1
    return _build_collection(
        pos, in_start, in_stop, in_step, out_start, out_stop, out_step
    )


def _build_small_raster_combined(spos, n_points_chunk, position_offset, scan_shape):
    # deal with partial chunk first
    n_remaining = n_points_chunk - position_offset
    # Two reads contributing to chunk
    current = SliceInOut(
        slice(0, scan_shape[-1] - n_remaining),
        slice(n_remaining, scan_shape[-1]),
    )
    last = SliceInOut(slice(n_points_chunk - n_remaining, None), slice(0, n_remaining))
    cc = ChunkSliceCollection("combined", spos)
    cc.current = current
    cc.last = last
    intermediate = SliceSize(slice(0, scan_shape[-1]), scan_shape[-1])
    cc.intermediate = intermediate
    return cc


def _build_small_snake_combined(spos, n_points_chunk, position_offset, scan_shape):
    # deal with partial chunk first
    n_remaining = n_points_chunk - position_offset
    # Two reads contributing to chunk
    current = SliceInOut(
        slice(scan_shape[-1] - n_remaining - 1, None, -1),
        slice(0, scan_shape[-1] - n_remaining),
    )
    last = SliceInOut(
        slice(n_points_chunk, position_offset - 1, -1),
        slice(scan_shape[-1] - n_remaining, scan_shape[-1]),
    )
    cc = ChunkSliceCollection("combined", spos)
    cc.current = current
    cc.last = last
    intermediate = SliceSize(slice(0, scan_shape[-1], 1), scan_shape[-1])
    cc.intermediate = intermediate
    return cc


def _x_smaller_than_chunk(index, spos, n_points_chunk, scan_shape, snake):
    is_snake_row = False
    if snake:
        spos, is_snake_row = utils.get_position_snake_row(
            index, scan_shape, len(scan_shape)
        )
    else:
        spos = utils.get_position(index, scan_shape, len(scan_shape))

    nrows = index // scan_shape[-1]
    complete = nrows * scan_shape[-1]
    # number of points left over from chunk compared to complete rows
    position_offset = complete % n_points_chunk
    slice_structure = []

    if position_offset == 0 and (spos[-1] == 0 or spos[-1] == scan_shape[-1] - 1):
        # Complete chunk read can be written to smaller chunks
        for i in range(n_points_chunk // scan_shape[-1]):
            if index + (i) * scan_shape[-1] == math.prod(scan_shape):
                break

            if snake:
                tpos, is_snake_row = utils.get_position_snake_row(
                    index + (i) * scan_shape[-1], scan_shape, len(scan_shape)
                )
            else:
                tpos = utils.get_position(
                    index + (i) * scan_shape[-1], scan_shape, len(scan_shape)
                )

            if is_snake_row and snake:
                cc = _build_small_snake_complete(tpos, i, scan_shape)
            else:
                cc = _build_small_raster_complete(tpos, i, scan_shape)
            slice_structure.append(cc)
            is_snake_row = not is_snake_row

        return slice_structure

    else:
        # deal with partial chunk first
        n_remaining = n_points_chunk - position_offset

        if snake and is_snake_row:
            cc = _build_small_snake_combined(
                spos, n_points_chunk, position_offset, scan_shape
            )
        else:
            cc = _build_small_raster_combined(
                spos, n_points_chunk, position_offset, scan_shape
            )

        slice_structure.append(cc)
        is_snake_row = not is_snake_row
        old_n_remaining = n_points_chunk - position_offset
        offset = scan_shape[-1] - n_remaining

        for i in range(0, (n_points_chunk - offset) // scan_shape[-1]):
            updated_index = index + (i + 1) * scan_shape[-1] - old_n_remaining

            if updated_index == math.prod(scan_shape):
                break

            if snake:
                tpos, is_snake_row = utils.get_position_snake_row(
                    updated_index, scan_shape, len(scan_shape)
                )
            else:
                tpos = utils.get_position(updated_index, scan_shape, len(scan_shape))

            if snake and is_snake_row:
                cc = _build_small_snake_offset(tpos, i, scan_shape, offset)
            else:
                cc = _build_small_raster_offset(tpos, i, scan_shape, offset)
            is_snake_row = not is_snake_row
            slice_structure.append(cc)

        return slice_structure


def get_slice_structure(index, n_points_chunk, scan_shape, snake):
    """
    Returns the slice structure for index, where the dataset has chunk size n_points_per_chunk and
    shape scan_shape. The snake flag indicates whether the scan is raster or snake pattern.

    The output slice structure will either chunk the slices to the output:
    (a) as the length of the scan grid row (if the row is smaller than the input data chunk size), or
    (b) as the input chunks size

    So if writing directly to HDF5, the dataset chunking should be set appropriately (probably wont cause error if
    not set correctly, but IO will be compromised). Returned structure copes with the "end of row" chunk not being filled
    by setting the input/output slices accordingly.

        Parameters:
            index (int): Flattened index of scan point
            n_points_chunk (int): number of points in a chunk
            scan_shape (array): Shape of scan to map chunks into
            snake (boolean): Whether the scan is snake or raster pattern

        Returns:
            slice_structure (list): List containing mappings of current and last chunk into dataset of shape scan_shape
    Examples
    --------

    >>> chunk_utils.get_slice_structure(0, 100, [123,456], False)
    [Type current, position (0, 0) Current input: slice(0, 100, 1) Output: slice(0, 100, 1)]

    """

    # Get position of the point in the scan grid that the chunk corresponds to
    # and, if snake, whether that row is running backwards
    is_snake_row = False
    if snake:
        spos, is_snake_row = utils.get_position_snake_row(
            index, scan_shape, len(scan_shape)
        )
    else:
        spos = utils.get_position(index, scan_shape, len(scan_shape))

    # if chunk is bigger than a row of the scan grid we use the row size as the chunk
    # not the input data chunk size, which is a different routine
    if n_points_chunk > scan_shape[-1]:
        return _x_smaller_than_chunk(index, spos, n_points_chunk, scan_shape, snake)

    nrows = index // scan_shape[-1]
    complete = nrows * scan_shape[-1]
    # number of points left over from chunk compared to complete rows
    poff = complete % n_points_chunk

    if is_snake_row:
        # run the snake routine
        return _snake_routine(poff, spos, n_points_chunk, scan_shape)
    else:
        # run the raster routine
        return _raster_routine(index, poff, spos, n_points_chunk, scan_shape)
