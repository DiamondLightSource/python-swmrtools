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
            out += " Last input: " + str(self.last.input) + " Output: " + str(self.last.output)


        if self.current:
            out += (
                " Current input: " + str(self.current.input) + " Output: " + str(self.current.output)
            )


        if self.intermediate:
            out += (
                " Intermed: "
                + str(self.intermediate.slice)
                + " size: "
                + str(self.intermediate.size)
            )

        return out


def calc_offset(pos, shape, chunk_size, snake=False):
    # 2D only
    # num = pos[-2] * shape[-1]spos
    # poff = num % chunk_size

    # ND
    num = np.ravel_multi_index(pos[:-1], shape[:-1]) * shape[-1]
    poff = num % chunk_size

    if snake and pos[0] % 2 == 1:
        end_section_size = shape[-1] % chunk_size

        # Last chunk completed last row
        if poff == 0 and pos[-1] + chunk_size >= shape[-1]:
            return poff
        elif poff == 0:
            poff = end_section_size
            return poff

        if pos[-1] < (shape[1] - end_section_size) and not (
            pos[-1] + (chunk_size - poff) + 1 == shape[-1]
        ):
            # Account for end section on snake row
            n_remaining = chunk_size - poff

            if n_remaining >= end_section_size:
                # can fill end section
                poff += end_section_size
            else:
                # update offset if remaining doesnt fill end sectino
                poff = end_section_size - n_remaining

    return poff


def build_collection(pos,in_start,in_stop,in_step, out_start, out_stop, out_step, type = "current"):
        input = slice(in_start, in_stop, in_step)
        output = slice(out_start, out_stop, out_step)
        current = SliceInOut(input, output)
        cc = ChunkSliceCollection(type, pos)
        if type == "current":
            cc.current = current
        else:
            cc.last = current
        return cc



def snake_routine(spos, n_points_chunk, scan_shape):
    position_offset = calc_offset(spos, scan_shape, n_points_chunk, snake=True)

    print(f"Snake position offset {position_offset}")

    slice_structure = []

    end_section_size = scan_shape[-1] % n_points_chunk

    if (
        position_offset == 0
        and (spos[-1] + n_points_chunk > scan_shape[-1])
        and end_section_size != 0
    ):
        # Complete "section" of chunk written (in reverse)

        in_start = end_section_size - 1
        in_stop = None
        in_step = -1
        out_start = scan_shape[-1] - end_section_size
        out_stop = scan_shape[-1]
        out_step = None

        cc = build_collection(spos,in_start,in_stop,in_step, out_start, out_stop, out_step)
        print("clause 1")

        slice_structure.append(cc)
        return slice_structure
    elif position_offset == 0 and not (spos[-1] + 1 - n_points_chunk < 0):
        # Complete chunk written (in reverse)
        in_start = None
        in_stop = None
        in_step = -1
        out_start = spos[-1] - n_points_chunk + 1
        out_stop = spos[-1] + 1
        out_step = None

        cc = build_collection(spos,in_start,in_stop,in_step, out_start, out_stop, out_step)

        slice_structure.append(cc)
        print("clause 2")

        return slice_structure
    else:
        # CASE 1 - backwards row start, enough remaining to fill complete partial end row chunk
        if spos[1] + (n_points_chunk - position_offset) + 1 == scan_shape[-1]:
            end_section_size = scan_shape[-1] % n_points_chunk

            if n_points_chunk - position_offset >= end_section_size:
                # Some of last chunk to complete end section

                in_start = position_offset + end_section_size - 1
                in_stop = position_offset - 1
                in_step = -1
                out_start = scan_shape[1] - end_section_size
                out_stop = None
                out_step = None

                cc = build_collection(spos,in_start,in_stop,in_step, out_start, out_stop, out_step, type = "last")

                slice_structure.append(cc)

                position_offset = position_offset + end_section_size

            else:
                # Combine current and last to fill end
                n_remaining = n_points_chunk - position_offset

                current = SliceInOut(
                    slice(end_section_size - n_remaining - 1, None, -1),
                    slice(None, end_section_size - n_remaining),
                )
                last = SliceInOut(
                    slice(scan_shape[1] - end_section_size, None), end_section_size
                )
                cc.intermediate = intermediate

                slice_structure.append(cc)

                position_offset = end_section_size - n_remaining
                return slice_structure

        if position_offset != 0:
            # Two reads contribute to written chunk
            start = spos[-1] - position_offset + 1
            end = spos[-1] - position_offset + n_points_chunk + 1

            n_remaining = n_points_chunk - position_offset

            current = SliceInOut(
                slice(position_offset - 1, None, -1), slice(None, position_offset)
            )
            last = SliceInOut(
                slice(None, -1 * (n_remaining + 1), -1), slice(position_offset, None)
            )
            cc = ChunkSliceCollection("combined", spos)
            cc.current = current
            cc.last = last
            intermediate = SliceSize(slice(start, end), n_points_chunk)
            cc.intermediate = intermediate

            slice_structure.append(cc)

    return slice_structure


def non_snake_routine(spos, n_points_chunk, scan_shape):
    position_offset = calc_offset(spos, scan_shape, n_points_chunk)

    print(f"Raster position offset {position_offset}")

    slice_structure = []

    if position_offset == 0 and not (spos[-1] + n_points_chunk > scan_shape[-1]):
        # Complete chunk read can be written
        in_start = 0
        in_stop = n_points_chunk
        in_step = None
        out_start = spos[-1]
        out_stop = spos[-1] + n_points_chunk
        out_step = None

        cc = build_collection(spos,in_start,in_stop,in_step, out_start, out_stop, out_step)

        slice_structure.append(cc)

    elif position_offset == 0 and spos[-1] + n_points_chunk > scan_shape[-1]:
        # Complete chunk read, section if start written to end of grid
        position_offset = scan_shape[-1] - spos[-1]

        in_start = 0
        in_stop = position_offset
        in_step = None
        out_start = spos[-1]
        out_stop = spos[-1] + n_points_chunk
        out_step = None

        cc = build_collection(spos,in_start,in_stop,in_step, out_start, out_stop, out_step)

        slice_structure.append(cc)

    else:
        n_remaining = n_points_chunk - position_offset

        nstart = spos[-1] - n_remaining

        if n_remaining + position_offset + nstart > scan_shape[-1]:
            # Two reads contributing to end "short" chunk
            num = scan_shape[-1] - nstart - n_remaining
            position_offset = num

            current = SliceInOut(slice(0, num), slice(n_remaining, None))
            last = SliceInOut(
                slice(n_points_chunk - n_remaining, None), slice(0, n_remaining)
            )
            cc = ChunkSliceCollection("combined", spos)
            cc.current = current
            cc.last = last
            intermediate = SliceSize(
                slice(nstart, nstart + n_remaining + num), n_remaining + num
            )
            cc.intermediate = intermediate

            slice_structure.append(cc)

        else:
            # Two reads contributing to chunk
            current = SliceInOut(
                slice(0, position_offset),
                slice(n_remaining, n_remaining + position_offset),
            )
            last = SliceInOut(
                slice(n_points_chunk - n_remaining, None), slice(0, n_remaining)
            )
            cc = ChunkSliceCollection("combined", spos)
            cc.current = current
            cc.last = last
            intermediate = SliceSize(
                slice(nstart, nstart + n_remaining + position_offset), n_points_chunk
            )
            cc.intermediate = intermediate

            slice_structure.append(cc)

        if (n_remaining + spos[-1] + position_offset) >= scan_shape[-1] and scan_shape[
            -1
        ] != spos[-1] + position_offset:
            # Single chunk contributing to end chunk
            w_end = scan_shape[-1] - spos[-1]

            in_start = position_offset
            in_stop = w_end
            in_step = None
            out_start = spos[-1] + position_offset
            out_stop = scan_shape[-1]
            out_step = None

            cc = build_collection(spos,in_start,in_stop,in_step, out_start, out_stop, out_step)

            slice_structure.append(cc)

    return slice_structure


def non_snake_routine_smallx(spos, n_points_chunk, scan_shape):
    position_offset = calc_offset(spos, scan_shape, n_points_chunk)
    slice_structure = []

    if position_offset == 0 and spos[-1] == 0:
        # Complete chunk read can be written to smaller chunks
        for i in range(n_points_chunk//scan_shape[-1]):
            pos = list(spos)
            pos[0] = i

            in_start = i*scan_shape[-1]
            in_stop = scan_shape[-1]*(i+1)
            in_step = None
            out_start = spos[-1]
            out_stop = spos[-1] + scan_shape[-1]
            out_step = None

            cc = build_collection(pos,in_start,in_stop,in_step, out_start, out_stop, out_step)

            slice_structure.append(cc)
        
        return slice_structure
    
    else:
        #deal with partial chunk first
        total = 2*n_points_chunk - position_offset
        n_remaining = n_points_chunk - position_offset
        # Two reads contributing to chunk
        current = SliceInOut(
            slice(0, scan_shape[-1] - n_remaining),
            slice(n_remaining, scan_shape[-1]),
        )
        last = SliceInOut(
            slice(n_points_chunk - n_remaining, None), slice(0, n_remaining)
        )
        cc = ChunkSliceCollection("combined", spos)
        cc.current = current
        cc.last = last
        intermediate = SliceSize(slice(0, scan_shape[-1]), scan_shape[-1])
        cc.intermediate = intermediate

        slice_structure.append(cc)

        offset = scan_shape[-1] - n_remaining

        for i in range(0, (n_points_chunk-offset)//scan_shape[-1]):
            pos = list(spos)
            pos[0] = i + spos[0] + 1
            pos[1] = 0

            in_start = i*scan_shape[-1]+offset
            in_stop = scan_shape[-1]*(i+1)+offset
            in_step = None
            out_start = 0
            out_stop = scan_shape[-1]
            out_step = None

            cc = build_collection(pos,in_start,in_stop,in_step, out_start, out_stop, out_step)
            
            slice_structure.append(cc)

        return slice_structure

def snake_routine_smallx(spos, n_points_chunk, scan_shape):
    position_offset = calc_offset(spos, scan_shape, n_points_chunk)
    slice_structure = []
    if position_offset == 0 and spos[-1] == 0 and spos[-2] % 2 == 0:

        for i in range(n_points_chunk//scan_shape[-1]):
            pos = list(spos)
            pos[0] = i

            if pos[-2] % 2 == 0:

                in_start = i*scan_shape[-1]
                in_stop = scan_shape[-1]*(i+1)
                in_step = None
                out_start = spos[-1]
                out_stop = spos[-1] + scan_shape[-1]
                out_step = None

                cc = build_collection(pos,in_start,in_stop,in_step, out_start, out_stop, out_step)

                slice_structure.append(cc)
            else:


                in_start = scan_shape[-1]*(i+1)-1
                in_stop = i*scan_shape[-1]-1
                in_step = -1
                out_start = spos[-1]
                out_stop = spos[-1] + scan_shape[-1]
                out_step = None

                cc = build_collection(pos,in_start,in_stop,in_step, out_start, out_stop, out_step)

                slice_structure.append(cc)

        return slice_structure
    else:

        snake_row = spos[-2] % 2 != 0
        #deal with partial chunk first
        total = 2*n_points_chunk - position_offset
        n_remaining = n_points_chunk - position_offset
        # Two reads contributing to chunk
        if not snake_row:
            current = SliceInOut(
                slice(0, scan_shape[-1] - n_remaining),
                slice(n_remaining, scan_shape[-1]),
            )
            last = SliceInOut(
                slice(n_points_chunk - n_remaining, None), slice(0, n_remaining)
            )
            cc = ChunkSliceCollection("combined", spos)
            cc.current = current
            cc.last = last
            intermediate = SliceSize(slice(0, scan_shape[-1]), scan_shape[-1])
            cc.intermediate = intermediate
            slice_structure.append(cc)
            offset = scan_shape[-1] - n_remaining

            for i in range(0, (n_points_chunk-offset)//scan_shape[-1]):
                pos = list(spos)
                pos[0] = i + spos[0] + 1
                pos[1] = 0

                if pos[-2] % 2 != 0:

                    in_start = scan_shape[-1]*(i+1)+offset-1
                    in_stop = i*scan_shape[-1]+offset-1
                    in_step = -1
                    out_start = 0
                    out_stop = scan_shape[-1]
                    out_step = None

                    cc = build_collection(pos,in_start,in_stop,in_step, out_start, out_stop, out_step)
                    
                    slice_structure.append(cc)
                else:

                    in_start = i*scan_shape[-1]+offset
                    in_stop = scan_shape[-1]*(i+1)+offset
                    in_step = None
                    out_start = 0
                    out_stop = scan_shape[-1]
                    out_step = None

                    cc = build_collection(pos,in_start,in_stop,in_step, out_start, out_stop, out_step)

                    slice_structure.append(cc)

        else:
            
            current = SliceInOut(
                slice(scan_shape[-1] - n_remaining-1,None,-1 ),
                slice(0,scan_shape[-1] - n_remaining),
            )
            last = SliceInOut(
                slice(n_points_chunk, n_points_chunk - n_remaining-1, -1), slice(scan_shape[-1]- n_remaining, scan_shape[-1])
            )
            cc = ChunkSliceCollection("combined", spos)
            cc.current = current
            cc.last = last
            intermediate = SliceSize(slice(0, scan_shape[-1]), scan_shape[-1])
            cc.intermediate = intermediate
            slice_structure.append(cc)
            offset = scan_shape[-1] - n_remaining

            for i in range(0, (n_points_chunk-offset)//scan_shape[-1]):
                pos = list(spos)
                pos[0] = i + spos[0] + 1
                pos[1] = 0
                
                if pos[-2] % 2 != 0:

                    in_start = scan_shape[-1]*(i+1)+offset-1
                    in_stop = i*scan_shape[-1]+offset-1
                    in_step = -1
                    out_start = 0
                    out_stop = scan_shape[-1]
                    out_step = None

                    cc = build_collection(pos,in_start,in_stop,in_step, out_start, out_stop, out_step)

                    slice_structure.append(cc)
                else:

                    in_start = i*scan_shape[-1]+offset
                    in_stop = scan_shape[-1]*(i+1)+offset
                    in_step = None
                    out_start = 0
                    out_stop = scan_shape[-1]
                    out_step = None

                    cc = build_collection(pos,in_start,in_stop,in_step, out_start, out_stop, out_step)

                    slice_structure.append(cc)

        return slice_structure



def get_slice_structure(spos, n_points_chunk, scan_shape, snake):

    if n_points_chunk > scan_shape[-1] and not snake:
        return non_snake_routine_smallx(spos, n_points_chunk, scan_shape)
    
    if n_points_chunk > scan_shape[-1] and snake:
        return snake_routine_smallx(spos, n_points_chunk, scan_shape)

    if snake and spos[-2] % 2 == 1:
        return snake_routine(spos, n_points_chunk, scan_shape)
    else:
        return non_snake_routine(spos, n_points_chunk, scan_shape)


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
            # print(f"{s}")
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


def calc_offset(index, shape, chunk_size, snake=False):
    # 2D only
    # num = pos[-2] * shape[-1]spos
    # poff = num % chunk_size

    # ND

    poff = index % chunk_size

    if snake:
         pos = utils.get_position_snake(index,shape,len(shape))
    else:
         pos = utils.get_position(index,shape,len(shape))

    if snake and pos[0] % 2 == 1:
        end_section_size = shape[-1] % chunk_size

        # Last chunk completed last row
        if poff == 0 and pos[-1] + chunk_size >= shape[-1]:
            return poff
        elif poff == 0:
            poff = end_section_size
            return poff

        if pos[-1] < (shape[1] - end_section_size) and not (
            pos[-1] + (chunk_size - poff) + 1 == shape[-1]
        ):
            # Account for end section on snake row
            n_remaining = chunk_size - poff

            if n_remaining >= end_section_size:
                # can fill end section
                poff += end_section_size
            else:
                # update offset if remaining doesnt fill end sectino
                poff = end_section_size - n_remaining

    return poff


def snake_routine2(index, poff, spos, n_points_chunk, scan_shape):
    ss = []
    remaining = n_points_chunk - poff

    last_chunk_width = scan_shape[-1] % n_points_chunk
    print(f"index {index} position {spos} off {poff} remain {remaining} last_width {last_chunk_width}")

    distance_from_end  = scan_shape[-1] - spos[-1]

    if distance_from_end < remaining + n_points_chunk:
        print("HERE")
        #fill end from last chunk
        if poff == 0 and last_chunk_width != 0 and  distance_from_end == 1:
            cc = build_collection(spos,0,last_chunk_width, 1,scan_shape[-1], scan_shape[-1]-last_chunk_width-1, -1, type = "current")
            ss.append(cc)

            #can do no more
            return ss
        
        elif remaining >= last_chunk_width:
            cc = build_collection(spos,poff,poff+last_chunk_width,1, scan_shape[-1], scan_shape[-1]-last_chunk_width-1, -1, type = "last")
            ss.append(cc)
        else:
            #combined end fill
            last_input_start = poff
            last_input_stop = n_points_chunk
            last_output_start = 0
            last_output_stop = remaining

            current_input_start = 0
            current_input_stop = last_chunk_width - remaining
            current_output_start = remaining
            current_output_stop = n_points_chunk

            last = SliceInOut(
                slice(last_input_start, last_input_stop), slice(last_output_start, last_output_stop)
            )
         
            current = SliceInOut(
                slice(current_input_start, current_input_stop),
                slice(current_output_start, current_output_stop),
            )

            cc = ChunkSliceCollection("combined", spos)
            cc.current = current
            cc.last = last
            intermediate = SliceSize(slice(scan_shape[-1], scan_shape[-1] - last_chunk_width-1, -1), last_chunk_width)
            cc.intermediate = intermediate
            ss.append(cc)
            #combined end fill means no more can be written
            print("OTHER")
            return ss

    print("LAST")

    if last_chunk_width == 0:
        #chunks align
        poff = n_points_chunk
    elif remaining < last_chunk_width:
        poff = last_chunk_width - remaining
    else:
        poff = poff + last_chunk_width


    remaining = n_points_chunk - poff
    print(f"REMAINING {remaining} POFF {poff}")
    if remaining == 0:

        stop = spos[-1]-n_points_chunk
        if stop == -1:
            stop = None

        cc = build_collection(spos,0,n_points_chunk, 1,spos[-1],stop,-1, type = "current")
        ss.append(cc)
    else:
        #can we write a combined?
        last_input_start = poff
        last_input_stop = n_points_chunk
        last_output_start = 0
        last_output_stop = remaining

        current_input_start = 0
        current_input_stop = n_points_chunk - remaining
        current_output_start = remaining
        current_output_stop = n_points_chunk

        stop = spos[-1]-poff
        if stop == -1:
            stop = None

        print(stop)

        last = SliceInOut(
            slice(last_input_start, last_input_stop), slice(last_output_start, last_output_stop)
        )
        
        current = SliceInOut(
            slice(current_input_start, current_input_stop),
            slice(current_output_start, current_output_stop),
        )

        cc = ChunkSliceCollection("combined", spos)
        cc.current = current
        cc.last = last
        intermediate = SliceSize(slice(spos[-1]+remaining, stop, -1), n_points_chunk)
        cc.intermediate = intermediate
        ss.append(cc)

    # print(f"UPDATE TO index {index} position {spos} off {poff} remain {remaining} last_width {last_chunk_width}")

    return ss

def snake_routine_smallx2(index, spos, n_points_chunk, scan_shape):
    return []

def write_small_raster_complete(pos, i, scan_shape):
    # pos = list(spos)
    # pos[-2] = i

    in_start = i*scan_shape[-1]
    in_stop = scan_shape[-1]*(i+1)
    in_step = None
    out_start = pos[-1]
    out_stop = pos[-1] + scan_shape[-1]
    out_step = None
    print(pos)
    return build_collection(pos,in_start,in_stop,in_step, out_start, out_stop, out_step)

# def write_small_raster_complete(spos, i, scan_shape):
#     pos = list(spos)
#     pos[-2] = i

#     in_start = i*scan_shape[-1]
#     in_stop = scan_shape[-1]*(i+1)
#     in_step = None
#     out_start = spos[-1]
#     out_stop = spos[-1] + scan_shape[-1]
#     out_step = None
#     print(pos)
#     return build_collection(pos,in_start,in_stop,in_step, out_start, out_stop, out_step)

def write_small_snake_complete(pos, i, scan_shape):
    # pos = list(spos)
    # pos[-2] = i

    in_start = i*scan_shape[-1]
    in_stop = scan_shape[-1]*(i+1)
    in_step = None
    out_stop = None
    out_start = pos[-1] + scan_shape[-1]
    out_step = -1
    print(pos)
    return build_collection(pos,in_start,in_stop,in_step, out_start, out_stop, out_step)

def write_small_raster_offset(pos, i, scan_shape, offset):
        in_start = i*scan_shape[-1]+offset
        in_stop = scan_shape[-1]*(i+1)+offset
        in_step = None
        out_start = 0
        out_stop = scan_shape[-1]
        out_step = None
        return build_collection(pos,in_start,in_stop,in_step, out_start, out_stop, out_step)

def write_small_snake_offset(pos, i, scan_shape, offset):

        in_start = i*scan_shape[-1]+offset
        in_stop = scan_shape[-1]*(i+1)+offset
        in_step = None
        out_start = scan_shape[-1]
        out_stop = None
        out_step = -1
        return build_collection(pos,in_start,in_stop,in_step, out_start, out_stop, out_step)

def write_small_raster_combined(spos, n_points_chunk, position_offset, scan_shape):
            #deal with partial chunk first
        n_remaining = n_points_chunk - position_offset
        # Two reads contributing to chunk
        current = SliceInOut(
            slice(0, scan_shape[-1] - n_remaining),
            slice(n_remaining, scan_shape[-1]),
        )
        last = SliceInOut(
            slice(n_points_chunk - n_remaining, None), slice(0, n_remaining)
        )
        cc = ChunkSliceCollection("combined", spos)
        cc.current = current
        cc.last = last
        intermediate = SliceSize(slice(0, scan_shape[-1]), scan_shape[-1])
        cc.intermediate = intermediate
        print(f"{spos} has remaining {n_remaining}")
        return cc

def write_small_snake_combined(spos, n_points_chunk, position_offset, scan_shape):
        #deal with partial chunk first
        n_remaining = n_points_chunk - position_offset
        # Two reads contributing to chunk
        current = SliceInOut(
            slice(0, scan_shape[-1] - n_remaining),
            slice(n_remaining, scan_shape[-1]),
        )
        last = SliceInOut(
            slice(n_points_chunk - n_remaining, None), slice(0, n_remaining)
        )
        cc = ChunkSliceCollection("combined", spos)
        cc.current = current
        cc.last = last
        intermediate = SliceSize(slice(scan_shape[-1], None, -1), scan_shape[-1])
        cc.intermediate = intermediate
        print(f"{spos} has remaining {n_remaining}")
        return cc
        

def non_snake_routine_smallx2(index, spos, n_points_chunk, scan_shape, snake):
    is_snake_row = False
    if snake:
        spos,is_snake_row = utils.get_position_general(index, scan_shape)
    else:
        spos = utils.get_position(index, scan_shape, len(scan_shape))

    print(f"Start {spos}")
    # position_offset = calc_offset(spos, scan_shape, n_points_chunk)
    nrows = index // scan_shape[-1]
    complete = nrows*scan_shape[-1]
    #number of points left over from chunk compared to complete rows
    position_offset = complete % n_points_chunk
    slice_structure = []

    if position_offset == 0 and spos[-1] == 0:
        # Complete chunk read can be written to smaller chunks
        for i in range(n_points_chunk//scan_shape[-1]):

            if snake:
                tpos,is_snake_row = utils.get_position_general(index + (i)*scan_shape[-1], scan_shape)
            else:
                tpos = utils.get_position(index + (i)*scan_shape[-1], scan_shape, len(scan_shape))
            
            print(f"New pos {tpos}")

            if is_snake_row and snake:
                cc = write_small_snake_complete(tpos, i, scan_shape)
            else:
                cc = write_small_raster_complete(tpos, i, scan_shape)
            slice_structure.append(cc)
            is_snake_row = not is_snake_row
        
        return slice_structure
    
    else:
        #deal with partial chunk first
        n_remaining = n_points_chunk - position_offset

        if snake and is_snake_row:
            cc = write_small_snake_combined(spos, n_points_chunk, position_offset, scan_shape)
        else:
            cc = write_small_raster_combined(spos, n_points_chunk, position_offset, scan_shape)
            
        slice_structure.append(cc)
        is_snake_row = not is_snake_row
        old_n_remaining = n_points_chunk - position_offset
        offset = scan_shape[-1] - n_remaining
        print(f"COMPLETED PARTIALS {offset}")
        
        for i in range(0, (n_points_chunk-offset)//scan_shape[-1]):

            updated_index = index + (i+1)*scan_shape[-1]-old_n_remaining

            if updated_index == math.prod(scan_shape):
                break
            
            if snake:
                tpos,is_snake_row = utils.get_position_general(updated_index, scan_shape)
            else:
                tpos = utils.get_position(updated_index, scan_shape, len(scan_shape))
                print(f"New pos2 {tpos}")

            if snake and is_snake_row:
                cc = write_small_snake_offset(tpos, i, scan_shape, offset)
            else:
                cc = write_small_raster_offset(tpos, i, scan_shape, offset)
            is_snake_row = not is_snake_row
            slice_structure.append(cc)

        return slice_structure

def get_slice_structure2(index, n_points_chunk, scan_shape, snake):

    is_snake_row = False
    if snake:
        spos,is_snake_row = utils.get_position_general(index, scan_shape)
    else:
        spos = utils.get_position(index, scan_shape, len(scan_shape))

    if n_points_chunk > scan_shape[-1]:
        return non_snake_routine_smallx2(index, spos, n_points_chunk, scan_shape, snake)
    # elif n_points_chunk > scan_shape[-1] and snake:
    #     print("DO SMALL SNAKE")
    #     return snake_routine_smallx2(index, spos, n_points_chunk, scan_shape)

    ss = []

    last_chunk_width = scan_shape[-1] % n_points_chunk


    nrows = index // scan_shape[-1]
    complete = nrows*scan_shape[-1]
    #number of points left over from chunk compared to complete rows
    poff = complete % n_points_chunk

    if is_snake_row:
        # return []
        return snake_routine2(index,poff, spos, n_points_chunk, scan_shape)



    print(f"index {index} position {spos} nrow {complete} comp {poff}")
    remaining = n_points_chunk - poff

    if poff == 0:
        remaining = 0

    start_write = index-remaining
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
        cc = build_collection(spos,0,end_write_offset,1, start[-1], end[-1], 1, type = "current")
        ss.append(cc)
    else:
        #combined
        print(f"offset {poff}, remaining {remaining}, last width {last_chunk_width}, npchunk {n_points_chunk} endoffset {end_write_offset}")

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
            slice(last_input_start, last_input_stop), slice(last_output_start, last_output_stop)
        )
        current = SliceInOut(
            slice(current_input_start, current_input_stop),
            slice(current_output_start, current_output_stop),
        )
        cc = ChunkSliceCollection("combined", spos)
        cc.current = current
        cc.last = last
        intermediate = SliceSize(slice(start[-1], start[-1] + end_write_offset), end_write_offset)
        cc.intermediate = intermediate
        ss.append(cc)


    if remaining >= last_chunk_width and end[-1] + last_chunk_width == scan_shape[-1]:
        print("Acccount for additional")
        new_start = list(end)
        new_end = list(new_start)
        new_end[-1] = new_end[-1] + last_chunk_width
        # print(f"ADDITIONAL Start {new_start} End {new_end}")
        cc = build_collection(spos,poff,poff+last_chunk_width,1, new_start[-1], new_end[-1], 1, type = "current")
        ss.append(cc)
        print(cc)


    return ss


        
