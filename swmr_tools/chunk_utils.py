import numpy as np

class SliceInOut:

    def __init__(self, input, output):
        self.input = input
        self.output = output

class SliceSize:
     
     def __init__(self,slice, size):
         self.slice = slice
         self.size = size

class ChunkSliceCollection:

    types = ("current", "last", "combined")
    
    def __init__(self, type):
        
        if type not in ChunkSliceCollection.types:
            raise RuntimeError(f"{type} not in {ChunkSliceCollection.types}")

        self.type = type
        self.current = None
        self.last = None
        self.intermediate = None

    def __repr__(self):
        out = self.type

        if self.current:
            out+= (" Scurrent " + str(self.current.input) + " " + str(self.current.output))
        
        if self.last:
            out+= (" Slast " + str(self.last.input) + " " + str(self.last.output))

        if self.intermediate:
            out+= (" Sinter " + str(self.intermediate.slice) + " " + str(self.intermediate.size))

        return out


def calc_offset(pos,shape,chunk_size, snake = False):
    num = pos[0] * shape[-1]
    poff = num%chunk_size

    if snake and pos[0] % 2 == 1:
        end_section_size = shape[1]%chunk_size

        if pos[-1] < (shape[1]-end_section_size) and not (pos[1] + (chunk_size - poff) + 1 == shape[1]):
            #Account for end section on snake row
            n_remaining = chunk_size - poff

            if n_remaining >= end_section_size:
                #can fill end section
                poff += end_section_size
            else:
                #update offset if remaining doesnt fill end sectino
                poff = end_section_size-n_remaining

    return poff


def snake_routine(spos, n_points_chunk, scan_shape):
    position_offset = calc_offset(spos,scan_shape,n_points_chunk,snake=True)

    slice_structure = []

    if position_offset == 0 and not (spos[-1] + 1 - n_points_chunk < 0):
        #Complete chunk written (in reverse)
        current = SliceInOut(slice(None,None,-1),slice(spos[1]-n_points_chunk+1,spos[1]+1))
        cc = ChunkSliceCollection("current")
        cc.current = current

        slice_structure.append(cc)

        return slice_structure
    elif position_offset == 0 and spos[-1] + n_points_chunk > scan_shape[-1]:
        #COMPLETE CHUNK READ, SECTION OF START WRITTEN TO END OF GRID
        # position_offset = scan_shape[-1] - spos[-1]
        raise Exception("here2")
        #return position_offset, []
        output[spos[0],spos[1]:spos[1]+position_offset] += data[:position_offset]
    else:

        #CASE 1 - backwards row start, enough remaining to fill complete partial end row chunk
        if spos[1] + (n_points_chunk - position_offset) + 1 == scan_shape[1]:

            end_section_size = scan_shape[1]%n_points_chunk

            if n_points_chunk - position_offset >= end_section_size:

                #was slice(-1*end_section_size,None,None)
                #now 

                last = SliceInOut(slice(position_offset+end_section_size-1, position_offset-1,-1),slice(scan_shape[1]-end_section_size,None,None))
                cc = ChunkSliceCollection("last")
                cc.last = last
                slice_structure.append(cc)

                position_offset = position_offset + end_section_size
                
            else:
                n_remaining = n_points_chunk - position_offset

                current = SliceInOut(slice(end_section_size-n_remaining-1,None,-1),slice(None,end_section_size-n_remaining))
                last = SliceInOut(slice(None,-1*(n_remaining+1),-1),slice(end_section_size-n_remaining,None))
                cc = ChunkSliceCollection("combined")
                cc.current = current
                cc.last = last
                intermediate = SliceSize(slice(scan_shape[1]-end_section_size,None),end_section_size)
                cc.intermediate = intermediate

                slice_structure.append(cc)

                position_offset = end_section_size-n_remaining
                return slice_structure
        

        if (position_offset != 0):
            #CASE 2 - 2 READS CONTRIBUTING TO CHUNK
            start = spos[1]-position_offset+1
            end = spos[1]-position_offset+n_points_chunk+1

            n_remaining = n_points_chunk - position_offset

            current = SliceInOut(slice(position_offset-1,None,-1),slice(None,position_offset))
            last = SliceInOut(slice(None,-1*(n_remaining+1),-1),slice(position_offset,None))
            cc = ChunkSliceCollection("combined")
            cc.current = current
            cc.last = last
            intermediate = SliceSize(slice(start,end),n_points_chunk)
            cc.intermediate = intermediate

            ss = {"last_input": slice(None,-1*(n_remaining+1),-1),
                  "current_input": slice(position_offset-1,None,-1),
                  "last_intermed" : slice(position_offset,None),
                  "current_intermed" : slice(None,position_offset),
                  "intermed_output" : slice(start,end),
                  "intermed_size": n_points_chunk}
            
            print("combined full chunk")
            slice_structure.append(cc)
            
    return slice_structure



def non_snake_routine(spos, n_points_chunk, scan_shape):

    position_offset = calc_offset(spos,scan_shape,n_points_chunk)

    slice_structure = []

    if position_offset == 0 and not (spos[-1] + n_points_chunk > scan_shape[-1]):
        #Complete chunk read can be written
        cc = ChunkSliceCollection("current")
        current = SliceInOut(slice(0,n_points_chunk),slice(spos[1],spos[1]+n_points_chunk))
        cc.current = current
        slice_structure.append(cc)
    elif position_offset == 0 and spos[-1] + n_points_chunk > scan_shape[-1]:
        #Complete chunk read, section if start written to end of grid
        position_offset = scan_shape[-1] - spos[-1]

        current = SliceInOut(slice(0,position_offset),slice(spos[1],spos[1]+position_offset))
        cc = ChunkSliceCollection("current")
        cc.current = current

        slice_structure.append(cc)

    else:
       
        n_remaining = n_points_chunk - position_offset

        nstart = spos[-1] - n_remaining

        if (n_remaining + position_offset + nstart > scan_shape[-1]):
            #Two reads contributing to end "short" chunk
            num = scan_shape[-1] - nstart - n_remaining
            position_offset = num

            current = SliceInOut(slice(0,num),slice(n_remaining, None))
            last = SliceInOut(slice(n_points_chunk-n_remaining,None),slice(0,n_remaining))
            cc = ChunkSliceCollection("combined")
            cc.current = current
            cc.last = last
            intermediate = SliceSize(slice(nstart,nstart+n_remaining+ num),n_remaining + num)
            cc.intermediate = intermediate

            slice_structure.append(cc)

        else:
            #Two reads contributing to chunk
            current = SliceInOut(slice(0,position_offset),slice(n_remaining,n_remaining + position_offset))
            last = SliceInOut(slice(n_points_chunk-n_remaining, None),slice(0,n_remaining))
            cc = ChunkSliceCollection("combined")
            cc.current = current
            cc.last = last
            intermediate = SliceSize(slice(nstart,nstart+n_remaining+ position_offset),n_points_chunk)
            cc.intermediate = intermediate

            slice_structure.append(cc)

        if (n_remaining + spos[-1] +position_offset) >= scan_shape[-1] and scan_shape[-1] != spos[-1]+position_offset:
            #Single chunk contributing to end chunk
            w_end = scan_shape[-1] -spos[-1]
            
            current = SliceInOut(slice(position_offset,w_end),slice(spos[1]+position_offset,scan_shape[1]))
            cc = ChunkSliceCollection("current")
            cc.current = current

            slice_structure.append(cc)

    return slice_structure