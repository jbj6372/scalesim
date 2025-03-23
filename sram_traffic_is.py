import math 
from tqdm import tqdm


def sram_traffic(
        dimension_rows=4,
        dimension_cols=4,
        ifmap_h=7, ifmap_w=7,
        filt_h=3, filt_w=3,
        num_channels=3,
        strides=1, num_filt=8,
        ofmap_base=2000000, filt_base=1000000, ifmap_base=0,
        sram_read_trace_file="sram_read.csv",
        sram_write_trace_file="sram_write.csv"
    ):

    total_pikachu = 0
    pikachu_i = 0
    pikachu_f = 0
    pikachu_o = 0
    pikachu_o_sram = 0
    pikachu_o_array = 0
    pikachu_o_rf = 0
    total_sram_pikachu = 0
    total_array_pikachu = 0
    total_rf_pikachu = 0

    # Dimensions of output feature map channel
    E_h = math.floor((ifmap_h - filt_h + strides) / strides)
    E_w = math.floor((ifmap_w - filt_w + strides) / strides)
    
    # Number of pixels in one convolution window
    px_per_conv_window = filt_h * filt_w * num_channels
    r2c = px_per_conv_window
    rc = filt_w * num_channels
    hc = ifmap_w * num_channels

    # Total number of ofmap px across all channels
    num_ofmap_px = E_h * E_w * num_filt
    e2  = E_h * E_w
    e2m = num_ofmap_px
    
    # Variables to calculate folds in runtime
    num_h_fold = 1
    num_v_fold = 1
    max_parallel_window = 1

    # Variables for utilization calculation
    util = 0
    compute_cycles = 0

    if dimension_rows < px_per_conv_window:
        num_h_fold = math.ceil(px_per_conv_window/dimension_rows)
    else:
        max_parallel_window = math.floor(dimension_rows/ px_per_conv_window)

    reqd_cols = e2          # Total number of cols need to be mapped
    max_cols_per_v_fold = max_parallel_window * dimension_cols
    num_v_fold = math.ceil(reqd_cols / max_cols_per_v_fold)
    # print("max_cols_per_v_fold: ", max_cols_per_v_fold)
    # print("num_v_fold: ", num_v_fold)
    # print("hc: ", hc)

    remaining_cols = reqd_cols
    cycles = 0
    prev_cycl = 0
    
    # These are the starting addresses of ifmap windows in the memory 
    #offmap 을 순회하며 ifmap 의 base 주소를 미리 계산, 이후 all_ifmap_base_addr_list 에 저장
    all_ifmap_base_addr_list = []
    for px in range(int(e2)):
                    #ifmap 행                   #ifmap 열   
        addr = int(px / E_w) * strides * hc + (px%E_w) * strides * num_channels
        all_ifmap_base_addr_list.append(addr)

    # These are the starting addresses of filter windows in the memory
    #filter 을 순회하며 filter 의 base 주소를 미리 계산, 이후 all_filt_addr_list 에 저장
    hc = ifmap_w * num_channels
    all_filt_addr_list = []
    for c in range(num_filt):         #number of ofmap px in a ofmap channel
        addr = (c) * r2c + filt_base 
        all_filt_addr_list.append(addr)

    # #진태 수정
    # temp_filt_base = filt_base

    #근데 왜 필터는 base 를 더해놓고 ifmap 은 안더하지? 진짜 모름
    for v in tqdm(range(int(num_v_fold))):

        # Take a slice of the starting addresses that are relevant for this v_fold
        done = reqd_cols - remaining_cols
        cols_this_fold = int(min(remaining_cols, max_parallel_window * dimension_cols))
        # print("cols_this_fold: ", cols_this_fold)
        idx_start = done
        idx_end = idx_start + cols_this_fold
        col_base_addr_list = all_ifmap_base_addr_list[idx_start:idx_end]

        if num_h_fold > 1:

            rem_h = r2c

            for h in range(num_h_fold):
                rows_this_fold = min(rem_h, dimension_rows)

                cycles_i, pikachu_i= \
                    gen_trace_ifmap_partial(
                        h_fold = h,
                        rc = rc, hc = hc,
                        col_addrs   = col_base_addr_list,
                        cycle       = cycles,
                        num_rows    = dimension_rows,
                        num_cols    = dimension_cols,
                        active_rows = rows_this_fold,
                        active_cols = cols_this_fold,
                        ifmap_base = ifmap_base,
                        sram_read_trace_file = sram_read_trace_file
                    )
                total_sram_pikachu+=pikachu_i

                data_out_cycles = cycles_i

                cycles_f, all_filt_addr_list, pikachu_f =\
                    gen_trace_filter_partial(
                        cycle = cycles_i,
                        h_fold = h, v_fold = v,
                        num_rows = dimension_rows, num_cols= dimension_cols,
                        num_filters= num_filt,
                        filt_addr_list= all_filt_addr_list,
                        active_rows= rows_this_fold, active_cols= cols_this_fold,
                        ofmap_base_addr= ofmap_base,
                        sram_read_trace_file= sram_read_trace_file
                    )
                total_sram_pikachu+=pikachu_f

                cycles_o, pikachu_o, pikachu_o_sram, pikachu_o_array, pikachu_o_rf = \
                    gen_trace_ofmap(
                        cycle = data_out_cycles,
                        v_fold= v, parallel_window= 1,
                        num_ofmap_this_fold= cols_this_fold,
                        window_size= rows_this_fold, num_filters= num_filt,
                        num_cols= dimension_cols, num_rows= dimension_rows,
                        ofmap_base= ofmap_base,
                        sram_write_trace_file= sram_write_trace_file
                    )
                total_sram_pikachu+=pikachu_o_sram
                total_array_pikachu+=pikachu_o_array
                total_rf_pikachu+=pikachu_o_rf

                util_this_fold = (rows_this_fold * cols_this_fold) /(dimension_rows * dimension_cols)

                rem_h -= rows_this_fold
                cycles = max(cycles_f, cycles_o)

                del_cycl = cycles - prev_cycl
                util += util_this_fold * del_cycl
                compute_cycles += del_cycl
                prev_cycl = cycles

        else:
            parallel_window = int(math.ceil(cols_this_fold / dimension_cols))

            cycles_i, pikachu_i = \
                    gen_trace_ifmap(
                        cycle = cycles,
                        r= filt_w, rc= rc, hc= hc,
                        parallel_window= parallel_window,
                        ifmap_base_this_fold=col_base_addr_list,
                        num_ifmap_this_fold= cols_this_fold,
                        num_rows= dimension_rows, num_cols= dimension_cols,
                        window_size= r2c,
                        ifmap_base= ifmap_base,
                        sram_read_trace_file= sram_read_trace_file
                    )
            total_srma_pikachu+=pikachu_i

            cycles_f, pikachu_f = \
                    gen_trace_filter(
                        cycle = cycles_i,
                        num_filters= num_filt, parallel_window= parallel_window,
                        window_size= r2c,
                        num_rows= dimension_rows, num_cols=dimension_cols,
                        filter_base= filt_base,
                        sram_read_trace_file= sram_read_trace_file
                    )
            total_sram_pikachu+=pikachu_f

            cycles_o, pikachu_o, pikachu_o_sram, pikachu_o_array, pikachu_o_rf  = \
                    gen_trace_ofmap(
                        cycle = cycles_i,
                        v_fold = v, parallel_window= parallel_window,
                        num_ofmap_this_fold= cols_this_fold,
                        window_size= r2c,
                        num_filters= num_filt,
                        num_rows= dimension_rows, num_cols= dimension_cols,
                        ofmap_base= ofmap_base,
                        sram_write_trace_file= sram_write_trace_file
                    )
            total_sram_pikachu+=pikachu_o_sram
            total_array_pikachu+=pikachu_o_array
            total_rf_pikachu+=pikachu_o_rf


            cycles = max(cycles_f, cycles_o)
            #rows_this_fold = parallel_window * r2c

            # Since multiple filters are being mapped on a single col due to large number of rows
            # util calculation is a little involved,
            # cols_this_fold --> number of filters mapped this fold
            rem = cols_this_fold
            tmp_util = 0
            for _ in range(parallel_window):
                col_used = min(rem, dimension_cols)
                row_used = r2c  # Number of row used will always be in multiple of r2c,
                # parallel window calc took care of this
                tmp_util += row_used * col_used
                rem -= col_used

            util_this_fold = tmp_util / (dimension_rows * dimension_cols)

            del_cycl = cycles - prev_cycl
            util += util_this_fold * del_cycl
            compute_cycles += del_cycl
            prev_cycl = cycles

        remaining_cols -= cols_this_fold
        # v_fold 한번 돌고 나면 all_filt_base 363 씩 빼주기
        for i in range(len(all_filt_addr_list)):
            all_filt_addr_list[i] -= e2m

    avg_util = (util / compute_cycles) * 100

    print("Total cycles: ", cycles)
    print("Total Pikachu: ", total_pikachu)
    print("Total SRAM Pikachu: ", total_sram_pikachu)
    print("Total Array Pikachu: ", total_array_pikachu)
    print("Total RF Pikachu: ", total_rf_pikachu)
    
    return (str(cycles), avg_util, total_sram_pikachu, total_array_pikachu, total_rf_pikachu)


def gen_trace_ifmap_partial(
                    h_fold = 0,
                    rc = 3, hc = 3,
                    col_addrs=[],       #Ensure that this takes care of the v_folding
                    cycle=0,
                    num_rows=4, num_cols=4,
                    active_rows=4, active_cols=4,
                    ifmap_base= 0,
                    sram_read_trace_file="sram_read.csv"
):
        # h_fold 는 0부터 쭉 증가.
        #num_rows = 12 고정
        index = h_fold * num_rows
        # print("h_fold: ", h_fold)
        # print("index: ", index)
        pikachu_i = 0

        outfile = open(sram_read_trace_file, 'a')
        # outfile.write("ifmap start\n")

        # jin = open("jin.csv", 'a')
        # jin.write("Cycle, Index, Row, Col, Addr, Row_idx, Col_idx\n")

        # output formatting: Add empty commas for row addresses as no element is fed from the left
        prefix = ""
        for r in range(num_rows):
            prefix += ", "

        # Entries per cycle 
        for r in range(active_rows):              # number of rows this fold
            entry = str(cycle) + ", " + prefix

            for c in range(active_cols):

                # Calculating next address
                row_idx = math.floor((index + r)/ rc)
                col_idx = (index + r) % rc

                addr = row_idx * hc + col_idx
                addr += col_addrs[c] + ifmap_base

                entry += str(int(addr)) + ", "
                pikachu_i+=6

                temp = str(cycle) + ", "
                #index, r, c, addr, row_idx, col_idx, col_addrs[c], ifmap_base
                temp += str(index) + ", " + str(r) + ", " + str(c) + ", " + str(addr) + ", " + str(row_idx) + ", " + str(col_idx) + ", " + "\n"
                # jin.write(temp)
                
                # if(addr == 32) :
                #     print("Index: ", index, " r: ", r)
                #     print("Row: ", r, " Col: ", c, " Addr: ", addr)
                #     print("Row_idx: ", row_idx, " Col_idx: ", col_idx)
                #     print("Col_addr: ", col_addrs[c])
                #     print("Ifmap_base: ", ifmap_base)
                #     print("--------------------")
                
                # if(addr == 672) :
                #     print("Index: ", index, " r: ", r)
                #     print("Row: ", r, " Col: ", c, " Addr: ", addr)
                #     print("Row_idx: ", row_idx, " Col_idx: ", col_idx)
                #     print("Col_addr: ", col_addrs[c])
                #     print("Ifmap_base: ", ifmap_base)


                # print("Row: ", r, " Col: ", c, " Addr: ", addr)

            #열이 부족할 경우 빈칸으로 채워줌
            if active_cols < num_cols:
                delta = num_cols - active_cols
                for c in range(delta):
                    entry += ", "

            cycle += 1
            entry += "\n"
            outfile.write(entry)
        # outfile.write("ifmap end\n")
        outfile.close()

        return cycle, pikachu_i


def gen_trace_filter_partial(
                    cycle = 0,
                    h_fold = 0, v_fold = 0,
                    num_rows = 4, num_cols = 4,
                    num_filters = 4,
                    filt_addr_list = [],
                    active_rows = 4,
                    active_cols = 4,
                    ofmap_base_addr = 20000000,
                    sram_read_trace_file = "sram_read.csv"
):

    pikachu_f = 0
    local_cycles = cycle

    outfile = open(sram_read_trace_file, 'a')

    # outfile.write("filter start\n")

    # This list tracks the PS address generation per col
    ofmap_px_id_list = []
    for c in range(active_cols):
        ofmap_index = v_fold * num_cols + c
        ofmap_px_id_list.append(ofmap_index)

    # Postfix is the empty string indicating that no data is fed from the cols
    postfix = ""
    for _ in range(active_cols):
        postfix += ", "


    # Per cycle one filter value is applied to all rows
    #num_row_traces = num_filters + active_cols
    for f in range(num_filters):
        this_filt_addr = filt_addr_list[f]
        
        entry = str(local_cycles) + ", "

        # Calculate the row addresses for this cycle
        row_entry = []
        for r in range(active_rows):
            row_entry.append(this_filt_addr)
            this_filt_addr += 1

        filt_addr_list[f] = this_filt_addr

        # The log will get the addresses in reverse
        l = len(row_entry)
        for ridx in range(l):
            entry += str(row_entry[l - ridx - 1]) + ", "
            pikachu_f += 6

        # Anand: TODO: Add partial sum input trace
        # Calculate the column addresses
        # In case of partial mapping partial sums (OFMAP addresses) need to be passed into the array
        # This partial sum is fed from the top of the array and summed with the sums generated in this h_fold
        #if h_fold == 0:
        #    for _ in range(num_cols):
        #        entry += ", "

        #else:
        #    for col in range(active_cols):
        #        ofmap_ch_index = f - col

        #        if ofmap_ch_index >= 0:
        #            ofmap_addr = ofmap_px_id_list[f] * num_filters + ofmap_ch_index
        #            ofmap_addr += ofmap_base_addr
        #            entry += str(ofmap_addr) + ", "

        #        else:
        #            entry += ", "

        local_cycles += 1
        entry += postfix + "\n"
        outfile.write(entry)

    # outfile.write("filter end\n")

    outfile.close()
    return local_cycles, filt_addr_list, pikachu_f


def gen_trace_ofmap(
                    cycle = 0,
                    v_fold = 0, parallel_window = 1,
                    num_ofmap_this_fold = 4,
                    window_size = 16,
                    num_filters = 4,
                    num_rows = 4, num_cols = 4,
                    ofmap_base = 2000000,
                    sram_write_trace_file = "sram_write.csv"
):

    pikachu_o = 0
    pikachu_move_i = 0
    pikachu_move_f = 0
    pikachu_calculate = 0
    pikachu_move_o = 0

    active_cols_list = []
    rem = num_ofmap_this_fold
    for p in range(parallel_window):
        a = min(rem, num_cols)
        active_cols_list.append(int(a))
        rem -= a

    start_index = num_cols * v_fold * parallel_window
    end_index   = start_index + num_ofmap_this_fold

    ofmap_px_index_list = []            # This list has the indices of ofmap px on one ofmap
    for px in range(start_index,end_index):
        add = px * num_filters
        ofmap_px_index_list.append(add)

    # This offset indicates the cycle in which the data from the first col is ready
    local_cycle = cycle + 1

    outfile = open(sram_write_trace_file, 'a')

    total_ofmap_cycles = num_filters + max(active_cols_list)
    for f in range(total_ofmap_cycles):
        entry = str(local_cycle) + ", "

        for p in range(parallel_window):
            active_cols = active_cols_list[p]

            for c in range(active_cols):
                ofmap_ch = f - c

                if (ofmap_ch >= 0) and (ofmap_ch < num_filters):
                    idx = c + p * num_cols
                    add = ofmap_px_index_list[idx] + ofmap_ch
                    add += ofmap_base
                    entry += str(add) + ", "

                    pikachu_move_i += 2
                    pikachu_move_f += 2
                    pikachu_calculate += 1
                    pikachu_move_o += 6

                else:
                    entry += ", "

        entry += "\n"
        outfile.write(entry)
        local_cycle += 1

    outfile.close()
    pikachu_o_sram = pikachu_move_o
    pikachu_o_array = pikachu_move_f+pikachu_move_i
    pikachu_o_rf = pikachu_calculate
    pikachu_o = pikachu_move_i + pikachu_move_f + pikachu_calculate + pikachu_move_o

    return (local_cycle - 1), pikachu_o, pikachu_o_sram, pikachu_o_array, pikachu_o_rf


def gen_trace_ifmap(
                    cycle = 0,
                    r = 3, rc = 9, hc = 27,
                    parallel_window = 1,
                    ifmap_base_this_fold = [],
                    num_ifmap_this_fold = 1,
                    num_rows =4, num_cols= 4,
                    window_size = 16,
                    ifmap_base = 0,
                    sram_read_trace_file = "sram_read.csv"
):
    local_cycle = cycle

    pikachu_i = 0

    outfile = open(sram_read_trace_file, 'a')

    active_cols_list = []
    rem = num_ifmap_this_fold
    for p in range(parallel_window):
        a = min(rem, num_cols)
        active_cols_list.append(int(a))
        rem -= a

    prefix = ""
    for _ in range(num_rows):
        prefix += ", "

    for p in range(parallel_window):
        start_idx = p * num_cols
        end_idx = start_idx + active_cols_list[p]
        ifmap_base_addr = ifmap_base_this_fold[start_idx:end_idx]

        for idx in range(window_size):
            entry = str(local_cycle) + ", "
            entry += prefix

            # Calculating address within a window
            row_idx = math.floor(idx / rc)
            col_idx = (idx) % rc

            local_addr = row_idx * hc + col_idx

            active_cols = active_cols_list[p]
            for col in range(active_cols):
                add = local_addr + ifmap_base_addr[col] +ifmap_base
                entry += str(int(add)) + ", "
                pikachu_i += 6

            if active_cols < num_cols:
                for _ in range(active_cols, num_cols):
                    entry += ", "

            entry += "\n"
            outfile.write(entry)
            local_cycle += 1

    outfile.close()
    return local_cycle, pikachu_i

def gen_trace_filter(
                    cycle = 0,
                    num_filters = 4, parallel_window = 1,
                    window_size = 27,
                    num_rows = 4, num_cols =4,
                    filter_base = 10000000,
                    sram_read_trace_file = "sram_read.csv"
):
    pikachu_f = 0
    local_cycle = cycle
    outfile = open(sram_read_trace_file, 'a')

    postfix = ""
    for _ in range(num_cols):
        postfix += ", "

    for f in range(num_filters):
        entry = str(local_cycle) + ", "

        for p in range(parallel_window):
            for indx in range(window_size):
                add = f * window_size  + filter_base + (window_size - indx - 1)
                entry += str(add) + ", "
                pikachu_f += 6

        rows_written = parallel_window * window_size
        if rows_written < num_rows:
            for _ in range(rows_written, num_rows):
                entry += ", "

        entry += postfix + "\n"
        outfile.write(entry)
        local_cycle += 1

    outfile.close()
    return local_cycle, pikachu_f

if __name__ == "__main__":
    h_h = 5
    h_w = 5

    r_h = 2
    r_w = 2

    c = 2
    u =2

    m = 30

    dim_h = 4
    dim_v = 9

    sram_traffic(
        dimension_rows= dim_h,
        dimension_cols= dim_v,

        ifmap_h= h_h, ifmap_w= h_w,
        filt_h= r_h, filt_w= r_w,
        num_channels= c,
        strides= u,

        num_filt= m
    )
