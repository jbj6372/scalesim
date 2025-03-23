import math 
from tqdm import tqdm

def sram_traffic(
        rf_size=512,
        dimension_rows=12,
        dimension_cols=14,
        ifmap_h=224, ifmap_w=224,
        filt_h=11, filt_w=11,
        num_channels=3,
        strides=4, num_filt=96,
        ofmap_base=2000000, filt_base=1000000, ifmap_base=0,
        sram_read_trace_file="sram_read.csv",
        sram_write_trace_file="sram_write.csv",
        sram_read_pikachu = 6,
        array_read_pikachu = 2,
        rf_read_pikachu = 1,
    ):

    avg_util = 0
    cycles = 0
    total_pikachu = 0
    pikachu_i = 0
    pikachu_f = 0
    pikachu_o = 0
    pikachu_o_sram = 0
    pikachu_o_array = 0
    pikachu_o_rf = 0

    total_pikachu_sram = 0
    total_pikachu_array = 0
    total_pikachu_rf = 0

    E_h = math.floor((ifmap_h - filt_h)/strides) + 1
    E_w = math.floor((ifmap_w - filt_w)/strides) + 1

    conv_window_size = E_w

    r2c = filt_h * filt_w * num_channels
    rc = filt_w * num_channels
    hc = ifmap_w * num_channels

    num_conv_windows = rc

    # Total number of ofmap px across all channels
    num_ofmap_px = E_h * E_w * num_filt
    e2  = E_h * E_w
    e2m = num_ofmap_px
    ef = num_filt*E_w
    
    # Variables to calculate folds in runtime
    num_h_fold = 1
    num_v_fold = 1
    max_parallel_window = 1

    # Variables for utilization calculation
    util = 0
    compute_cycles = 0
    ifmap_pixel_per_conv = hc
    affordable_ifmap_pixel = rf_size - rc

    num_logical_row = filt_h
    num_logical_col = hc-num_logical_row+1

    num_last_row = num_logical_row%dimension_rows
    num_last_col = num_logical_col%dimension_cols

    num_v_fold = math.ceil(num_logical_row / dimension_rows)
    num_h_fold = math.ceil(num_logical_col / dimension_cols)

    all_ifmap_row_index_list = []
    all_filt_row_index_list = []

    total_ofmap_cols = 0
    total_ofmap_rows = 0

    cycles_f = 0
    cycles_o = 0
    prev_cycl = 0

    # print(num_v_fold)
    # print(num_h_fold)

    for i in range(num_v_fold):
        for j in range(num_h_fold):
            idx = i*dimension_rows + j*dimension_cols
            all_ifmap_row_index_list.append(idx)
            # print(idx)

    for i in range(num_filt):
        idx = i*r2c
        all_filt_row_index_list.append(idx)

    # for i in range(num_v_fold):


    for c in range(num_channels):
        for v in tqdm(range(int(num_v_fold))):
            remained_cols = num_logical_col
            rows_this_fold = dimension_rows
            if(v==num_v_fold-1):
                rows_this_fold = num_last_row
            for h in range(int(num_h_fold)):
                cols_this_fold = dimension_cols
                if(h==num_h_fold):
                    cols_this_fold = num_last_col
                
                cycles_i, pikachu_i = \
                    gen_trace_ifmap_partial(
                        h_fold = h,
                        v_fold = v,
                        rc = rc, hc = hc,
                        index = all_ifmap_row_index_list[v+h],
                        cycle       = cycles,
                        num_rows    = dimension_rows,
                        num_cols    = dimension_cols,
                        num_channels = num_channels,
                        cnt_per_row = ifmap_w,
                        active_rows = rows_this_fold,
                        active_cols = cols_this_fold,
                        channel_idx = c,
                        ifmap_base = ifmap_base,
                        sram_read_trace_file = sram_read_trace_file,
                        sram_read_pikachu = 6,

                    )
                total_pikachu_sram += pikachu_i
                
                data_out_cycles = cycles_i

                for f in range(num_filt):

                    cycles_f, pikachu_f =\
                            gen_trace_filter_partial(
                                cycle = data_out_cycles,
                                h_fold = h, v_fold = v,
                                num_rows = dimension_rows, num_cols= dimension_cols,
                                num_filters= num_filt,
                                current_filter = f,
                                num_channels= num_channels,
                                filt_addr_list= all_filt_row_index_list,
                                active_rows= rows_this_fold, active_cols= cols_this_fold,
                                ofmap_base_addr= ofmap_base,
                                filter_base_addr= filt_base,
                                channel_idx = c,
                                cnt_per_row= filt_w,
                                rc = rc,
                                sram_read_trace_file= sram_read_trace_file,
                                sram_read_pikachu = 6
                            )

                    cycles_o, total_ofmap_cols, pikachu_o, pikachu_o_sram, pikachu_o_array, pikachu_o_rf = \
                        gen_trace_ofmap(
                            cycle = data_out_cycles,
                            v_fold= v, parallel_window= 1,
                            num_ofmap_this_fold= cols_this_fold,
                            window_size= rows_this_fold, num_filters= num_filt,
                            num_cols= dimension_cols, num_rows= dimension_rows,
                            ofmap_base= ofmap_base,
                            sram_write_trace_file= sram_write_trace_file,
                            strides= strides,
                            num_channels= num_channels,
                            h_fold = h,
                            total_ofmap_cols = total_ofmap_cols,
                            active_rows= rows_this_fold, active_cols= cols_this_fold,
                            total_ofmap_rows= total_ofmap_rows,
                            ew = E_w,
                            currnet_filter= f,
                            ef = ef,
                            fw = filt_w,
                            iw = ifmap_w,
                            dimension_cols= dimension_cols,
                            sram_read_pikachu = 6,
                            array_read_pikachu = 2,
                            rf_read_pikachu = 1
                        )

                    #output picxel 하나를 계산하기 위한


                    total_pikachu_sram += pikachu_f+pikachu_o_sram
                    total_pikachu_array += pikachu_o_array
                    total_pikachu_rf += pikachu_o_rf

                    data_out_cycles = max(cycles_f, cycles_o)

                #이 아래는 아직 조작 안함
        
                # util_this_fold = (rows_this_fold * cols_this_fold) /(dimension_rows * dimension_cols)

                cycles = max(cycles_f, cycles_o)

                del_cycl = cycles - prev_cycl
                # util += util_this_fold * del_cycl
                compute_cycles += del_cycl
                prev_cycl = cycles

    # print("Total Pikachu: ", total_pikachu)
    # print("Total Cycle: ", cycles)
    return (str(cycles), avg_util, total_pikachu_sram, total_pikachu_array, total_pikachu_rf)

def gen_trace_ifmap_partial(
                    h_fold = 0,
                    v_fold = 0,
                    rc = 3, hc = 3,
                    index = 0,
                    cycle = 0,
                    cnt_per_row = 10,
                    num_rows=4, num_cols=4,num_channels = 3,
                    active_rows=4, active_cols=4,
                    ifmap_base= 0,
                    channel_idx = 0,
                    sram_read_trace_file="sram_read.csv",
                    sram_read_pikachu = 6
):
        outfile = open(sram_read_trace_file,'a')

        prefix = ""
        pikachu_i = 0
        sram_read_pikachu = 6
        for r in range(num_rows):
            prefix += ", "

        for i in range(cnt_per_row):              # number of rows this fold
            entry = str(cycle) + ", " + prefix

            for r in range(active_rows):
                row_idx = index

                addr = (index+r) * hc + i * num_channels + channel_idx
                addr += ifmap_base

                entry += str(int(addr)) + ", "
                pikachu_i += sram_read_pikachu
            for c in range(active_cols):
                col_index = index+active_rows

                addr = (col_index+c) * hc + i * num_channels + channel_idx
                addr += ifmap_base
                entry += str(int(addr)) + ", "
                pikachu_i += sram_read_pikachu



            cycle +=1
            entry += "\n"
            outfile.write(entry)

        outfile.close()

        return cycle, pikachu_i

def gen_trace_filter_partial(
                    cycle = 0,
                    h_fold = 0, v_fold = 0,
                    num_rows = 4, num_cols = 4,
                    num_filters = 4,
                    num_channels = 3,
                    filt_addr_list = [],
                    active_rows = 4,
                    active_cols = 4,
                    filter_base_addr = 10000000,
                    ofmap_base_addr = 20000000,
                    channel_idx = 0,
                    cnt_per_row = 10,
                    rc = 3,
                    current_filter = 0,
                    sram_read_trace_file = "sram_read.csv",
                    sram_read_pikachu = 6
):


    pikachu_f = 0

    local_cycles = cycle
    outfile = open(sram_read_trace_file, 'a')

    postfix = ""
    for r in range(num_rows):
        postfix += ", "
    
    this_filt_base_addr = filt_addr_list[current_filter]
    for i in range(cnt_per_row):
        entry = str(local_cycles) + ", "

        for r in range(active_rows):
            
            addr = this_filt_base_addr
            addr += r*rc + i*num_channels + channel_idx
            addr += filter_base_addr
            entry += str(int(addr)) + ", "
            pikachu_f += sram_read_pikachu

        local_cycles += 1
        entry += postfix
        entry += "\n"
        outfile.write(entry)
    outfile.close()

    return local_cycles, pikachu_f

def gen_trace_ofmap(
                    cycle = 0,
                    v_fold = 0, parallel_window = 1,
                    num_ofmap_this_fold = 4,
                    window_size = 4, num_filters = 4,
                    num_cols = 4, num_rows = 4,
                    ofmap_base = 20000000,
                    sram_write_trace_file = "sram_write.csv",
                    strides = 1,
                    num_channels = 3,
                    h_fold = 0,
                    total_ofmap_cols = 0,
                    total_ofmap_rows = 0,
                    active_rows = 4, active_cols = 4,
                    ew = 4,
                    currnet_filter = 0,
                    ef = 4,
                    fw = 11,
                    iw = 224,
                    dimension_cols = 4,
                    sram_read_pikachu = 6,
                    array_read_pikachu = 2,
                    rf_read_pikachu = 1
):
    outfile = open(sram_write_trace_file, 'a')

    pikachu_o = 0
    pikachu_calculate = 0
    pikachu_move = 0
    pikachu_sram_write = 0

    #사이클 계산 부분

    current_col = h_fold*dimension_cols
    start_col = current_col
    avg_col_idx = math.ceil(dimension_cols/strides)

    # print("h_fold: ", h_fold)

    # print("current_col: ", current_col)
    # print("start_col: ", start_col)
    # print("avg_col_idx: ", avg_col_idx)

    # exit()

    for i in range(ew):
        temp = 0
        entry = str(cycle) + ", "
        for j in range(active_cols):
            if((start_col+j)%strides != 0):
                entry += ", "
            else:
                of_row = h_fold*avg_col_idx + temp
                of_col = i
                of_depth = currnet_filter

                # print("of_row: ", of_row)
                # print("of_col: ", of_col)
                # print("of_depth: ", of_depth)

                addr = of_row*ef + of_col*num_filters + of_depth
                addr += ofmap_base

                entry += str(int(addr)) + ", "
                temp += 1
                #row 데이터 이동 (fillter 이동 ifmap 이동)
                # pikachu_move += (array_read_pikachu)*active_rows
                # pikachu_move += (array_read_pikachu)*active_rows
                # pikachu_move += (array_read_pikachu)*active_rows

                # #1D conv 계산
                # pikachu_calculate += (3*fw*(iw-fw+1)*rf_read_pikachu)*active_rows
                
                # #SRAM Write
                # pikachu_sram_write += sram_read_pikachu
        cycle += 1
        entry += "\n"
        outfile.write(entry)

    for j in range(active_cols):
        if((start_col+j)%strides != 0):
                entry += ", "
        pikachu_move += iw*(array_read_pikachu)*active_rows
        pikachu_move += fw*(array_read_pikachu)*active_rows
        pikachu_move += ew*(array_read_pikachu)*active_rows

        pikachu_calculate += fw*ew*active_rows*rf_read_pikachu
        pikachu_sram_write += sram_read_pikachu

    pikachu_move -= (array_read_pikachu)*(fw+iw+ew)

    pikachu_o_sram = pikachu_sram_write
    pikachu_o_array = pikachu_move
    pikachu_o_rf = pikachu_calculate
    pikachu_o = pikachu_calculate + pikachu_move + pikachu_sram_write
    outfile.close()

    # if(of_depth==2):
    #     print("cycle: ", cycle)
    #     print("of_row: ", of_row)
    #     print("of_col: ", of_col)
    #     print("of_depth: ", of_depth)
    #     exit()

    return cycle, total_ofmap_cols, pikachu_o, pikachu_o_sram, pikachu_o_array, pikachu_o_rf


if __name__ == "__main__":
    sram_traffic(
        rf_size=512,
        dimension_rows=12,
        dimension_cols=14,
        ifmap_h=224, ifmap_w=224,
        filt_h=11, filt_w=11,
        num_channels=3,
        strides=4, num_filt=96,
        ofmap_base=2000000, filt_base=1000000, ifmap_base=0,
        sram_read_trace_file="sram_read.csv",
        sram_write_trace_file="sram_write.csv"
    )