#include <stdint.h>
#include <hls_stream.h>

#define DATA_SIZE 128 
#define TOTAL_DATA_SIZE 16384 // 128 * 128
#define BLK_ROW 16 
#define A_BLK_NUM (DATA_SIZE / BLK_ROW)
#define B_BLK_NUM (DATA_SIZE / BLK_ROW)

typedef uint32_t buf_t;
const int c_size = TOTAL_DATA_SIZE;

static void load_input(uint32_t* in, hls::stream<uint32_t>& inStream, int size) {
mem_rd:
    for (int i = 0; i < size; i++) {
        #pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
        #pragma HLS PIPELINE II=1
        inStream << in[i];
    }
}

static void read_mats(hls::stream<uint32_t>& in1_stream,
                      hls::stream<uint32_t>& in2_stream,
                      uint32_t *a, 
                      uint32_t *b, 
                      int total_size){
read_mats_loop:
    for (int i = 0; i < total_size; i++) {
        #pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
        #pragma HLS PIPELINE II=1
        a[i] = in1_stream.read();
        b[i] = in2_stream.read();
    }
}

static void compute_mult(uint32_t *a,
                         uint32_t *b,
                         hls::stream<uint32_t>& out_stream) {
    
    buf_t Abuf[BLK_ROW][DATA_SIZE];
    buf_t Bbuf[DATA_SIZE][BLK_ROW];
    buf_t Cbuf[BLK_ROW][BLK_ROW];
    uint32_t c[TOTAL_DATA_SIZE];

    #pragma HLS ARRAY_PARTITION variable=Abuf complete dim=2
    #pragma HLS ARRAY_PARTITION variable=Bbuf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=Cbuf complete dim=0

    SWEEP_B_BLK:
    for (int i = 0; i < B_BLK_NUM; i++) {
        READ_B_BLOCK:
        for (int b_i = 0; b_i < DATA_SIZE; b_i++) {
            for (int b_j = 0; b_j < BLK_ROW; b_j++) {
                #pragma HLS PIPELINE II=1
                Bbuf[b_i][b_j] = b[b_i * DATA_SIZE + i * BLK_ROW + b_j];
            }
        }

        SWEEP_A_BLK:
        for (int j = 0; j < A_BLK_NUM; j++) {
            READ_A_BLK:
            for (int a_i = 0; a_i < BLK_ROW; a_i++) {
                for (int a_j = 0; a_j < DATA_SIZE; a_j++) {
                    #pragma HLS PIPELINE II=1
                    Abuf[a_i][a_j] = a[(j * BLK_ROW + a_i) * DATA_SIZE + a_j];
                }
            }

            COMPUTE_C_BLOCK:
            for (int c_i = 0; c_i < BLK_ROW; c_i++) {
                for (int c_j = 0; c_j < BLK_ROW; c_j++) {
                    #pragma HLS PIPELINE II=1
                    uint32_t c_tmp = 0;
                    
                    COMPUTE_C_ELE:
                    for (int c_k = 0; c_k < DATA_SIZE; c_k++) {
                        #pragma HLS UNROLL
                        c_tmp += Abuf[c_i][c_k] * Bbuf[c_k][c_j];
                    }
                    Cbuf[c_i][c_j] = c_tmp;
                }
            }

            WRITE_C_BLOCK_TO_LOCAL:
            for (int w_i = 0; w_i < BLK_ROW; w_i++) {
                for (int w_j = 0; w_j < BLK_ROW; w_j++) {
                    #pragma HLS PIPELINE II=1
                    int new_row = j * BLK_ROW + w_i;
                    int new_col = i * BLK_ROW + w_j;
                    c[new_row * DATA_SIZE + new_col] = Cbuf[w_i][w_j];
                }
            }
        }
    }

    WRITE_C_TO_STREAM:
    for (int k = 0; k < TOTAL_DATA_SIZE; k++) {
        #pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
        #pragma HLS PIPELINE II=1
        out_stream << c[k];
    }
}

static void store_result(uint32_t* out, hls::stream<uint32_t>& out_stream, int total_size) {
mem_wr:
    for (int i = 0; i < total_size; i++) {
        #pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
        #pragma HLS PIPELINE II=1
        out[i] = out_stream.read();
    }
}

extern "C" {

void krnl_mult(uint32_t* in1, uint32_t* in2, uint32_t* out, int size) {
    uint32_t a[TOTAL_DATA_SIZE];
    uint32_t b[TOTAL_DATA_SIZE];

    #pragma HLS INTERFACE m_axi port = in1 bundle = gmem0 depth = 16384 max_read_burst_length = 256
    #pragma HLS INTERFACE m_axi port = in2 bundle = gmem1 depth = 16384 max_read_burst_length = 256
    #pragma HLS INTERFACE m_axi port = out bundle = gmem0 depth = 16384 max_write_burst_length = 256
    
    #pragma HLS INTERFACE s_axilite port = in1 bundle = control
    #pragma HLS INTERFACE s_axilite port = in2 bundle = control
    #pragma HLS INTERFACE s_axilite port = out bundle = control
    #pragma HLS INTERFACE s_axilite port = size bundle = control
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    // Streams FIFO locais
    static hls::stream<uint32_t> in1_stream("input_stream_1");
    static hls::stream<uint32_t> in2_stream("input_stream_2");
    static hls::stream<uint32_t> out_stream("output_stream");

    #pragma HLS DATAFLOW
    
    load_input(in1, in1_stream, TOTAL_DATA_SIZE);
    load_input(in2, in2_stream, TOTAL_DATA_SIZE);
    
    read_mats(in1_stream, in2_stream, a, b, TOTAL_DATA_SIZE);
    
    compute_mult(a, b, out_stream);
    
    store_result(out, out_stream, TOTAL_DATA_SIZE);
}

}