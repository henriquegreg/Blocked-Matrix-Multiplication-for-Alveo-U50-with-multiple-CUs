#include <stdint.h>
#include <omp.h> // Incluído para suporte a funções do OpenMP, se necessário

#define CPU_BLOCK 64
#define NUM_THREADS 4

void matmul_large_cpu(uint32_t* A, uint32_t* B, uint32_t* C, int N) {
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (int i = 0; i < N * N; i++) {
        C[i] = 0;
    }

    #pragma omp parallel for num_threads(NUM_THREADS) shared(A, B, C, N) schedule(dynamic)
    for (int i_b = 0; i_b < N; i_b += CPU_BLOCK) {
        for (int k_b = 0; k_b < N; k_b += CPU_BLOCK) {
            for (int j_b = 0; j_b < N; j_b += CPU_BLOCK) {
                
                for (int i = i_b; i < i_b + CPU_BLOCK && i < N; i++) {
                    for (int k = k_b; k < k_b + CPU_BLOCK && k < N; k++) {
                        
                        uint32_t a_ik = A[i * N + k]; 
                        
                        for (int j = j_b; j < j_b + CPU_BLOCK && j < N; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
                
            }
        }
    }
}