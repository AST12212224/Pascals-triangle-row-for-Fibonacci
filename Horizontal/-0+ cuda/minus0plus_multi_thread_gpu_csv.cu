#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define MAX_N 24
#define MAX_RESULTS 10000000

struct SequenceResult {
    unsigned long long step;    // iteration/step number
    long long sequence[MAX_N];  // sequence of values (-n, 0, +n)
    char choices[MAX_N];        // choices (-, 0, +)
};

// GPU kernel
__global__ void bruteForceKernel(long long *n, int n_size, long long x,
                                 unsigned long long *d_count, SequenceResult *d_results) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // total cases = 3^n
    unsigned long long total_cases = 1;
    for (int i = 0; i < n_size; i++) total_cases *= 3ULL;
    if (idx >= total_cases) return;

    long long temp = idx;
    long long sum = 0;
    char choices[MAX_N];
    long long sequence[MAX_N];

    // Generate sequence for this idx
    for (int i = n_size - 1; i >= 0; i--) {
        int choice = temp % 3;
        temp /= 3;

        if (choice == 0) {
            sequence[i] = -n[i];
            choices[i] = '-';
            sum -= n[i];
        } else if (choice == 1) {
            sequence[i] = 0;
            choices[i] = '0';
        } else {
            sequence[i] = n[i];
            choices[i] = '+';
            sum += n[i];
        }
    }

    if (sum == x) {
        unsigned long long pos = atomicAdd(d_count, 1ULL);
        if (pos < MAX_RESULTS) {
            d_results[pos].step = idx + 1; // step index (1-based like CPU code)
            for (int i = 0; i < n_size; i++) {
                d_results[pos].sequence[i] = sequence[i];
                d_results[pos].choices[i] = choices[i];
            }
        }
    }
}

int main() {
    int n_size;
    long long n[MAX_N], x;
    char filename[200];

    printf("Enter number of elements (n <= %d): ", MAX_N);
    scanf("%d", &n_size);
    if (n_size > MAX_N) {
        printf("n too large, max allowed is %d\n", MAX_N);
        return 1;
    }

    printf("Enter %d numbers: ", n_size);
    for (int i = 0; i < n_size; i++) scanf("%lld", &n[i]);

    printf("Enter target sum (x): ");
    scanf("%lld", &x);

    printf("Enter CSV filename (no leading -): ");
    scanf("%s", filename);

    // compute total cases = 3^n
    unsigned long long total_cases = 1;
    for (int i = 0; i < n_size; i++) total_cases *= 3ULL;

    // Allocate device memory
    long long *d_n;
    cudaMalloc(&d_n, sizeof(long long) * n_size);
    cudaMemcpy(d_n, n, sizeof(long long) * n_size, cudaMemcpyHostToDevice);

    unsigned long long *d_count;
    cudaMalloc(&d_count, sizeof(unsigned long long));
    cudaMemset(d_count, 0, sizeof(unsigned long long));

    SequenceResult *d_results;
    cudaMalloc(&d_results, sizeof(SequenceResult) * MAX_RESULTS);

    // Launch kernel
    int threadsPerBlock = 256;
    unsigned long long blocks = (total_cases + threadsPerBlock - 1) / threadsPerBlock;

    bruteForceKernel<<<blocks, threadsPerBlock>>>(d_n, n_size, x, d_count, d_results);
    cudaDeviceSynchronize();

    // Copy results back
    unsigned long long useful_count;
    cudaMemcpy(&useful_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    SequenceResult *h_results = (SequenceResult*)malloc(sizeof(SequenceResult) *
                                (useful_count > MAX_RESULTS ? MAX_RESULTS : useful_count));
    cudaMemcpy(h_results, d_results,
               sizeof(SequenceResult) * (useful_count > MAX_RESULTS ? MAX_RESULTS : useful_count),
               cudaMemcpyDeviceToHost);

    // Save CSV
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error opening file\n");
        return 1;
    }

    fprintf(file, "Step,Sequence,Choices,Sum\n");
    for (unsigned long long i = 0; i < (useful_count > MAX_RESULTS ? MAX_RESULTS : useful_count); i++) {
        fprintf(file, "%llu,\"[", h_results[i].step);
        for (int j = 0; j < n_size; j++) {
            fprintf(file, "%lld ", h_results[i].sequence[j]);
        }
        fprintf(file, "]\",\"");
        for (int j = 0; j < n_size; j++) {
            fprintf(file, "%c", h_results[i].choices[j]);
        }
        fprintf(file, "\",%lld\n", x);
    }
    fclose(file);

    printf("Total combinations tried: %llu\n", total_cases);
    printf("Useful combinations found: %llu\n", useful_count);
    printf("Results saved to %s\n", filename);

    // Free memory
    free(h_results);
    cudaFree(d_n);
    cudaFree(d_count);
    cudaFree(d_results);

    return 0;
}

