#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Constants
#define MAX_LIMIT 10000
#define MAX_DIGITS 5000
#define BLOCK_SIZE 256

// Structure for big integers (host-side)
typedef struct {
    char digits[MAX_DIGITS];
    int length;
} BigInt;

// Initialize BigInt
void initBigInt(BigInt* num) {
    strcpy(num->digits, "0");
    num->length = 1;
}

// Convert double to BigInt (for GPU results)
void doubleToBigInt(BigInt* result, double value) {
    if (value < 1e15) {
        sprintf(result->digits, "%.0f", value);
    } else {
        sprintf(result->digits, "%.3e", value);
    }
    result->length = strlen(result->digits);
}

// Matrix structure for 2x2 matrices
typedef struct {
    double a, b, c, d;  // [[a,b],[c,d]]
} Matrix2x2;

// Device function for matrix multiplication
__device__ Matrix2x2 matrixMultiply(Matrix2x2 A, Matrix2x2 B) {
    Matrix2x2 result;
    result.a = A.a * B.a + A.b * B.c;
    result.b = A.a * B.b + A.b * B.d;
    result.c = A.c * B.a + A.d * B.c;
    result.d = A.c * B.b + A.d * B.d;
    return result;
}

// Device function for matrix power
__device__ Matrix2x2 matrixPower(Matrix2x2 base, int exp) {
    Matrix2x2 result = {1.0, 0.0, 0.0, 1.0};  // Identity matrix

    while (exp > 0) {
        if (exp & 1) {
            result = matrixMultiply(result, base);
        }
        base = matrixMultiply(base, base);
        exp >>= 1;
    }
    return result;
}

// CUDA kernel for Fibonacci computation using matrix exponentiation
__global__ void fibonacciKernel(double* results, int* indices, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int n = indices[idx];
    if (n <= 0) {
        results[idx] = 0.0;
        return;
    }
    if (n <= 2) {
        results[idx] = 1.0;
        return;
    }

    // Fibonacci base matrix: [[1,1],[1,0]]
    Matrix2x2 fibMatrix = {1.0, 1.0, 1.0, 0.0};
    Matrix2x2 result = matrixPower(fibMatrix, n - 1);

    // F(n) is the top-left element
    results[idx] = result.a;
}

// CUDA kernel for computing Pascal coefficient terms
__global__ void computePascalTerms(double* terms, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k > n) return;

    if (k == 0) {
        terms[k] = 1.0;
    } else {
        terms[k] = (double)(n - k + 1) / (double)k;
    }
}

// CUDA kernel for parallel prefix product (simple version)
__global__ void prefixProduct(double* data, double* temp, int n, int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (idx >= step) {
        temp[idx] = data[idx] * data[idx - step];
    } else {
        temp[idx] = data[idx];
    }
}

// Host function to compute Pascal's triangle row using GPU
void computePascalRowGPU(double* hostCoeffs, int n) {
    if (n == 0) {
        hostCoeffs[0] = 1.0;
        return;
    }

    size_t size = (n + 1) * sizeof(double);

    // Allocate device memory
    double *d_terms, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_terms, size));
    CUDA_CHECK(cudaMalloc(&d_temp, size));

    // Initialize terms
    int gridSize = (n + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    computePascalTerms<<<gridSize, BLOCK_SIZE>>>(d_terms, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Parallel prefix product using iterative approach
    for (int step = 1; step <= n; step *= 2) {
        prefixProduct<<<gridSize, BLOCK_SIZE>>>(d_terms, d_temp, n + 1, step);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap pointers
        double* swap = d_terms;
        d_terms = d_temp;
        d_temp = swap;
    }

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(hostCoeffs, d_terms, size, cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_terms));
    CUDA_CHECK(cudaFree(d_temp));
}

// Convert double array to string representation
void pascalRowToString(char* rowStr, double* coeffs, int n) {
    rowStr[0] = '\0';

    for (int k = 0; k <= n; k++) {
        char coeffStr[50];
        if (coeffs[k] < 1e15) {
            sprintf(coeffStr, "%.0f", coeffs[k]);
        } else {
            sprintf(coeffStr, "%.3e", coeffs[k]);
        }

        if (k > 0) strcat(rowStr, " ");
        strcat(rowStr, coeffStr);
    }
}

// Progress tracking
typedef struct {
    clock_t startTime;
    int totalRows;
} ProgressTracker;

void updateProgress(ProgressTracker* progress, int currentRow, double fibValue) {
    if (currentRow % 100 == 0 || currentRow <= 10) {
        clock_t currentTime = clock();
        double elapsed = ((double)(currentTime - progress->startTime)) / CLOCKS_PER_SEC;
        double rate = (elapsed > 0) ? currentRow / elapsed : 0;
        double eta = (rate > 0) ? (progress->totalRows - currentRow) / rate : 0;

        printf("Row %4d: F(%d)=%.3e | %.1f rows/sec | ETA: %.0fs\n",
               currentRow, currentRow, fibValue, rate, eta);
    }
}

// Batch Fibonacci computation
void computeFibonacciBatch(double* results, int start, int count) {
    // Prepare indices
    int* h_indices = (int*)malloc(count * sizeof(int));
    for (int i = 0; i < count; i++) {
        h_indices[i] = start + i;
    }

    // Allocate device memory
    int* d_indices;
    double* d_results;
    CUDA_CHECK(cudaMalloc(&d_indices, count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_results, count * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, count * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fibonacciKernel<<<gridSize, BLOCK_SIZE>>>(d_results, d_indices, count);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(results, d_results, count * sizeof(double), cudaMemcpyDeviceToHost));

    // Cleanup
    free(h_indices);
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_results));
}

// Main computation function
void generateDataGPU(FILE* file, int limit) {
    printf("Initializing GPU...\n");

    // Get GPU properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s\n", prop.name);
    printf("Global memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    ProgressTracker progress;
    progress.startTime = clock();
    progress.totalRows = limit;

    // Batch processing
    int batchSize = 1000;
    printf("\nGenerating data in batches of %d...\n", batchSize);

    for (int batchStart = 1; batchStart <= limit; batchStart += batchSize) {
        int batchEnd = (batchStart + batchSize - 1 > limit) ? limit : batchStart + batchSize - 1;
        int currentBatchSize = batchEnd - batchStart + 1;

        // Compute Fibonacci batch
        double* fibResults = (double*)malloc(currentBatchSize * sizeof(double));
        computeFibonacciBatch(fibResults, batchStart, currentBatchSize);

        // Process each row in the batch
        for (int i = 0; i < currentBatchSize; i++) {
            int rowIndex = batchStart + i;

            // Write Pascal index
            fprintf(file, "%d,", rowIndex);

            // Write Fibonacci number
            BigInt fibBig;
            doubleToBigInt(&fibBig, fibResults[i]);
            fprintf(file, "%s,", fibBig.digits);

            // Compute and write Pascal row
            double* pascalCoeffs = (double*)malloc(rowIndex * sizeof(double));
            computePascalRowGPU(pascalCoeffs, rowIndex - 1);

            char* pascalRowStr = (char*)malloc(100000);
            pascalRowToString(pascalRowStr, pascalCoeffs, rowIndex - 1);
            fprintf(file, "\"%s\"\n", pascalRowStr);

            // Update progress
            updateProgress(&progress, rowIndex, fibResults[i]);

            free(pascalCoeffs);
            free(pascalRowStr);
        }

        free(fibResults);
        printf("Batch %d-%d completed\n", batchStart, batchEnd);
    }
}

// Check CUDA setup
int checkCUDA() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return 0;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 0;
    }

    printf("Found %d CUDA device(s)\n", deviceCount);
    return 1;
}

int main() {
    printf("CUDA GPU-Accelerated Fibonacci Pascal Generator\n");
    printf("===============================================\n");

    // Check CUDA
    if (!checkCUDA()) {
        printf("CUDA not available. Please install CUDA drivers.\n");
        return 1;
    }

    // Get user input with proper error checking
    int limit;
    printf("\nEnter limit (1 to %d): ", MAX_LIMIT);
    if (scanf("%d", &limit) != 1) {
        printf("Invalid input!\n");
        return 1;
    }

    if (limit < 1 || limit > MAX_LIMIT) {
        printf("Error: Limit must be between 1 and %d\n", MAX_LIMIT);
        return 1;
    }

    // Resource estimation
    printf("\n=== GPU RESOURCE ESTIMATION ===\n");
    printf("Fibonacci computations: %d\n", limit);
    printf("Pascal rows: %d\n", limit);
    size_t estimatedMemory = (size_t)limit * 1000 * sizeof(double);
    printf("Estimated GPU memory: %.1f MB\n", estimatedMemory / (1024.0 * 1024.0));
    printf("==============================\n\n");

    char confirm;
    printf("Start GPU computation? (y/n): ");
    if (scanf(" %c", &confirm) != 1) {
        printf("Invalid input!\n");
        return 1;
    }

    if (confirm != 'y' && confirm != 'Y') {
        printf("Cancelled.\n");
        return 0;
    }

    // Open output file
    FILE* file = fopen("fib_pascal_data_gpu.csv", "w");
    if (!file) {
        printf("Error: Cannot create output file!\n");
        return 1;
    }

    // Write CSV header
    fprintf(file, "Pas Index,Fibonacci,Pascal's Triangle\n");

    // Generate data using GPU
    clock_t startTime = clock();
    generateDataGPU(file, limit);
    clock_t endTime = clock();

    fclose(file);

    // Final statistics
    double totalTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
    printf("\n✅ GPU computation completed!\n");
    printf("================================\n");
    printf("File: fib_pascal_data_gpu.csv\n");
    printf("Rows generated: %d\n", limit);
    printf("Total time: %.2f seconds\n", totalTime);
    if (totalTime > 0) {
        printf("Average rate: %.1f rows/second\n", limit / totalTime);
    }

    return 0;
}
