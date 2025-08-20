#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Optimized for limit 1000 - fixed but generous buffer sizes
#define MAX_FIBONACCI_DIGITS 500    // F(1000) needs ~209 digits
#define MAX_PASCAL_DIGITS 300       // Individual Pascal coefficient max
#define MAX_PASCAL_ROW_SIZE 500000  // Entire row of Pascal coefficients
#define MAX_LIMIT 1000              // Maximum supported limit

// Structure for big integers with fixed but optimal size
typedef struct {
    char digits[MAX_FIBONACCI_DIGITS];
    int length;
} BigInt;

// Initialize BigInt
void initBigInt(BigInt* num) {
    strcpy(num->digits, "0");
    num->length = 1;
}

// Set BigInt from integer
void setBigInt(BigInt* num, long long value) {
    if (value == 0) {
        initBigInt(num);
        return;
    }
    sprintf(num->digits, "%lld", value);
    num->length = strlen(num->digits);
}

// Copy BigInt
void copyBigInt(BigInt* dest, const BigInt* src) {
    strcpy(dest->digits, src->digits);
    dest->length = src->length;
}

// Add two BigInts
void addBigInt(BigInt* result, const BigInt* a, const BigInt* b) {
    int carry = 0;
    int maxLen = (a->length > b->length) ? a->length : b->length;
    char temp[MAX_FIBONACCI_DIGITS];
    int tempLen = 0;

    for (int i = 0; i < maxLen || carry; i++) {
        int sum = carry;

        if (i < a->length) {
            sum += a->digits[a->length - 1 - i] - '0';
        }
        if (i < b->length) {
            sum += b->digits[b->length - 1 - i] - '0';
        }

        temp[tempLen++] = (sum % 10) + '0';
        carry = sum / 10;
    }

    // Reverse result
    for (int i = 0; i < tempLen; i++) {
        result->digits[i] = temp[tempLen - 1 - i];
    }
    result->digits[tempLen] = '\0';
    result->length = tempLen;
}

// Multiply BigInt by small integer
void multiplyBigIntByLong(BigInt* result, const BigInt* a, long long multiplier) {
    if (multiplier == 0) {
        initBigInt(result);
        return;
    }
    if (multiplier == 1) {
        copyBigInt(result, a);
        return;
    }

    long long carry = 0;
    char temp[MAX_FIBONACCI_DIGITS];
    int tempLen = 0;

    for (int i = a->length - 1; i >= 0; i--) {
        long long prod = (a->digits[i] - '0') * multiplier + carry;
        temp[tempLen++] = (prod % 10) + '0';
        carry = prod / 10;
    }

    while (carry > 0) {
        temp[tempLen++] = (carry % 10) + '0';
        carry /= 10;
    }

    // Reverse result
    for (int i = 0; i < tempLen; i++) {
        result->digits[i] = temp[tempLen - 1 - i];
    }
    result->digits[tempLen] = '\0';
    result->length = tempLen;
}

// Divide BigInt by small integer
void divideBigIntByLong(BigInt* result, const BigInt* a, long long divisor) {
    if (divisor == 0 || a->length == 0) {
        initBigInt(result);
        return;
    }
    if (divisor == 1) {
        copyBigInt(result, a);
        return;
    }

    long long remainder = 0;
    char temp[MAX_FIBONACCI_DIGITS];
    int tempLen = 0;

    for (int i = 0; i < a->length; i++) {
        long long current = remainder * 10 + (a->digits[i] - '0');
        int quotient = current / divisor;

        if (quotient > 0 || tempLen > 0) {
            temp[tempLen++] = quotient + '0';
        }
        remainder = current % divisor;
    }

    if (tempLen == 0) {
        initBigInt(result);
    } else {
        strncpy(result->digits, temp, tempLen);
        result->digits[tempLen] = '\0';
        result->length = tempLen;
    }
}

// Calculate Fibonacci iteratively (most efficient for our range)
void calculateFibonacci(BigInt* result, int n) {
    if (n <= 0) {
        initBigInt(result);
        return;
    }
    if (n <= 2) {
        setBigInt(result, 1);
        return;
    }

    BigInt prev1, prev2, temp;
    setBigInt(&prev1, 1);  // F(1) = 1
    setBigInt(&prev2, 1);  // F(2) = 1

    for (int i = 3; i <= n; i++) {
        addBigInt(&temp, &prev1, &prev2);
        copyBigInt(&prev1, &prev2);
        copyBigInt(&prev2, &temp);
    }

    copyBigInt(result, &prev2);
}

// Calculate Pascal's triangle coefficient C(n,k)
void calculatePascalCoeff(BigInt* result, int n, int k) {
    if (k > n - k) k = n - k;  // Use symmetry
    if (k == 0 || k == n) {
        setBigInt(result, 1);
        return;
    }

    setBigInt(result, 1);

    // Use iterative approach: C(n,k) = C(n,k-1) * (n-k+1) / k
    for (int i = 1; i <= k; i++) {
        BigInt temp;
        multiplyBigIntByLong(&temp, result, n - i + 1);
        divideBigIntByLong(result, &temp, i);
    }
}

// Generate complete Pascal's triangle row
void generatePascalRow(char* rowStr, int n) {
    rowStr[0] = '\0';

    for (int k = 0; k <= n; k++) {
        BigInt coeff;
        calculatePascalCoeff(&coeff, n, k);

        if (k > 0) {
            strcat(rowStr, " ");
        }
        strcat(rowStr, coeff.digits);
    }
}

// Progress tracking structure
typedef struct {
    clock_t startTime;
    int totalRows;
    int currentRow;
} ProgressTracker;

void initProgress(ProgressTracker* progress, int totalRows) {
    progress->startTime = clock();
    progress->totalRows = totalRows;
    progress->currentRow = 0;
}

void updateProgress(ProgressTracker* progress, int row, const BigInt* fib) {
    progress->currentRow = row;

    if (row % 50 == 0 || row <= 10 || row == progress->totalRows) {
        clock_t currentTime = clock();
        double elapsed = ((double)(currentTime - progress->startTime)) / CLOCKS_PER_SEC;
        double rate = row / elapsed;
        double eta = (progress->totalRows - row) / rate;

        printf("Row %4d: F(%d)=%s (%d digits) | %.1f rows/sec | ETA: %.0fs\n",
               row, row, fib->digits, fib->length, rate, eta);
    }
}

// Memory usage estimation
void printMemoryEstimate(int limit) {
    printf("\n=== MEMORY ESTIMATION FOR LIMIT %d ===\n", limit);

    // Fibonacci estimation
    double fibDigits = limit * 0.209;  // log10(golden ratio)
    printf("Fibonacci F(%d): ~%.0f digits (%.1f KB)\n",
           limit, fibDigits, fibDigits / 1024.0);

    // Pascal row estimation
    double avgPascalDigits = limit * 0.15;  // Conservative estimate
    double pascalRowSize = (limit + 1) * avgPascalDigits;
    printf("Pascal row %d: ~%d coefficients, %.1f KB\n",
           limit, limit + 1, pascalRowSize / 1024.0);

    // Total memory per row
    double totalPerRow = (fibDigits + pascalRowSize) / 1024.0;
    printf("Memory per row: ~%.1f KB\n", totalPerRow);

    // Total file size estimate
    double totalFileSize = totalPerRow * limit / 1024.0;  // MB
    printf("Estimated file size: ~%.1f MB\n", totalFileSize);

    printf("========================================\n\n");
}

// Validation function
void runValidation() {
    printf("Running validation tests...\n");

    // Test known Fibonacci values
    BigInt fib;
    int fibTests[] = {1, 2, 3, 4, 5, 10, 15, 20};
    long long expected[] = {1, 1, 2, 3, 5, 55, 610, 6765};

    for (int i = 0; i < 8; i++) {
        calculateFibonacci(&fib, fibTests[i]);
        long long actual = atoll(fib.digits);
        if (actual == expected[i]) {
            printf("✓ F(%d) = %lld\n", fibTests[i], expected[i]);
        } else {
            printf("✗ F(%d) expected %lld, got %lld\n", fibTests[i], expected[i], actual);
        }
    }

    // Test Pascal coefficients
    printf("\nPascal's triangle validation (first 5 rows):\n");
    char row[1000];
    for (int i = 0; i < 5; i++) {
        generatePascalRow(row, i);
        printf("Row %d: %s\n", i, row);
    }

    printf("Validation completed.\n\n");
}

int main() {
    printf("Fibonacci Pascal Generator - Optimized for Limit 1000\n");
    printf("======================================================\n");

    // Get user input
    int limit;
    printf("Enter limit (1 to %d): ", MAX_LIMIT);
    scanf("%d", &limit);

    if (limit < 1 || limit > MAX_LIMIT) {
        printf("Error: Limit must be between 1 and %d\n", MAX_LIMIT);
        return 1;
    }

    // Show estimates
    printMemoryEstimate(limit);

    // Ask for validation
    char runTests;
    printf("Run validation tests? (y/n): ");
    scanf(" %c", &runTests);
    if (runTests == 'y' || runTests == 'Y') {
        runValidation();
    }

    // Confirm generation
    printf("Generate CSV file for limit %d? (y/n): ", limit);
    char confirm;
    scanf(" %c", &confirm);
    if (confirm != 'y' && confirm != 'Y') {
        printf("Generation cancelled.\n");
        return 0;
    }

    // Open output file
    FILE* file = fopen("fib_pascal_data.csv", "w");
    if (!file) {
        printf("Error: Cannot create output file!\n");
        return 1;
    }

    // Allocate memory for Pascal row
    char* pascalRow = malloc(MAX_PASCAL_ROW_SIZE);
    if (!pascalRow) {
        printf("Error: Memory allocation failed!\n");
        fclose(file);
        return 1;
    }

    // Write CSV header
    fprintf(file, "Pas Index,Fibonacci,Pascal's Triangle\n");

    // Initialize progress tracking
    ProgressTracker progress;
    initProgress(&progress, limit);

    printf("\nGenerating data...\n");
    printf("==================\n");

    // Generate data
    BigInt fib;
    for (int i = 1; i <= limit; i++) {
        // Write Pascal index
        fprintf(file, "%d,", i);

        // Calculate and write Fibonacci
        calculateFibonacci(&fib, i);
        fprintf(file, "%s,", fib.digits);

        // Calculate and write Pascal row
        generatePascalRow(pascalRow, i - 1);  // 0-indexed
        fprintf(file, "\"%s\"\n", pascalRow);

        // Update progress
        updateProgress(&progress, i, &fib);
    }

    // Cleanup
    free(pascalRow);
    fclose(file);

    // Final statistics
    clock_t endTime = clock();
    double totalTime = ((double)(endTime - progress.startTime)) / CLOCKS_PER_SEC;

    printf("\n✅ Generation completed successfully!\n");
    printf("======================================\n");
    printf("File: fib_pascal_data.csv\n");
    printf("Rows generated: %d\n", limit);
    printf("Total time: %.2f seconds\n", totalTime);
    printf("Average rate: %.1f rows/second\n", limit / totalTime);

    // File size check
    FILE* sizeCheck = fopen("fib_pascal_data.csv", "r");
    if (sizeCheck) {
        fseek(sizeCheck, 0, SEEK_END);
        long fileSize = ftell(sizeCheck);
        fclose(sizeCheck);
        printf("File size: %.2f MB\n", fileSize / (1024.0 * 1024.0));
    }

    return 0;
}
