#include <stdio.h>

#define MAX_N 20  // Max elements (safe for 3^20 = ~3.5 billion combinations)

// Integer power function to avoid floating point
long long int_pow(int base, int exp) {
    long long result = 1;
    for (int i = 0; i < exp; i++) {
        result *= base;
    }
    return result;
}

int main() {
    long long n[MAX_N];     // Handle large input values
    int n_size;             // Number of input values
    long long x;            // Target sum
    char filename[100];     // CSV filename

    printf("Enter the filename to save results (e.g., output.csv): ");
    scanf("%s", filename);

    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Could not create file.\n");
        return 1;
    }

    // Input size and values
    printf("Enter number of elements (n): ");
    scanf("%d", &n_size);

    if (n_size > MAX_N) {
        printf("Too many elements. Max allowed is %d\n", MAX_N);
        return 1;
    }

    printf("Enter %d numbers: ", n_size);
    for (int i = 0; i < n_size; i++) {
        scanf("%lld", &n[i]);
    }

    printf("Enter target sum (x): ");
    scanf("%lld", &x);

    long long total_cases = int_pow(3, n_size);  // 3^n
    long long step = 0;
    long long useful_count = 0;

    fprintf(file, "Step,Sequence,Sum,Choices\n");

    for (long long mask = 0; mask < total_cases; mask++) {
        long long sum = 0;
        long long sequence[MAX_N];
        char choices[MAX_N][3];

        long long temp = mask;

        // Apply choices in -, 0, + order
        for (int i = n_size - 1; i >= 0; i--) {
            int choice = temp % 3;
            temp /= 3;

            if (choice == 0) {
                sequence[i] = -n[i];
                sprintf(choices[i], "-");
            } else if (choice == 1) {
                sequence[i] = 0;
                sprintf(choices[i], "0");
            } else {
                sequence[i] = n[i];
                sprintf(choices[i], "+");
            }

            sum += sequence[i];
        }

        step++;

        if (sum == x) {
            useful_count++;

            // Save to file
            fprintf(file, "%lld,\"[ ", step);
            for (int i = 0; i < n_size; i++) {
                fprintf(file, "%lld ", sequence[i]);
            }
            fprintf(file, "]\",%lld,\"", sum);
            for (int i = 0; i < n_size; i++) {
                fprintf(file, "%s ", choices[i]);
            }
            fprintf(file, "\"\n");
        }
    }

    fclose(file);

    // Final stats
    printf("Results saved to %s\n", filename);
    printf("Total combinations tried: %lld\n", total_cases);
    printf("Useful combinations that matched target sum: %lld\n", useful_count);

    return 0;
}
