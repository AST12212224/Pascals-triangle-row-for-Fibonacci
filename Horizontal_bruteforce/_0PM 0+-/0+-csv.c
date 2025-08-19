#include <stdio.h>
#include <math.h>

#define MAX_N 20  // Maximum allowed elements

int main() {
    int n[MAX_N];  // Array to store input numbers
    int n_size;    // Number of elements in the array
    int x;         // Target sum
    char filename[100]; // CSV filename

    // Ask user for filename
    printf("Enter the filename to save results (e.g., output.csv): ");
    scanf("%s", filename);

    // Open CSV file
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Could not create file.\n");
        return 1;
    }

    // User input
    printf("Enter number of elements (n): ");
    scanf("%d", &n_size);

    printf("Enter %d numbers: ", n_size);
    for (int i = 0; i < n_size; i++) {
        scanf("%d", &n[i]);
    }

    printf("Enter target sum (x): ");
    scanf("%d", &x);

    int total_cases = pow(3, n_size);  // 3^n possible sequences (0, -1, +1)
    int step = 0;  // Global step counter (counts all cases)

    // Write CSV header
    fprintf(file, "Step,Sequence,Sum,Choices\n");

    // Iterate through all possible sequences
    for (int mask = 0; mask < total_cases; mask++) {
        int sum = 0;
        int sequence[MAX_N];
        char choices[MAX_N][3];

        int temp = mask;

        // Reverse the order in which choices are applied (to match recursion)
        for (int i = n_size - 1; i >= 0; i--) {
            int choice = temp % 3;  // Extract choice for this index
            temp /= 3;

            if (choice == 0) {
                sequence[i] = 0;
                sprintf(choices[i], "0");
            } else if (choice == 1) {
                sequence[i] = n[i];
                sprintf(choices[i], "+");
            } else {
                sequence[i] = -n[i];
                sprintf(choices[i], "-");
            }

            sum += sequence[i];
        }

        step++;  // Step increments for EVERY case

        if (sum == x) {
            // Write to CSV file
            fprintf(file, "%d,\"[ ", step);
            for (int i = 0; i < n_size; i++) {
                fprintf(file, "%d ", sequence[i]);
            }
            fprintf(file, "]\",%d,\"", sum);
            for (int i = 0; i < n_size; i++) {
                fprintf(file, "%s ", choices[i]); // Show the choices used
            }
            fprintf(file, "\"\n");
        }
    }

    fclose(file);
    printf("Results saved to %s\n", filename);

    return 0;
}
