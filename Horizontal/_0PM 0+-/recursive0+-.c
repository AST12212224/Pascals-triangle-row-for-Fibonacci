#include <stdio.h>

#define MAX_N 12  // Adjust if needed

int n[MAX_N];  // Array to store input numbers
int n_size;    // Number of elements in the array
int x;         // Target sum
int step = 0;  // Step counter

// Recursive function to generate sequences
void generate(int index, int sum, int sequence[], char choices[][3]) {
    if (index == n_size) {
        step++; // Increment step counter
        if (sum == x) {  // Only print if sum matches x
            printf("Step %d -> [ ", step);
            for (int i = 0; i < n_size; i++) {
                printf("%d ", sequence[i]);
            }
            printf("] = %d, ", sum);
            for (int i = 0; i < n_size; i++) {
                printf("%s ", choices[i]); // Show the choices used
            }
            printf("\n");
        }
        return;
    }

    // Try multiplying current number by 0, 1, and -1
    int options[3] = {0, 1, -1};
    char option_labels[3][3] = {"0", "+", "-"}; // For displaying choices

    for (int i = 0; i < 3; i++) {
        sequence[index] = n[index] * options[i];
        sprintf(choices[index], "%s", option_labels[i]); // Store the choice as a string
        generate(index + 1, sum + sequence[index], sequence, choices);
    }
}

int main() {
    // User input
    printf("Enter number of elements (n): ");
    scanf("%d", &n_size);

    printf("Enter %d numbers: ", n_size);
    for (int i = 0; i < n_size; i++) {
        scanf("%d", &n[i]);
    }

    printf("Enter target sum (x): ");
    scanf("%d", &x);

    // Arrays to store current sequence and choices
    int sequence[MAX_N];
    char choices[MAX_N][3]; // Stores "+", "-", or "0" as strings

    printf("\nSequences that sum to %d:\n", x);
    generate(0, 0, sequence, choices);

    return 0;
}
