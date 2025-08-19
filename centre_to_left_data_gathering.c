#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int abs_val(int x) {
    return x < 0 ? -x : x;
}

// Function to decide operation with palindromic consideration
int choose_operation(int current_sum, int target, int element, char *op) {
    int current_distance = abs_val(current_sum - target);
    int best_sum = current_sum;
    char best_op = '0';
    int best_distance = current_distance;

    // Try subtraction operations first
    // Regular subtraction
    int new_sum = current_sum - element;
    int new_distance = abs_val(new_sum - target);
    if (new_distance < best_distance) {
        best_sum = new_sum;
        best_op = '-';
        best_distance = new_distance;
    }

    // Palindromic subtraction (subtract 2*element)
    new_sum = current_sum - 2 * element;
    new_distance = abs_val(new_sum - target);
    if (new_distance < best_distance) {
        best_sum = new_sum;
        best_op = 'S'; // 'S' for palindromic subtraction
        best_distance = new_distance;
    }

    // If subtraction helped, don't try addition
    if (best_distance < current_distance) {
        *op = best_op;
        return best_sum;
    }

    // Try addition operations if subtraction didn't help
    // Regular addition
    new_sum = current_sum + element;
    new_distance = abs_val(new_sum - target);
    if (new_distance < best_distance) {
        best_sum = new_sum;
        best_op = '+';
        best_distance = new_distance;
    }

    // Palindromic addition (add 2*element)
    new_sum = current_sum + 2 * element;
    new_distance = abs_val(new_sum - target);
    if (new_distance < best_distance) {
        best_sum = new_sum;
        best_op = 'A'; // 'A' for palindromic addition
        best_distance = new_distance;
    }

    *op = best_op;
    return best_sum;
}

// Function to solve single Pascal-Fibonacci problem
void solve_pascal_fibonacci(int target, int n, int *arr, char *expression, char *choices) {
    int center = ceil(n / 2.0) - 1;
    int current_sum = arr[center];
    char ops[n];

    // Initialize all as skipped
    for (int i = 0; i < n; i++) ops[i] = '0';
    ops[center] = '='; // starting point

    // Process left elements (from center-1 down to 0)
    for (int i = center - 1; i >= 0; i--) {
        current_sum = choose_operation(current_sum, target, arr[i], &ops[i]);

        // If we used palindromic operation, mark the corresponding right element
        if (ops[i] == 'S') { // palindromic subtraction
            int right_index = n - 1 - i;
            if (right_index < n && right_index != center) {
                ops[right_index] = '-';
            }
            ops[i] = '-'; // change 'S' to '-' for display
        }
        else if (ops[i] == 'A') { // palindromic addition
            int right_index = n - 1 - i;
            if (right_index < n && right_index != center) {
                ops[right_index] = '+';
            }
            ops[i] = '+'; // change 'A' to '+' for display
        }

        if (current_sum == target) break;
    }

    // Build expression string
    strcpy(expression, "");
    strcpy(choices, "");

    for (int i = 0; i < n; i++) {
        char temp[20];
        if (ops[i] == '0') {
            sprintf(temp, "0*%d ", arr[i]);
            strcat(expression, temp);
            strcat(choices, "0 ");
        }
        else if (ops[i] == '=') {
            sprintf(temp, "+%d ", arr[i]);
            strcat(expression, temp);
            strcat(choices, "+ ");
        }
        else {
            sprintf(temp, "%c%d ", ops[i], arr[i]);
            strcat(expression, temp);
            if (ops[i] == '+') {
                strcat(choices, "+ ");
            } else {
                strcat(choices, "- ");
            }
        }
    }
}

// Function to parse Pascal triangle row from string
int parse_pascal_row(char *pascal_str, int *arr) {
    int count = 0;
    char *token = strtok(pascal_str, " ");
    while (token != NULL) {
        arr[count++] = atoi(token);
        token = strtok(NULL, " ");
    }
    return count;
}

int main() {
    FILE *input_file = fopen("fibonacci_pascals.csv", "r");
    if (input_file == NULL) {
        printf("Error: Cannot open fibonacci_pascals.csv\n");
        return 1;
    }

    FILE *output_file = fopen("centre_to_left.csv", "w");
    if (output_file == NULL) {
        printf("Error: Cannot create centre_to_left.csv\n");
        fclose(input_file);
        return 1;
    }

    // Write CSV header
    fprintf(output_file, "Pas Index,Fibonacci,Expression,Choices\n");

    char line[1000];
    int first_line = 1;

    // Skip header line
    if (fgets(line, sizeof(line), input_file) != NULL) {
        // Header skipped
    }

    while (fgets(line, sizeof(line), input_file)) {
        // Remove newline character
        line[strcspn(line, "\n")] = 0;

        // Parse CSV line
        char *fib_index_str = strtok(line, ",");
        char *pas_index_str = strtok(NULL, ",");
        char *fibonacci_str = strtok(NULL, ",");
        char *pascal_str = strtok(NULL, ",");

        if (fib_index_str && pas_index_str && fibonacci_str && pascal_str) {
            int fib_index = atoi(fib_index_str);
            int pas_index = atoi(pas_index_str);
            int fibonacci = atoi(fibonacci_str);

            // Parse Pascal triangle row
            int pascal_arr[100]; // Assuming max 100 elements
            char pascal_copy[500];
            strcpy(pascal_copy, pascal_str);
            int n = parse_pascal_row(pascal_copy, pascal_arr);

            // Solve the problem
            char expression[1000];
            char choices[200];
            solve_pascal_fibonacci(fibonacci, n, pascal_arr, expression, choices);

            // Calculate final sum for verification
            // Removed final_sum calculation as it's not needed for output

            // Write to output CSV
            fprintf(output_file, "%d,%d,\"%s\",\"%s\"\n",
                    pas_index, fibonacci, expression, choices);

            printf("Processed: Fib=%d, PasIndex=%d, Target=%d\n", fibonacci, pas_index, fibonacci);
        }
    }

    fclose(input_file);
    fclose(output_file);

    printf("Processing completed! Results saved in centre_to_left.csv\n");
    return 0;
}
