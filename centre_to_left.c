#include <stdio.h>
#include <stdlib.h>
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

int main() {
    int target, n;
    printf("Enter target Fibonacci term: ");
    scanf("%d", &target);
    printf("Enter number of elements in Pascal triangle row: ");
    scanf("%d", &n);

    int arr[n];
    printf("Enter %d elements:\n", n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

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

    // Output expression
    printf("\nTarget = %d\n", target);
    printf("Expression: ");
    for (int i = 0; i < n; i++) {
        if (ops[i] == '0') {
            printf("0*%d ", arr[i]);
        }
        else if (ops[i] == '=') {
            printf("+%d ", arr[i]);
        }
        else {
            printf("%c%d ", ops[i], arr[i]);
        }
    }
    printf("\n");

    // Print choices line
    printf("Choices: ");
    for (int i = 0; i < n; i++) {
        if (ops[i] == '0') {
            printf("0 ");
        } else if (ops[i] == '+') {
            printf("+ ");
        } else if (ops[i] == '-') {
            printf("- ");
        } else if (ops[i] == '=') {
            printf("+ ");
        }
    }
    printf("\n");

    return 0;
}
