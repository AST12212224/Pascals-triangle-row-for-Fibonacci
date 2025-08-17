#include <stdio.h>
#include <stdlib.h>

int abs_val(int x) {
    return x < 0 ? -x : x;
}

// Function to decide operation
int choose_operation(int current_sum, int target, int element, char *op) {
    // Try subtraction first
    int new_sum = current_sum - element;
    if (abs_val(new_sum - target) < abs_val(current_sum - target)) {
        *op = '-';
        return new_sum;
    }

    // Try addition next
    new_sum = current_sum + element;
    if (abs_val(new_sum - target) < abs_val(current_sum - target)) {
        *op = '+';
        return new_sum;
    }

    // Otherwise skip
    *op = '0';
    return current_sum;
}

int main() {
    int target, n;
    printf("Enter target Fibonacci term: ");
    scanf("%d", &target);

    printf("Enter number of elements in array: ");
    scanf("%d", &n);

    int arr[n];
    printf("Enter %d elements:\n", n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    int center = n / 2;
    int current_sum = arr[center];
    char ops[n];

    for (int i = 0; i < n; i++) ops[i] = '0'; // initialize as skipped
    ops[center] = '='; // starting point

    for (int i = center - 1; i >= 0; i--) {
        current_sum = choose_operation(current_sum, target, arr[i], &ops[i]);
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
            // print center element with a +
            printf("+%d ", arr[i]);
        }
        else {
            printf("%c%d ", ops[i], arr[i]);
        }
    }
    printf("= %d", current_sum);

    // Print choices line
    printf("\nChoices: ");
    for (int i = 0; i < n; i++) {
        if (ops[i] == '0') {
            printf("0 ");
        } else if (ops[i] == '+') {
            printf("+ ");
        } else if (ops[i] == '-') {
            printf("- ");
        } else if (ops[i] == '=') {
            printf("+ "); // treat center as "+"
        }
    }
    printf("\n");

    return 0;
}
