#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef long long big_int;

big_int abs_val(big_int x) {
    return x < 0 ? -x : x;
}

// Function to decide operation with palindromic consideration
big_int choose_operation(big_int current_sum, big_int target, big_int element, char *op) {
    big_int current_distance = abs_val(current_sum - target);
    big_int best_sum = current_sum;
    char best_op = '0';
    big_int best_distance = current_distance;

    // Try subtraction operations first
    // Regular subtraction
    big_int new_sum = current_sum - element;
    big_int new_distance = abs_val(new_sum - target);
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
    big_int target;
    int n;

    printf("Enter target sum: ");
    scanf("%lld", &target);

    printf("Enter number of elements: ");
    scanf("%d", &n);

    big_int *arr = malloc(n * sizeof(big_int));
    printf("Enter %d elements:\n", n);
    for (int i = 0; i < n; i++) {
        scanf("%lld", &arr[i]);
    }

    printf("\n=== CENTER-TO-LEFT ALGORITHM ===\n");
    printf("Target: %lld\n", target);
    printf("Elements: ");
    for (int i = 0; i < n; i++) {
        printf("%lld ", arr[i]);
    }
    printf("\n\n");

    // Center-to-left algorithm
    int center = (n % 2 == 0) ? (n/2 - 1) : (n/2);  // For even n, use left-center
    big_int current_sum = arr[center];
    char *ops = malloc(n * sizeof(char));

    // Initialize all as skipped
    for (int i = 0; i < n; i++) ops[i] = '0';
    ops[center] = '='; // starting point (will display as +)

    printf("Starting from center position %d with value %lld\n", center, arr[center]);
    printf("Initial sum: %lld\n\n", current_sum);

    // Process left elements (from center-1 down to 0)
    for (int i = center - 1; i >= 0; i--) {
        big_int old_sum = current_sum;
        big_int distance_before = abs_val(current_sum - target);

        current_sum = choose_operation(current_sum, target, arr[i], &ops[i]);
        big_int distance_after = abs_val(current_sum - target);

        printf("Position %d (value %lld):\n", i, arr[i]);
        printf("  Before: sum=%lld, distance=%lld\n", old_sum, distance_before);
        printf("  Operation: %c\n", ops[i]);

        // Handle palindromic operations
        if (ops[i] == 'S') { // palindromic subtraction
            int right_index = n - 1 - i;
            if (right_index < n && right_index != center) {
                ops[right_index] = '-';
                printf("  Palindromic: also set position %d to '-'\n", right_index);
            }
            ops[i] = '-'; // change 'S' to '-' for display
        }
        else if (ops[i] == 'A') { // palindromic addition
            int right_index = n - 1 - i;
            if (right_index < n && right_index != center) {
                ops[right_index] = '+';
                printf("  Palindromic: also set position %d to '+'\n", right_index);
            }
            ops[i] = '+'; // change 'A' to '+' for display
        }

        printf("  After: sum=%lld, distance=%lld\n", current_sum, distance_after);

        if (current_sum == target) {
            printf("  *** EXACT TARGET FOUND! ***\n");
            break;
        }
        printf("\n");
    }

    // Output results
    printf("\n=== FINAL RESULTS ===\n");
    printf("Target: %lld\n", target);
    printf("Final sum: %lld\n", current_sum);
    printf("Match: %s\n", (current_sum == target) ? "YES" : "NO");
    if (current_sum != target) {
        printf("Difference: %lld\n", current_sum - target);
    }

    printf("\nExpression: ");
    for (int i = 0; i < n; i++) {
        if (ops[i] == '0') {
            printf("0*%lld ", arr[i]);
        }
        else if (ops[i] == '=') {
            printf("+%lld ", arr[i]);
        }
        else {
            printf("%c%lld ", ops[i], arr[i]);
        }
    }
    printf("\n");

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

    // Detailed verification
    printf("\n=== VERIFICATION ===\n");
    big_int verification_sum = 0;
    for (int i = 0; i < n; i++) {
        big_int contribution = 0;
        if (ops[i] == '0') {
            contribution = 0;
        }
        else if (ops[i] == '=' || ops[i] == '+') {
            contribution = arr[i];
            verification_sum += arr[i];
        }
        else if (ops[i] == '-') {
            contribution = -arr[i];
            verification_sum -= arr[i];
        }

        if (contribution != 0) {
            printf("Position %d: %c%lld = %lld\n", i, (ops[i] == '=') ? '+' : ops[i], arr[i], contribution);
        }
    }

    printf("Verification sum: %lld\n", verification_sum);
    printf("Target sum: %lld\n", target);
    printf("Verification match: %s\n", (verification_sum == target) ? "YES" : "NO");

    // Cleanup
    free(arr);
    free(ops);

    return 0;
}
