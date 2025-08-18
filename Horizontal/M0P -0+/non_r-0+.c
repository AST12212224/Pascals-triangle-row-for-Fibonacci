#include <stdio.h>
#include <math.h>

#define MAX_N 12

int main() {
    int n[MAX_N];
    int n_size;
    int x;

    printf("Enter number of elements (n): ");
    scanf("%d", &n_size);

    printf("Enter %d numbers: ", n_size);
    for (int i = 0; i < n_size; i++) {
        scanf("%d", &n[i]);
    }

    printf("Enter target sum (x): ");
    scanf("%d", &x);

    int total_cases = pow(3, n_size);
    int step = 0;
    int useful_combinations = 0;

    printf("\nSequences that sum to %d:\n", x);

    for (int mask = 0; mask < total_cases; mask++) {
        int sum = 0;
        int sequence[MAX_N];
        char choices[MAX_N][3];

        int temp = mask;

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
            useful_combinations++;
            printf("Step %d -> [ ", step);
            for (int i = 0; i < n_size; i++) {
                printf("%d ", sequence[i]);
            }
            printf("] = %d, ", sum);
            for (int i = 0; i < n_size; i++) {
                printf("%s ", choices[i]);
            }
            printf("\n");
        }
    }

    printf("\nTotal combinations checked: %d\n", total_cases);
    printf("Total useful combinations (sum = %d): %d\n", x, useful_combinations);

    return 0;
}
