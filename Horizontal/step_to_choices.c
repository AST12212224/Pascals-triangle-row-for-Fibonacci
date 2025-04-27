#include <stdio.h>
#include <math.h>

void get_choices_from_step(int step, int n) {
    char operations[3] = {'0', '+', '-'};
    int ternary[20];

    for (int i = 0; i < n; i++) {
        ternary[i] = 0;
    }

    int temp = step;
    for (int i = n - 1; i >= 0; i--) {
        ternary[i] = temp % 3;
        temp /= 3;
    }

    printf("Step %d -> Choices: ", step);
    for (int i = 0; i < n; i++) {
        printf("%c ", operations[ternary[i]]);
    }
    printf("\n");
}

int main() {
    int n, step;

    printf("Enter number of elements (n): ");
    scanf("%d", &n);

    int total_steps = pow(3, n);
    printf("Total combinations: %d (valid step: 0 to %d)\n", total_steps, total_steps - 1);

    printf("Enter step number: ");
    scanf("%d", &step);

    if (step < 0 || step >= total_steps) {
        printf("Invalid step. It must be between 0 and %d\n", total_steps - 1);
        return 1;
    }

    get_choices_from_step(step, n);

    return 0;
}
