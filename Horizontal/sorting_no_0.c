#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 1024
#define MAX_RESULTS 1000

int count_zeros(const char *str) {
    int count = 0;
    while (*str) {
        if (*str == '0') count++;
        str++;
    }
    return count;
}

int main() {
    char filename[100];
    printf("Enter CSV filename to read: ");
    scanf("%s", filename);

    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Could not open file %s\n", filename);
        return 1;
    }

    char line[MAX_LINE];
    int min_zeros = 1000; // Arbitrary large initial value

    int steps[MAX_RESULTS];
    char choices[MAX_RESULTS][100];
    int result_count = 0;

    // Skip header
    fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file)) {
        char *step_str = strtok(line, ",");
        strtok(NULL, ",");  // Skip Sequence
        strtok(NULL, ",");  // Skip Sum
        char *choices_str = strtok(NULL, "\n");

        if (step_str && choices_str) {
            int step = atoi(step_str);
            int zero_count = count_zeros(choices_str);

            if (zero_count < min_zeros) {
                // New minimum found — reset
                min_zeros = zero_count;
                result_count = 0;
                steps[result_count] = step;
                strcpy(choices[result_count], choices_str);
                result_count++;
            } else if (zero_count == min_zeros) {
                // Same minimum — store too
                steps[result_count] = step;
                strcpy(choices[result_count], choices_str);
                result_count++;
            }
        }
    }

    fclose(file);

    if (result_count > 0) {
        printf("\nSteps with the least 0s (%d zero%s):\n",
               min_zeros, min_zeros == 1 ? "" : "s");
        for (int i = 0; i < result_count; i++) {
            printf("Step %d -> Choices: %s\n", steps[i], choices[i]);
        }
    } else {
        printf("No valid data found.\n");
    }

    return 0;
}
