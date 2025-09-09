#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#define MAX_LINE_SIZE 50000  // Increased for large Pascal rows
#define MAX_CHOICES_SIZE 5000 // Increased buffer
#define MAX_ELEMENTS 200     // Reasonable limit to prevent overflow

// Function to extract choices field from CSV line
char* extract_choices_from_csv(char* line) {
    static char result[MAX_CHOICES_SIZE];

    // Find the last quoted field (which should be choices)
    char* last_quote_start = NULL;
    char* last_quote_end = NULL;

    // Find all quote pairs and take the last one
    char* ptr = line;
    while (*ptr) {
        if (*ptr == '"') {
            char* quote_start = ptr + 1;
            ptr++;

            // Find the closing quote
            while (*ptr && *ptr != '"') {
                ptr++;
            }

            if (*ptr == '"') {
                last_quote_start = quote_start;
                last_quote_end = ptr;
            }
        }
        if (*ptr) ptr++;
    }

    if (last_quote_start && last_quote_end && last_quote_end > last_quote_start) {
        int len = last_quote_end - last_quote_start;
        if (len < MAX_CHOICES_SIZE - 1) {
            strncpy(result, last_quote_start, len);
            result[len] = '\0';
            return result;
        }
    }

    return NULL;
}

// Function to convert decimal to choices using enumeration order (-0+)
void decimalToChoices(long long decimal_number, int array_length, char* choices_result) {
    // Convert decimal to base-3 representation for enumeration
    long long temp = decimal_number;
    int digits[array_length];

    // Convert to base-3 digits (0, 1, 2)
    for (int i = array_length - 1; i >= 0; i--) {
        digits[i] = temp % 3;
        temp /= 3;
    }

    // Convert digits to choices using -0+ mapping
    // 0 -> -, 1 -> 0, 2 -> +
    strcpy(choices_result, "");
    for (int i = 0; i < array_length; i++) {
        if (i > 0) strcat(choices_result, " ");

        if (digits[i] == 0) {
            strcat(choices_result, "-");
        } else if (digits[i] == 1) {
            strcat(choices_result, "0");
        } else { // digits[i] == 2
            strcat(choices_result, "+");
        }
    }
}

// Function to convert choices to decimal using enumeration order (-0+)
long long choicesToDecimal(char* choices_str, int expected_length) {
    // Safety check for array size
    if (expected_length > MAX_ELEMENTS) {
        printf("Warning: Array too large (%d elements), skipping\n", expected_length);
        return -1;
    }

    char* token;
    char choices_copy[MAX_CHOICES_SIZE];
    strncpy(choices_copy, choices_str, MAX_CHOICES_SIZE - 1);
    choices_copy[MAX_CHOICES_SIZE - 1] = '\0';

    long long decimal = 0;
    int position = 0;

    token = strtok(choices_copy, " ");

    while (token != NULL && position < expected_length) {
        int digit_value;
        if (strcmp(token, "-") == 0) {
            digit_value = 0;  // - maps to 0
        } else if (strcmp(token, "0") == 0) {
            digit_value = 1;  // 0 maps to 1
        } else if (strcmp(token, "+") == 0) {
            digit_value = 2;  // + maps to 2
        } else {
            printf("Warning: Invalid choice token '%s' at position %d\n", token, position);
            return -1; // Invalid data
        }

        // Calculate contribution to decimal (avoid overflow)
        long long power = 1;
        for (int i = 0; i < (expected_length - position - 1); i++) {
            if (power > LLONG_MAX / 3) {
                printf("Warning: Number too large for PasIndex=%d, skipping\n", expected_length);
                return -1;
            }
            power *= 3;
        }

        decimal += digit_value * power;
        position++;
        token = strtok(NULL, " ");
    }

    if (position != expected_length) {
        printf("Warning: Expected %d choices, got %d\n", expected_length, position);
        return -1;
    }

    return decimal;
}

int main() {
    FILE *input_file = fopen("centre_to_left_enhanced.csv", "r");
    if (input_file == NULL) {
        printf("Error: Cannot open centre_to_left_enhanced.csv\n");
        printf("Please ensure the file exists and was generated correctly.\n");
        return 1;
    }

    FILE *output_file = fopen("symbol_choices_conversion_to_decimal.csv", "w");
    if (output_file == NULL) {
        printf("Error: Cannot create symbol_choices_conversion_to_decimal.csv\n");
        fclose(input_file);
        return 1;
    }

    // Write CSV header
    fprintf(output_file, "Pas Index,Decimal Number\n");

    char line[MAX_LINE_SIZE];
    int processed_count = 0;
    int error_count = 0;

    // Skip header line
    if (fgets(line, sizeof(line), input_file) != NULL) {
        printf("Processing centre_to_left_enhanced.csv...\n");
        printf("Header: %s", line);
    }

    while (fgets(line, sizeof(line), input_file)) {
        // Remove newline character
        line[strcspn(line, "\n")] = 0;

        // Parse CSV line - expecting: Pas Index,Fibonacci,Expression,Choices
        char line_copy[MAX_LINE_SIZE];
        strcpy(line_copy, line);

        char *pas_index_str = strtok(line_copy, ",");
        char *fibonacci_str = strtok(NULL, ",");

        // Skip the expression field (it's quoted, so we need to handle it carefully)
        char *expression_start = strtok(NULL, "\"");
        char *expression_content = strtok(NULL, "\"");
        char *comma_after_expression = strtok(NULL, ",");

        // Now get the choices field
        char *choices_str = strtok(NULL, "\"");
        if (choices_str == NULL) {
            // Try alternative parsing if the above fails
            strcpy(line_copy, line);
            pas_index_str = strtok(line_copy, ",");
            fibonacci_str = strtok(NULL, ",");

            // Find the last quoted section (choices)
            char *last_quote = strrchr(line, '"');
            if (last_quote != NULL) {
                *last_quote = '\0';
                char *second_last_quote = strrchr(line, '"');
                if (second_last_quote != NULL) {
                    choices_str = second_last_quote + 1;
                }
            }
        }

        if (pas_index_str && choices_str) {
            int pas_index = atoi(pas_index_str);

            // Convert choices to decimal using enumeration order
            long long decimal_number = choicesToDecimal(choices_str, pas_index);

            // Write to output CSV
            fprintf(output_file, "%d,%lld\n", pas_index, decimal_number);

            processed_count++;
            if (processed_count % 50 == 0 || processed_count <= 10) {
                printf("Processed %d: PasIndex=%d, Choices='%s', Decimal=%lld\n",
                       processed_count, pas_index, choices_str, decimal_number);

                // Verification: convert back to show it's correct
                char verification[MAX_CHOICES_SIZE];
                decimalToChoices(decimal_number, pas_index, verification);
                printf("  Verification: %lld -> '%s'\n", decimal_number, verification);
            }
        } else {
            error_count++;
            printf("Warning: Failed to parse line: %s\n", line);
        }
    }

    fclose(input_file);
    fclose(output_file);

    printf("\n=== CONVERSION COMPLETED ===\n");
    printf("Total rows processed: %d\n", processed_count);
    printf("Errors encountered: %d\n", error_count);
    printf("Output saved to: symbol_choices_conversion_to_decimal.csv\n");

    if (processed_count > 0) {
        printf("\nSample output format:\n");
        printf("Pas Index,Decimal Number\n");
        printf("3,19\n");
        printf("4,67\n");
        printf("...\n");
    }

    return 0;
}
