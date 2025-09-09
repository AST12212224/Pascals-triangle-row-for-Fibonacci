#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Enhanced data types to match Big_sequence_generator.c
#define MAX_DIGITS 500
#define MAX_ARRAY_SIZE 1000
#define MAX_LINE_SIZE 10000
#define MAX_EXPRESSION_SIZE 5000

// BigInt structure for handling very large numbers
typedef struct {
    char digits[MAX_DIGITS];
    int length;
    int is_negative;
} BigInt;

// Function declarations
void initBigInt(BigInt* num);
void setBigIntFromString(BigInt* num, const char* str);
void copyBigInt(BigInt* dest, const BigInt* src);
int compareBigInt(const BigInt* a, const BigInt* b);
void addBigInt(BigInt* result, const BigInt* a, const BigInt* b);
void subtractBigInt(BigInt* result, const BigInt* a, const BigInt* b);
void multiplyBigIntBy2(BigInt* result, const BigInt* a);
void absDifferenceBigInt(BigInt* result, const BigInt* a, const BigInt* b);
void bigIntToString(const BigInt* num, char* str);
void choose_operation(BigInt* result, const BigInt* current_sum, const BigInt* target, const BigInt* element, char *op);
void solve_pascal_fibonacci(const BigInt* target, int n, BigInt* arr, char* expression, char* choices);
int parse_pascal_row(char *pascal_str, BigInt *arr);

// Initialize BigInt
void initBigInt(BigInt* num) {
    strcpy(num->digits, "0");
    num->length = 1;
    num->is_negative = 0;
}

// Set BigInt from string
void setBigIntFromString(BigInt* num, const char* str) {
    int start = 0;
    num->is_negative = 0;

    if (str[0] == '-') {
        num->is_negative = 1;
        start = 1;
    }

    strcpy(num->digits, str + start);
    num->length = strlen(num->digits);
}

// Copy BigInt
void copyBigInt(BigInt* dest, const BigInt* src) {
    strcpy(dest->digits, src->digits);
    dest->length = src->length;
    dest->is_negative = src->is_negative;
}

// Compare two BigInts (returns -1 if a < b, 0 if equal, 1 if a > b)
int compareBigInt(const BigInt* a, const BigInt* b) {
    if (a->is_negative && !b->is_negative) return -1;
    if (!a->is_negative && b->is_negative) return 1;

    int sign = a->is_negative ? -1 : 1;

    if (a->length < b->length) return -1 * sign;
    if (a->length > b->length) return 1 * sign;

    int cmp = strcmp(a->digits, b->digits);
    if (cmp < 0) return -1 * sign;
    if (cmp > 0) return 1 * sign;
    return 0;
}

// Add two BigInts
void addBigInt(BigInt* result, const BigInt* a, const BigInt* b) {
    if (a->is_negative == b->is_negative) {
        // Same sign - add magnitudes
        int carry = 0;
        int maxLen = (a->length > b->length) ? a->length : b->length;
        char temp[MAX_DIGITS];
        int tempLen = 0;

        for (int i = 0; i < maxLen || carry; i++) {
            int sum = carry;

            if (i < a->length) {
                sum += a->digits[a->length - 1 - i] - '0';
            }
            if (i < b->length) {
                sum += b->digits[b->length - 1 - i] - '0';
            }

            temp[tempLen++] = (sum % 10) + '0';
            carry = sum / 10;
        }

        // Reverse result
        for (int i = 0; i < tempLen; i++) {
            result->digits[i] = temp[tempLen - 1 - i];
        }
        result->digits[tempLen] = '\0';
        result->length = tempLen;
        result->is_negative = a->is_negative;
    } else {
        // Different signs - subtract magnitudes
        subtractBigInt(result, a, b);
    }
}

// Subtract two BigInts (a - b)
void subtractBigInt(BigInt* result, const BigInt* a, const BigInt* b) {
    if (a->is_negative != b->is_negative) {
        // Different signs - add magnitudes
        BigInt temp_b;
        copyBigInt(&temp_b, b);
        temp_b.is_negative = a->is_negative;
        addBigInt(result, a, &temp_b);
        return;
    }

    // Same signs - subtract magnitudes
    const BigInt* larger;
    const BigInt* smaller;
    int result_negative = 0;

    int cmp = compareBigInt(a, b);
    if (cmp == 0) {
        initBigInt(result);
        return;
    }

    if (cmp > 0) {
        larger = a;
        smaller = b;
        result_negative = a->is_negative;
    } else {
        larger = b;
        smaller = a;
        result_negative = !a->is_negative;
    }

    int borrow = 0;
    char temp[MAX_DIGITS];
    int tempLen = 0;

    for (int i = 0; i < larger->length; i++) {
        int digit1 = larger->digits[larger->length - 1 - i] - '0';
        int digit2 = (i < smaller->length) ? smaller->digits[smaller->length - 1 - i] - '0' : 0;

        digit1 -= borrow;
        if (digit1 < digit2) {
            digit1 += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }

        temp[tempLen++] = (digit1 - digit2) + '0';
    }

    // Remove leading zeros
    while (tempLen > 1 && temp[tempLen - 1] == '0') {
        tempLen--;
    }

    // Reverse result
    for (int i = 0; i < tempLen; i++) {
        result->digits[i] = temp[tempLen - 1 - i];
    }
    result->digits[tempLen] = '\0';
    result->length = tempLen;
    result->is_negative = result_negative;
}

// Multiply BigInt by 2
void multiplyBigIntBy2(BigInt* result, const BigInt* a) {
    addBigInt(result, a, a);
}

// Calculate absolute difference between two BigInts
void absDifferenceBigInt(BigInt* result, const BigInt* a, const BigInt* b) {
    subtractBigInt(result, a, b);
    result->is_negative = 0; // Make it positive
}

// Convert BigInt to string for display
void bigIntToString(const BigInt* num, char* str) {
    if (num->is_negative) {
        strcpy(str, "-");
        strcat(str, num->digits);
    } else {
        strcpy(str, num->digits);
    }
}

// Function to decide operation with palindromic consideration
void choose_operation(BigInt* result, const BigInt* current_sum, const BigInt* target, const BigInt* element, char *op) {
    BigInt current_distance, best_sum, best_distance;
    char best_op = '0';

    absDifferenceBigInt(&current_distance, current_sum, target);
    copyBigInt(&best_sum, current_sum);
    copyBigInt(&best_distance, &current_distance);

    // Try subtraction operations first
    // Regular subtraction
    BigInt new_sum, new_distance;
    subtractBigInt(&new_sum, current_sum, element);
    absDifferenceBigInt(&new_distance, &new_sum, target);

    if (compareBigInt(&new_distance, &best_distance) < 0) {
        copyBigInt(&best_sum, &new_sum);
        best_op = '-';
        copyBigInt(&best_distance, &new_distance);
    }

    // Palindromic subtraction (subtract 2*element)
    BigInt double_element;
    multiplyBigIntBy2(&double_element, element);
    subtractBigInt(&new_sum, current_sum, &double_element);
    absDifferenceBigInt(&new_distance, &new_sum, target);

    if (compareBigInt(&new_distance, &best_distance) < 0) {
        copyBigInt(&best_sum, &new_sum);
        best_op = 'S'; // 'S' for palindromic subtraction
        copyBigInt(&best_distance, &new_distance);
    }

    // If subtraction helped, don't try addition
    if (compareBigInt(&best_distance, &current_distance) < 0) {
        *op = best_op;
        copyBigInt(result, &best_sum);
        return;
    }

    // Try addition operations if subtraction didn't help
    // Regular addition
    addBigInt(&new_sum, current_sum, element);
    absDifferenceBigInt(&new_distance, &new_sum, target);

    if (compareBigInt(&new_distance, &best_distance) < 0) {
        copyBigInt(&best_sum, &new_sum);
        best_op = '+';
        copyBigInt(&best_distance, &new_distance);
    }

    // Palindromic addition (add 2*element)
    addBigInt(&new_sum, current_sum, &double_element);
    absDifferenceBigInt(&new_distance, &new_sum, target);

    if (compareBigInt(&new_distance, &best_distance) < 0) {
        copyBigInt(&best_sum, &new_sum);
        best_op = 'A'; // 'A' for palindromic addition
        copyBigInt(&best_distance, &new_distance);
    }

    *op = best_op;
    copyBigInt(result, &best_sum);
}

// Function to solve single Pascal-Fibonacci problem
void solve_pascal_fibonacci(const BigInt* target, int n, BigInt* arr, char* expression, char* choices) {
    int center = (int)ceil(n / 2.0) - 1;
    BigInt current_sum;
    copyBigInt(&current_sum, &arr[center]);
    char ops[MAX_ARRAY_SIZE];

    // Initialize all as skipped
    for (int i = 0; i < n; i++) ops[i] = '0';
    ops[center] = '='; // starting point

    // Process left elements (from center-1 down to 0)
    for (int i = center - 1; i >= 0; i--) {
        BigInt new_sum;
        choose_operation(&new_sum, &current_sum, target, &arr[i], &ops[i]);
        copyBigInt(&current_sum, &new_sum);

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

        if (compareBigInt(&current_sum, target) == 0) break;
    }

    // Build expression string
    strcpy(expression, "");
    strcpy(choices, "");

    for (int i = 0; i < n; i++) {
        char temp[MAX_DIGITS + 10];
        char element_str[MAX_DIGITS];
        bigIntToString(&arr[i], element_str);

        if (ops[i] == '0') {
            sprintf(temp, "0*%s ", element_str);
            strcat(expression, temp);
            strcat(choices, "0 ");
        }
        else if (ops[i] == '=') {
            sprintf(temp, "+%s ", element_str);
            strcat(expression, temp);
            strcat(choices, "+ ");
        }
        else {
            sprintf(temp, "%c%s ", ops[i], element_str);
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
int parse_pascal_row(char *pascal_str, BigInt *arr) {
    int count = 0;

    // Remove quotes if present
    if (pascal_str[0] == '"') {
        pascal_str++;
        int len = strlen(pascal_str);
        if (len > 0 && pascal_str[len-1] == '"') {
            pascal_str[len-1] = '\0';
        }
    }

    char *token = strtok(pascal_str, " ");
    while (token != NULL && count < MAX_ARRAY_SIZE) {
        setBigIntFromString(&arr[count], token);
        count++;
        token = strtok(NULL, " ");
    }
    return count;
}

int main() {
    FILE *input_file = fopen("fib_pascal_data.csv", "r");
    if (input_file == NULL) {
        printf("Error: Cannot open fib_pascal_data.csv\n");
        printf("Please ensure the file exists and was generated by Big_sequence_generator\n");
        return 1;
    }

    FILE *output_file = fopen("centre_to_left_enhanced.csv", "w");
    if (output_file == NULL) {
        printf("Error: Cannot create centre_to_left_enhanced.csv\n");
        fclose(input_file);
        return 1;
    }

    // Write CSV header
    fprintf(output_file, "Pas Index,Fibonacci,Expression,Choices\n");

    char line[MAX_LINE_SIZE];
    int processed_count = 0;

    // Skip header line
    if (fgets(line, sizeof(line), input_file) != NULL) {
        printf("Skipped header: %s", line);
    }

    while (fgets(line, sizeof(line), input_file)) {
        // Remove newline character
        line[strcspn(line, "\n")] = 0;

        // Parse CSV line - handle quoted Pascal triangle data
        char *pas_index_str = strtok(line, ",");
        char *fibonacci_str = strtok(NULL, ",");
        char *pascal_str = strtok(NULL, "\"");  // Get everything after second comma

        if (pas_index_str && fibonacci_str && pascal_str) {
            int pas_index = atoi(pas_index_str);

            // Handle large Fibonacci numbers
            BigInt fibonacci;
            setBigIntFromString(&fibonacci, fibonacci_str);

            // Parse Pascal triangle row
            BigInt pascal_arr[MAX_ARRAY_SIZE];
            char pascal_copy[MAX_LINE_SIZE];

            // Find the actual Pascal data (after the quote)
            char *actual_pascal = pascal_str;

            // Skip any leading whitespace or quotes
            while (*actual_pascal == ' ' || *actual_pascal == '"' || *actual_pascal == ',') {
                actual_pascal++;
            }

            strcpy(pascal_copy, actual_pascal);

            // Remove trailing quote if present
            int len = strlen(pascal_copy);
            if (len > 0 && pascal_copy[len-1] == '"') {
                pascal_copy[len-1] = '\0';
            }

            int n = parse_pascal_row(pascal_copy, pascal_arr);

            if (n > 0) {
                // Solve the problem
                char expression[MAX_EXPRESSION_SIZE];
                char choices[MAX_EXPRESSION_SIZE];
                solve_pascal_fibonacci(&fibonacci, n, pascal_arr, expression, choices);

                // Write to output CSV
                char fib_str[MAX_DIGITS];
                bigIntToString(&fibonacci, fib_str);

                fprintf(output_file, "%d,%s,\"%s\",\"%s\"\n",
                        pas_index, fib_str, expression, choices);

                processed_count++;
                if (processed_count % 10 == 0 || processed_count <= 5) {
                    printf("Processed %d: PasIndex=%d, Fibonacci=%s\n",
                           processed_count, pas_index, fib_str);
                }
            } else {
                printf("Warning: Failed to parse Pascal row for index %d\n", pas_index);
            }
        } else {
            printf("Warning: Failed to parse line: %s\n", line);
        }
    }

    fclose(input_file);
    fclose(output_file);

    printf("\nProcessing completed!\n");
    printf("Total rows processed: %d\n", processed_count);
    printf("Results saved in centre_to_left_enhanced.csv\n");
    return 0;
}
