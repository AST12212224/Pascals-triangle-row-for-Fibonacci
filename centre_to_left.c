#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

// Enhanced data types to match Big_sequence_generator.c
#define MAX_DIGITS 500
#define MAX_ARRAY_SIZE 1000

// BigInt structure for handling very large numbers
typedef struct {
    char digits[MAX_DIGITS];
    int length;
    int is_negative;
} BigInt;

// Function declarations
void initBigInt(BigInt* num);
void setBigIntFromString(BigInt* num, const char* str);
void setBigIntFromLongLong(BigInt* num, long long value);
void copyBigInt(BigInt* dest, const BigInt* src);
int compareBigInt(const BigInt* a, const BigInt* b);
void addBigInt(BigInt* result, const BigInt* a, const BigInt* b);
void subtractBigInt(BigInt* result, const BigInt* a, const BigInt* b);
void multiplyBigIntBy2(BigInt* result, const BigInt* a);
void absDifferenceBigInt(BigInt* result, const BigInt* a, const BigInt* b);
void bigIntToString(const BigInt* num, char* str);
void choose_operation(BigInt* result, const BigInt* current_sum, const BigInt* target, const BigInt* element, char *op);

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

// Set BigInt from long long
void setBigIntFromLongLong(BigInt* num, long long value) {
    num->is_negative = 0;
    if (value < 0) {
        num->is_negative = 1;
        value = -value;
    }

    if (value == 0) {
        initBigInt(num);
        return;
    }

    sprintf(num->digits, "%lld", value);
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

int main() {
    BigInt target;
    int n;
    char target_str[MAX_DIGITS];

    printf("Enter target Fibonacci term: ");
    scanf("%s", target_str);
    setBigIntFromString(&target, target_str);

    printf("Enter number of elements in Pascal triangle row: ");
    scanf("%d", &n);

    if (n > MAX_ARRAY_SIZE) {
        printf("Error: Array size too large. Maximum supported: %d\n", MAX_ARRAY_SIZE);
        return 1;
    }

    BigInt arr[MAX_ARRAY_SIZE];
    printf("Enter %d elements:\n", n);
    for (int i = 0; i < n; i++) {
        char element_str[MAX_DIGITS];
        scanf("%s", element_str);
        setBigIntFromString(&arr[i], element_str);
    }

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
        choose_operation(&new_sum, &current_sum, &target, &arr[i], &ops[i]);
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

        if (compareBigInt(&current_sum, &target) == 0) break;
    }

    // Output expression
    char target_display[MAX_DIGITS];
    bigIntToString(&target, target_display);
    printf("\nTarget = %s\n", target_display);
    printf("Expression: ");

    for (int i = 0; i < n; i++) {
        char element_str[MAX_DIGITS];
        bigIntToString(&arr[i], element_str);

        if (ops[i] == '0') {
            printf("0*%s ", element_str);
        }
        else if (ops[i] == '=') {
            printf("+%s ", element_str);
        }
        else {
            printf("%c%s ", ops[i], element_str);
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

    // Calculate and verify the sum
    BigInt verification_sum;
    initBigInt(&verification_sum);
    printf("Verification: ");

    for (int i = 0; i < n; i++) {
        if (ops[i] == '0') {
            // Skip this element
        }
        else if (ops[i] == '=' || ops[i] == '+') {
            BigInt temp_sum;
            addBigInt(&temp_sum, &verification_sum, &arr[i]);
            copyBigInt(&verification_sum, &temp_sum);
        }
        else if (ops[i] == '-') {
            BigInt temp_sum;
            subtractBigInt(&temp_sum, &verification_sum, &arr[i]);
            copyBigInt(&verification_sum, &temp_sum);
        }
    }

    char verification_str[MAX_DIGITS];
    bigIntToString(&verification_sum, verification_str);

    printf("Sum = %s, Target = %s, Match = %s\n",
           verification_str, target_display,
           (compareBigInt(&verification_sum, &target) == 0) ? "YES" : "NO");

    return 0;
}
