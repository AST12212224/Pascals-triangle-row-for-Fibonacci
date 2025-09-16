#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define MAX_VALUES 600000  // Maximum unique values for -2^18 to 2^18 range (~524K values)
#define MAX_LINE 1000     // Maximum line length in CSV

// Simple hash set implementation
typedef struct {
    long long values[MAX_VALUES];
    bool used[MAX_VALUES];
    int count;
} ValueSet;

// Initialize the set
void init_set(ValueSet* set) {
    set->count = 0;
    for (int i = 0; i < MAX_VALUES; i++) {
        set->used[i] = false;
    }
}

// Add value to set (if not already present)
void add_to_set(ValueSet* set, long long value) {
    // Simple linear search for small sets
    for (int i = 0; i < set->count; i++) {
        if (set->values[i] == value) {
            return; // Already exists
        }
    }
    
    // Add new value
    if (set->count < MAX_VALUES) {
        set->values[set->count] = value;
        set->used[set->count] = true;
        set->count++;
    }
}

// Check if value exists in set
bool contains(ValueSet* set, long long value) {
    for (int i = 0; i < set->count; i++) {
        if (set->values[i] == value) {
            return true;
        }
    }
    return false;
}

// Integer power function
long long int_pow(int base, int exp) {
    long long result = 1;
    for (int i = 0; i < exp; i++) {
        result *= base;
    }
    return result;
}

int main() {
    char input_filename[100];
    char output_filename[100];
    int n_size;
    
    printf("Enter the CSV filename to verify (e.g., 1_1.csv): ");
    scanf("%s", input_filename);
    
    printf("Enter the array size (n) that was used to generate this data: ");
    scanf("%d", &n_size);
    
    // Create output filename with _Checked suffix
    strcpy(output_filename, input_filename);
    char* dot = strrchr(output_filename, '.');
    if (dot != NULL) {
        *dot = '\0'; // Remove extension
        strcat(output_filename, "_Checked.csv");
    } else {
        strcat(output_filename, "_Checked.csv");
    }
    
    // Open input file
    FILE* input_file = fopen(input_filename, "r");
    if (input_file == NULL) {
        printf("Error: Could not open input file %s\n", input_filename);
        return 1;
    }
    
    // Initialize set to store unique values
    ValueSet found_values;
    init_set(&found_values);
    
    char line[MAX_LINE];
    
    // Skip header line
    if (fgets(line, sizeof(line), input_file) != NULL) {
        // Header skipped
    }
    
    // Read all combination values and add to set
    while (fgets(line, sizeof(line), input_file)) {
        // Parse CSV line to extract combination value (third column)
        char* token1 = strtok(line, ",");  // Step
        char* token2 = strtok(NULL, ",");  // Sequence
        char* token3 = strtok(NULL, ",\n"); // Combination Value
        
        if (token3 != NULL) {
            long long value = atoll(token3);
            add_to_set(&found_values, value);
        }
    }
    fclose(input_file);
    
    // Calculate theoretical range [-2^n, 2^n]
    long long max_range = int_pow(2, n_size);
    long long min_range = -max_range;
    
    // Open output file
    FILE* output_file = fopen(output_filename, "w");
    if (output_file == NULL) {
        printf("Error: Could not create output file %s\n", output_filename);
        return 1;
    }
    
    // Write header
    fprintf(output_file, "Theoretical Value,Found\n");
    
    // Check each value in theoretical range
    int total_values = 0;
    int found_count = 0;
    
    for (long long value = min_range; value <= max_range; value++) {
        total_values++;
        bool found = contains(&found_values, value);
        if (found) found_count++;
        
        fprintf(output_file, "%lld,%s\n", value, found ? "Yes" : "No");
    }
    
    fclose(output_file);
    
    // Print results
    printf("\n=== VERIFICATION RESULTS ===\n");
    printf("Input file: %s\n", input_filename);
    printf("Output file: %s\n", output_filename);
    printf("Array size (n): %d\n", n_size);
    printf("Theoretical range: [%lld, %lld]\n", min_range, max_range);
    printf("Total theoretical values: %d\n", total_values);
    printf("Values found in data: %d\n", found_count);
    printf("Missing values: %d\n", total_values - found_count);
    
    if (found_count == total_values) {
        printf("✅ CLAIM VERIFIED: All values in range [-2^%d, 2^%d] are present!\n", n_size, n_size);
    } else {
        printf("❌ CLAIM FAILED: %d values missing from theoretical range.\n", total_values - found_count);
    }
    
    printf("============================\n");
    
    return 0;
}