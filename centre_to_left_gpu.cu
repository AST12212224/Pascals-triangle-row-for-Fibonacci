// run_wrapper.cu
// CUDA wrapper that reads fibonacci_pascals.csv and writes centre_to_left.csv
// Input columns used: Pas Index, Fibonacci, Pascal's Triangle
// Output columns: Pas Index, Fibonacci, Expression, Choices, Final Sum

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

// ---------------- Device-side logic ----------------

__device__ __forceinline__ int d_abs_val(int x) { return x < 0 ? -x : x; }

__device__ int d_choose_operation(int current_sum, int target, int element, char *op) {
    int new_sum = current_sum - element;
    if (d_abs_val(new_sum - target) < d_abs_val(current_sum - target)) {
        *op = '-';
        return new_sum;
    }
    new_sum = current_sum + element;
    if (d_abs_val(new_sum - target) < d_abs_val(current_sum - target)) {
        *op = '+';
        return new_sum;
    }
    *op = '0';
    return current_sum;
}

// Kernel processes one CSV row per thread
__global__ void process_rows_kernel(const int *values, const int *row_offsets,
                                    const int *row_lengths, const int *targets,
                                    char *ops_out, int *final_sums,
                                    int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    int start = row_offsets[row];
    int len   = row_lengths[row];
    int target = targets[row];

    if (len <= 0) {
        final_sums[row] = 0;
        return;
    }

    int center = len / 2;
    int current_sum = values[start + center];

    // ops for this row live in ops_out[start .. start+len-1]
    char *ops = ops_out + start;
    for (int i = 0; i < len; i++) ops[i] = '0';
    ops[center] = '=';

    for (int i = center - 1; i >= 0; i--) {
        current_sum = d_choose_operation(current_sum, target, values[start + i], &ops[i]);
        if (current_sum == target) break;
    }

    final_sums[row] = current_sum;
}

// ---------------- Host-side helpers ----------------

static inline std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return s.substr(a, b - a + 1);
}

// Split a CSV line into top-level comma-separated fields
static std::vector<std::string> split_csv_line(const std::string &line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false;
    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
            continue;
        }
        if (c == ',' && !in_quotes) {
            out.push_back(trim(cur));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(trim(cur));
    return out;
}

// Parse space separated integers from the Pascal's Triangle field
static std::vector<int> parse_pascal_row(const std::string &field) {
    std::vector<int> v;
    std::istringstream iss(field);
    std::string tok;
    while (iss >> tok) {
        // strip any stray commas
        tok.erase(std::remove(tok.begin(), tok.end(), ','), tok.end());
        if (!tok.empty()) v.push_back(std::stoi(tok));
    }
    return v;
}

// ---------------- Main ----------------

int main(int argc, char **argv) {
    std::string input_path  = (argc >= 2) ? argv[1] : "fibonacci_pascals.csv";
    std::string output_path = (argc >= 3) ? argv[2] : "centre_to_left.csv";

    std::ifstream fin(input_path);
    if (!fin) {
        std::cerr << "Failed to open input CSV: " << input_path << "\n";
        return 1;
    }

    // Storage for all rows
    std::vector<int> h_targets;          // Fibonacci
    std::vector<int> h_row_lengths;      // Pas Index
    std::vector<int> h_row_offsets;      // prefix offsets into values
    std::vector<int> h_values;           // flattened Pascal rows
    std::vector<std::vector<int>> per_row_values; // keep per row to format strings later

    std::string header;
    if (!std::getline(fin, header)) {
        std::cerr << "Empty CSV\n";
        return 1;
    }

    int total_vals = 0;
    std::string line;
    int line_no = 1;

    while (std::getline(fin, line)) {
        line_no++;
        if (trim(line).empty()) continue;

        auto cols = split_csv_line(line);
        // Expect at least 4 columns, but we only use:
        // Pas Index, Fibonacci, Pascal's Triangle
        // From your screenshot the order looked like:
        // Fib Index | Pas Index | Fibonacci | Pascal's Triangle
        if (cols.size() < 4) {
            std::cerr << "Skipping line " << line_no << " due to insufficient columns\n";
            continue;
        }

        // Map columns
        int pas_index  = 0;
        int fibonacci  = 0;
        std::string pascal_field;

        // Assuming columns are exactly as above
        // cols[0] = Fib Index  (ignored)
        // cols[1] = Pas Index
        // cols[2] = Fibonacci
        // cols[3] = Pascal's Triangle
        try {
            pas_index = std::stoi(cols[1]);
            fibonacci = std::stoi(cols[2]);
        } catch (...) {
            std::cerr << "Skipping line " << line_no << " due to non-integer Pas Index or Fibonacci\n";
            continue;
        }
        pascal_field = cols[3];

        auto row_vals = parse_pascal_row(pascal_field);
        if ((int)row_vals.size() != pas_index) {
            // If CSV row length differs, trust actual parsed length
            pas_index = (int)row_vals.size();
        }
        if (pas_index <= 0) continue;

        h_row_offsets.push_back(total_vals);
        h_row_lengths.push_back(pas_index);
        h_targets.push_back(fibonacci);
        per_row_values.push_back(row_vals);

        // append into flat values
        h_values.insert(h_values.end(), row_vals.begin(), row_vals.end());
        total_vals += pas_index;
    }
    fin.close();

    int num_rows = (int)h_row_lengths.size();
    if (num_rows == 0) {
        std::cerr << "No usable rows found\n";
        return 1;
    }

    // Allocate device memory
    int *d_values = nullptr, *d_row_offsets = nullptr, *d_row_lengths = nullptr, *d_targets = nullptr, *d_final_sums = nullptr;
    char *d_ops = nullptr;

    cudaMalloc((void**)&d_values,      h_values.size() * sizeof(int));
    cudaMalloc((void**)&d_row_offsets, num_rows * sizeof(int));
    cudaMalloc((void**)&d_row_lengths, num_rows * sizeof(int));
    cudaMalloc((void**)&d_targets,     num_rows * sizeof(int));
    cudaMalloc((void**)&d_final_sums,  num_rows * sizeof(int));
    cudaMalloc((void**)&d_ops,         h_values.size() * sizeof(char));

    cudaMemcpy(d_values,      h_values.data(),      h_values.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_offsets, h_row_offsets.data(), num_rows * sizeof(int),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_lengths, h_row_lengths.data(), num_rows * sizeof(int),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets,     h_targets.data(),     num_rows * sizeof(int),        cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (num_rows + threads - 1) / threads;
    process_rows_kernel<<<blocks, threads>>>(d_values, d_row_offsets, d_row_lengths, d_targets,
                                             d_ops, d_final_sums, num_rows);
    cudaDeviceSynchronize();

    // Copy back
    std::vector<int>  h_final_sums(num_rows);
    std::vector<char> h_ops(h_values.size());
    cudaMemcpy(h_final_sums.data(), d_final_sums, num_rows * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ops.data(),        d_ops,        h_values.size() * sizeof(char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_values);
    cudaFree(d_row_offsets);
    cudaFree(d_row_lengths);
    cudaFree(d_targets);
    cudaFree(d_final_sums);
    cudaFree(d_ops);

    // Write output CSV
    std::ofstream fout(output_path);
    if (!fout) {
        std::cerr << "Failed to open output CSV: " << output_path << "\n";
        return 1;
    }
    fout << "Pas Index,Fibonacci,Expression,Choices,Final Sum\n";

    for (int row = 0; row < num_rows; ++row) {
        int start = h_row_offsets[row];
        int len   = h_row_lengths[row];
        int target = h_targets[row];

        // Build Expression string and Choices string on host
        std::ostringstream expr, choices;
        for (int i = 0; i < len; ++i) {
            char op = h_ops[start + i];
            int val = per_row_values[row][i];
            if (op == '0') {
                expr << "0*" << val << " ";
                choices << "0";
            } else if (op == '=') {
                expr << "+" << val << " ";
                choices << "+";
            } else if (op == '+' || op == '-') {
                expr << op << val << " ";
                choices << op;
            } else {
                // default safeguard
                expr << "0*" << val << " ";
                choices << "0";
            }
            if (i + 1 < len) choices << " ";
        }
        // CSV-safe: wrap fields that may contain spaces in quotes
        fout << len << ",";
        fout << target << ",";
        fout << "\"" << expr.str() << "= " << h_final_sums[row] << "\""
             << ",";
        fout << "\"" << choices.str() << "\""
             << ",";
        fout << h_final_sums[row] << "\n";
    }
    fout.close();

    std::cout << "Wrote " << num_rows << " rows to " << output_path << "\n";
    return 0;
}
