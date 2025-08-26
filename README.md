# Pascals-Triangle-Row-for-Fibonacci

This repository explores a **new mathematical method** to compute Fibonacci numbers using **rows of Pascal’s Triangle**, not through shallow diagonals but via **coefficient permutations**. It’s a discovery-in-progress based on experimental brute-force analysis using C.

## 🧠 Concept

Given a Fibonacci number `F(n)`, we use the Pascal Triangle's `(n−2)`th row.  
Example:

- To compute `F(5) = 3`, we use row `3`: `[1, 3, 3, 1]`
- To compute `F(6) = 5`, we use row `4`: `[1, 4, 6, 4, 1]`

Each coefficient in the row is multiplied by one of three values: `0`, `+1`, or `−1`. The resulting numbers are summed, and if the total equals `F(n)`, that permutation is considered a valid representation.

There are `3^n` possible permutations per row (where `n` is the number of elements in the row).

### 🧪 Example: Using Pascal's Row `[1, 1]` to compute `F(3) = 1`

We use the 1st row of Pascal’s Triangle: `[1, 1]`  
We apply all possible combinations using multipliers: `−1`, `0`, `+1`  
Total combinations: `3² = 9`

> Format: `[a × m₁, b × m₂] = sum → ✅ if sum = 1`

| Combination | Evaluation | Result |
|-------------|------------|--------|
| `[-1, -1]`  | `−1 + (−1) = −2` | ❌ |
| `[-1, 0]`   | `−1 + 0 = −1`    | ❌ |
| `[-1, +1]`  | `−1 + 1 = 0`     | ❌ |
| `[0, -1]`   | `0 + (−1) = −1`  | ❌ |
| `[0, 0]`    | `0 + 0 = 0`      | ❌ |
| `[0, +1]`   | `0 + 1 = 1`      | ✅ |
| `[+1, -1]`  | `1 + (−1) = 0`   | ❌ |
| `[+1, 0]`   | `1 + 0 = 1`      | ✅ |
| `[+1, +1]`  | `1 + 1 = 2`      | ❌ |

✅ **Valid combinations**:  
- `[0, +1]`  
- `[+1, 0]`


## ⚙️ Algorithms

Six brute-force strategies are implemented in C, differing in the order of multiplier assignments:

- `0+−`
- `0−+`
- `−0+`
- `−+0`
- `+0−`
- `+-0`

Each one tries every combination of `0`, `+1`, and `−1` across the row coefficients to find valid sums that match the Fibonacci number.

## 📁 Output

Valid combinations are exported to `.csv` files for future analysis. These datasets are intended for spotting patterns, statistical trends, or even formulaic relationships between Pascal’s Triangle and Fibonacci numbers.


## Centre-to-Left Target Search Algorithm, Greedy Algorithm

### Overview
This algorithm scans from the center of a Pascal triangle row towards the left to find combinations that sum to a target Fibonacci number.

### Input Parameters
- `target_value`: The Fibonacci number to find
- `pascal_array[]`: Pascal triangle row coefficients
- `n`: Number of elements in array

### Output
- `operations[]`: Array of operations (+, -, or 0 for each element)
- `success`: Boolean indicating if target was found

### Algorithm Steps

#### 1. Initialize
```pseudocode
center_index = ceiling(n/2) - 1
current_sum = pascal_array[center_index]
operations[center_index] = "+" (mark center as starting point)

FOR i = 0 to n-1:
    IF i ≠ center_index:
        operations[i] = "0" (initially skip all non-center elements)
```

#### 2. Scan Left from Center
```pseudocode
FOR i = center_index-1 DOWN TO 0:
    
    a. Calculate current distance from target:
       current_distance = abs(current_sum - target_value)
    
    b. Test all possible operations for element i:
       best_sum = current_sum
       best_operation = "0"
       best_distance = current_distance
       
       // Try subtraction
       new_sum = current_sum - pascal_array[i]
       new_distance = abs(new_sum - target_value)
       IF new_distance < best_distance:
           best_sum = new_sum
           best_operation = "-"
           best_distance = new_distance
       
       // Try palindromic subtraction (subtract 2*element)
       new_sum = current_sum - (2 * pascal_array[i])
       new_distance = abs(new_sum - target_value)
       IF new_distance < best_distance:
           best_sum = new_sum
           best_operation = "S" (palindromic subtract)
           best_distance = new_distance
       
       // Try addition only if subtraction didn't improve
       IF best_distance >= current_distance:
           new_sum = current_sum + pascal_array[i]
           new_distance = abs(new_sum - target_value)
           IF new_distance < best_distance:
               best_sum = new_sum
               best_operation = "+"
               best_distance = new_distance
           
           // Try palindromic addition
           new_sum = current_sum + (2 * pascal_array[i])
           new_distance = abs(new_sum - target_value)
           IF new_distance < best_distance:
               best_sum = new_sum
               best_operation = "A" (palindromic add)
               best_distance = new_distance
    
    c. Apply best operation:
       current_sum = best_sum
       operations[i] = best_operation
       
       // Handle palindromic operations
       IF best_operation = "S":
           mirror_index = n - 1 - i
           IF mirror_index ≠ center_index AND mirror_index < n:
               operations[mirror_index] = "-"
           operations[i] = "-" (display as regular subtraction)
       
       IF best_operation = "A":
           mirror_index = n - 1 - i
           IF mirror_index ≠ center_index AND mirror_index < n:
               operations[mirror_index] = "+"
           operations[i] = "+" (display as regular addition)
    
    d. Check for exact match:
       IF current_sum = target_value:
           BREAK (target found)
```

#### 3. Return Results
```pseudocode
success = (current_sum = target_value)
RETURN operations[], success
```

### Key Features

- **Greedy approach**: Always chooses the operation that gets closest to target
- **Palindromic operations**: Can affect both current and mirrored positions  
- **Priority order**: Tries subtraction before addition
- **Early termination**: Stops when exact target is found
- **Center-outward**: Only processes left side, relies on palindromic operations for right side

### Algorithm Characteristics

This algorithm prioritizes finding solutions quickly rather than finding optimal solutions. It uses a distance-minimization heuristic to guide the search process.


## 🔬 Research Purpose

This is part of an **ongoing personal mathematical research project**. The ultimate goal is to derive a meaningful formula or structural insight from this relationship, potentially revealing a new connection in number theory.

You are welcome to explore, test, or contribute ideas — but please **do not publish or claim credit** for the core theory until the official work is released.

## 📜 License

This work is licensed under the [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

- **Attribution** — You must credit the author.
- **NonCommercial** — No commercial use allowed.
- **ShareAlike** — Share any derivatives under the same license.

## 🙌 Contributions

You're welcome to contribute by analyzing output, proposing optimizations, or helping with pattern recognition. Open an issue or start a discussion to collaborate.

---

> ✨ Let’s uncover the hidden paths from Pascal to Fibonacci — one permutation at a time.
