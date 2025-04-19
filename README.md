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
