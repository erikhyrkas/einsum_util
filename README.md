# Einsum Utilities

This project provides two utilities for working with Einstein summation (einsum) expressions in Python:

1. **EinsumGeneratorUtil**: A utility for building einsum expressions.
2. **EinsumExplainer**: A utility for explaining and validating einsum expressions.

These tools are designed to make working with einsum expressions easier, more intuitive, and less error-prone.

## Motivation

Working with einsum is easy, until it isn't. I created these utilities to help me validate my assumptions and make sure that my code was right.
## Features

### EinsumGeneratorUtil

- Create matrices with named dimensions and shapes
- Generate einsum expressions for common operations:
  - Dot product
  - Matrix-vector product
  - Matrix multiplication
  - Outer product
  - Transpose
  - Sum over axis
- Handle multi-dimensional einsum expressions
- Validate shape consistency across operations

### EinsumExplainer

- Parse and explain einsum operations in detail
- Categorize operations (single input, two input, multi-input)
- Calculate and explain the number of FLOPs (Floating Point Operations) for each operation
- Generate pseudo-code to illustrate the operation
- Provide specific explanations for different types of operations
- Validate input shapes and dimensions

## Installation

To use these utilities, simply clone this repository:

```bash
git clone https://github.com/yourusername/einsum-utilities.git
cd einsum-utilities
```

## Usage

### EinsumGeneratorUtil

```python
from einsum_generator import EinsumGeneratorUtil

util = EinsumGeneratorUtil()

# Define matrices
A = util.Matrix('A', ['i', 'j'], [3, 4])
B = util.Matrix('B', ['j', 'k'], [4, 5])

# Generate einsum expression for matrix multiplication
result = util.matrix_multiplication(A, B)
print(result)  # Output: torch.einsum('ij,jk->ik', [A, B])
```

### EinsumExplainer

```python
from einsum_explainer import EinsumExplainer

# Define an einsum expression and matrix specifications
einsum_expr = "ij,jk->ik"
matrix_specs = [('A', [2, 3]), ('B', [3, 4])]

# Create an EinsumExplainer instance
explainer = EinsumExplainer(einsum_expr, *matrix_specs)

# Get the explanation
explanation = explainer.parse_einsum()
print(explanation)
```

## Testing

The project currently includes test functions for visual validation. These can be found in the source files and can be run to check the functionality of the utilities.

## Command-Line Interfaces (CLIs)

Both utilities now have command-line interfaces for easier interaction.

### EinsumGeneratorUtil CLI

The EinsumGeneratorUtil CLI guides users through a series of questions to generate the desired einsum expression. To use it:

```bash
python einsum_generator_cli.py
```

Example interaction:

```
Welcome to EinsumGeneratorUtil CLI!

What operation would you like to perform?
1) Dot product
2) Matrix-vector product
3) Matrix multiplication
4) Outer product
5) Transpose
6) Sum over axis
7) Custom multi-dimensional einsum
Enter the number of your choice: 3

Enter the name and dimensions for the first matrix (e.g., A,2,3): A,2,3
Enter the name and dimensions for the second matrix (e.g., B,3,4): B,3,4

Generated einsum expression: torch.einsum('ij,jk->ik', [A, B])
```

### EinsumExplainer CLI

The EinsumExplainer CLI now operates in a loop, allowing users to input multiple einsum expressions for explanation. To use it:

```bash
python einsum_explainer_cli.py
```

Example interaction:

```
Welcome to the EinsumExplainer CLI!
Enter einsum expressions in the format: 'expression matrix1,dim1,dim2 matrix2,dim1,dim2 ...'
For example: 'ij,jk->ik A,2,3 B,3,4'
Type 'exit' to quit the program.

Enter einsum expression (or 'exit'): ij,jk->ik A,2,3 B,3,4

Explanation:
(Detailed explanation appears here)

Enter einsum expression (or 'exit'): ii-> A,3,3

Explanation:
(Explanation for trace operation appears here)

Enter einsum expression (or 'exit'): exit
Exiting EinsumExplainer CLI. Goodbye!
```

This interface allows users to continuously input einsum expressions and receive explanations without restarting the program.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.