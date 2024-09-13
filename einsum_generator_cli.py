import argparse
from einsum_generator import EinsumGeneratorUtil


def main():
    util = EinsumGeneratorUtil()

    print("Welcome to EinsumGeneratorUtil CLI!")

    operations = {
        1: ("Dot product", util.dot_product),
        2: ("Matrix-vector product", util.matrix_vector_product),
        3: ("Matrix multiplication", util.matrix_multiplication),
        4: ("Outer product", util.outer_product),
        5: ("Transpose", util.transpose),
        6: ("Sum over axis", util.sum_over_axis),
        7: ("Custom multi-dimensional einsum", util.multi_einsum)
    }

    print("\nWhat operation would you like to perform?")
    for key, value in operations.items():
        print(f"{key}) {value[0]}")

    choice = int(input("Enter the number of your choice: "))

    if choice not in operations:
        print("Invalid choice. Exiting.")
        return

    operation = operations[choice][1]

    if choice == 7:
        einsum_expr = input("Enter the custom einsum expression: ")
        num_matrices = int(input("How many matrices are involved? "))
        matrices = []
        for i in range(num_matrices):
            matrix_input = input(f"Enter the name and dimensions for matrix {i + 1} (e.g., A,2,3,4): ").split(',')
            name = matrix_input[0]
            dims = [chr(ord('i') + j) for j in range(len(matrix_input) - 1)]
            shape = [int(x) for x in matrix_input[1:]]
            matrices.append(util.Matrix(name, dims, shape))
        result = operation(einsum_expr, *matrices)
    else:
        if choice in [1, 2, 3, 4]:  # Operations with two inputs
            matrix1_input = input("Enter the name and dimensions for the first matrix/vector (e.g., A,2,3): ").split(
                ',')
            matrix2_input = input("Enter the name and dimensions for the second matrix/vector (e.g., B,3,4): ").split(
                ',')

            name1, name2 = matrix1_input[0], matrix2_input[0]
            dims1 = [chr(ord('i') + i) for i in range(len(matrix1_input) - 1)]
            dims2 = [chr(ord('i') + i) for i in range(len(matrix2_input) - 1)]
            shape1 = [int(x) for x in matrix1_input[1:]]
            shape2 = [int(x) for x in matrix2_input[1:]]

            matrix1 = util.Matrix(name1, dims1, shape1)
            matrix2 = util.Matrix(name2, dims2, shape2)

            result = operation(matrix1, matrix2)
        elif choice in [5, 6]:  # Operations with one input
            matrix_input = input("Enter the name and dimensions for the matrix (e.g., A,2,3): ").split(',')
            name = matrix_input[0]
            dims = [chr(ord('i') + i) for i in range(len(matrix_input) - 1)]
            shape = [int(x) for x in matrix_input[1:]]
            matrix = util.Matrix(name, dims, shape)

            if choice == 5:  # Transpose
                new_dims = input("Enter the new dimension order (e.g., ji): ")
                result = operation(matrix, new_dims)
            else:  # Sum over axis
                axis = int(input("Enter the axis to sum over: "))
                result = operation(matrix, axis)

    print(f"\nGenerated einsum expression: {result}")


if __name__ == "__main__":
    main()