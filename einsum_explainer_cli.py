import sys
from einsum_explainer import EinsumExplainer


def parse_input(input_str):
    parts = input_str.split()
    expression = parts[0]
    matrix_specs = [parse_matrix_spec(spec) for spec in parts[1:]]
    return expression, matrix_specs


def parse_matrix_spec(spec):
    parts = spec.split(',')
    return (parts[0], [int(x) for x in parts[1:]])


def main():
    print("Welcome to the EinsumExplainer CLI!")
    print("Enter einsum expressions in the format: 'expression matrix1,dim1,dim2 matrix2,dim1,dim2 ...'")
    print("For example: 'ij,jk->ik A,2,3 B,3,4'")
    print("Type 'exit' to quit the program.")
    print()

    while True:
        try:
            user_input = input("Enter einsum expression (or 'exit'): ").strip()

            if user_input.lower() == 'exit':
                print("Exiting EinsumExplainer CLI. Goodbye!")
                break

            expression, matrix_specs = parse_input(user_input)

            explainer = EinsumExplainer(expression, *matrix_specs)
            explanation = explainer.parse_einsum()

            print("\nExplanation:")
            print(explanation)
            print()

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please check your input and try again.")
            print()


if __name__ == "__main__":
    main()