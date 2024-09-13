class EinsumExplainer:
    def __init__(self, einsum_expression: str, *matrix_specs):
        self.inputs, self.output = einsum_expression.split('->')
        self.input_subscripts = self.inputs.split(',')
        self.matrix_names = [spec[0] for spec in matrix_specs]
        self.matrix_shapes = {spec[0]: spec[1] for spec in matrix_specs}
        self.output_dims = set(self.output)
        self.matrix_shapes_ordered = [spec[1] for spec in matrix_specs]
        self.operation_type = self._categorize_operation()

    def _categorize_operation(self):
        if len(self.matrix_names) == 1:
            return "single_input"
        elif len(self.matrix_names) == 2:
            return "two_input"
        else:
            return "multi_input"

    def parse_einsum(self):
        explanation = [
            f"Explaining einsum operation:",
            f"result = tensor.einsum('{self.inputs}->{self.output}', {', '.join(self.matrix_names)})\n",
            "Input Tensors:"
        ]

        for matrix_name, subscripts in zip(self.matrix_names, self.input_subscripts):
            expected_shape = self.matrix_shapes[matrix_name]
            shape_str = f"{', '.join(f'{dim}:{size}' for dim, size in zip(subscripts, expected_shape))}"
            explanation.append(f"* {matrix_name}[{shape_str}]")

        explanation.append("\nOperations:")
        try:
            explanation.extend(self._explain_operations())

            output_shape = self._calculate_output_shape()
            output_str = f"{', '.join(f'{dim}:{size}' for dim, size in zip(self.output, output_shape))}" if output_shape else "scalar"
            explanation.append(f"\nOutput: result[{output_str}]")

            explanation.extend(self._add_specific_explanations())

        except ValueError as e:
            explanation.append(f"\nError: {str(e)}")

        return "\n".join(explanation)

    def _explain_operations(self):
        explanation = []
        total_flops = 0

        if len(self.matrix_names) == 1:
            operation_description = f"Operation: Transform {self.matrix_names[0]} to match output subscripts '{self.output}'"
        else:
            operation_description = f"Operation: Contract tensors {', '.join(self.matrix_names)} over subscripts '{self.inputs}' to get '{self.output}'"

        explanation.append(operation_description)

        # FLOPs calculation
        flops = self._calculate_total_flops()
        total_flops += flops

        # Generate detailed explanation
        detailed_explanation = self._generate_detailed_explanation()
        explanation.extend(detailed_explanation)

        explanation.append(f"\nTotal FLOPs: {total_flops:,}")
        return explanation

    def _generate_detailed_explanation(self):
        explanation = []
        dim_to_size = self._get_dim_sizes()
        summed_dims = self._get_summed_dims()

        # Explain dimensions involved in the operation
        explanation.append("\nDetailed Steps:")
        for dim in sorted(dim_to_size.keys()):
            if dim in summed_dims:
                explanation.append(f"- Dimension '{dim}' of size {dim_to_size[dim]} is summed over.")
            elif dim in self.output:
                explanation.append(f"- Dimension '{dim}' of size {dim_to_size[dim]} is retained in the output.")
            else:
                explanation.append(f"- Dimension '{dim}' is not present in the output.")

        # Generate pseudo-code
        if len(self.matrix_names) == 1:
            pseudo_code = self._generate_pseudo_code_single_input(summed_dims, dim_to_size)
        else:
            pseudo_code = self._generate_pseudo_code_multi_input(summed_dims, dim_to_size)
        explanation.append("\nPseudo-code for the operation:")
        explanation.extend(pseudo_code)

        return explanation

    def _get_summed_dims(self):
        input_dims = set(''.join(self.input_subscripts))
        output_dims = set(self.output)
        return input_dims - output_dims

    def _generate_pseudo_code_single_input(self, summed_dims, dim_to_size):
        pseudo_code = []
        dim_to_index = {dim: f"{dim}_idx" for dim in dim_to_size.keys()}
        indent = ''

        output_dims = [dim for dim in self.output]
        input_subscripts = self.input_subscripts[0]

        # Generate loops over output dimensions
        for dim in output_dims:
            index = dim_to_index[dim]
            size = dim_to_size[dim]
            pseudo_code.append(f"{indent}for {index} in range({size}):")
            indent += '    '

        # Initialize result element if summation is involved
        if summed_dims:
            result_indices = [dim_to_index[dim] for dim in output_dims]
            pseudo_code.append(f"{indent}result[{', '.join(result_indices)}] = 0  # Initialize the result element")

            # Generate loops over summed dimensions
            for dim in summed_dims:
                index = dim_to_index[dim]
                size = dim_to_size[dim]
                pseudo_code.append(f"{indent}for {index} in range({size}):")
                indent += '    '

            # Compute the result
            input_indices = [dim_to_index[dim] for dim in input_subscripts]
            output_indices = [dim_to_index[dim] for dim in self.output]
            pseudo_code.append(
                f"{indent}result[{', '.join(output_indices)}] += {self.matrix_names[0]}[{', '.join(input_indices)}]")
        else:
            # Direct assignment for transformations
            input_indices = [dim_to_index[dim] for dim in input_subscripts]
            output_indices = [dim_to_index[dim] for dim in self.output]
            pseudo_code.append(
                f"{indent}result[{', '.join(output_indices)}] = {self.matrix_names[0]}[{', '.join(input_indices)}]")

        return pseudo_code

    def _generate_pseudo_code_multi_input(self, summed_dims, dim_to_size):
        pseudo_code = []
        all_dims = set(''.join(self.input_subscripts))
        dim_to_index = {dim: f"{dim}_idx" for dim in all_dims}
        indent = ''

        output_dims = [dim for dim in self.output]

        # Generate loops over output dimensions
        for dim in output_dims:
            index = dim_to_index[dim]
            size = dim_to_size[dim]
            pseudo_code.append(f"{indent}for {index} in range({size}):")
            indent += '    '

        # Initialize result element if summation is involved
        result_indices = [dim_to_index[dim] for dim in output_dims]
        if summed_dims:
            pseudo_code.append(f"{indent}result[{', '.join(result_indices)}] = 0  # Initialize the result element")

        # Generate loops over summed dimensions
        for dim in sorted(summed_dims):
            index = dim_to_index[dim]
            size = dim_to_size[dim]
            pseudo_code.append(f"{indent}for {index} in range({size}):")
            indent += '    '

        # Compute the result
        product_terms = [
            f"{name}[{', '.join(dim_to_index[dim] for dim in subscripts)}]"
            for name, subscripts in zip(self.matrix_names, self.input_subscripts)
        ]
        if summed_dims:
            pseudo_code.append(f"{indent}result[{', '.join(result_indices)}] += " + " * ".join(product_terms))
        else:
            pseudo_code.append(f"{indent}result[{', '.join(result_indices)}] = " + " * ".join(product_terms))

        return pseudo_code

    def _calculate_total_flops(self):
        if self.operation_type == "single_input":
            return self._calculate_flops_single_input()
        elif self.operation_type == "two_input":
            return self._calculate_flops_two_input()
        else:
            return self._calculate_flops_multi_input()

    def _calculate_flops_single_input(self):
        summed_dims = self._get_summed_dims()
        if not summed_dims:
            return 0  # Transpose-like operations
        elif len(self.output) == 0:
            return len(self._get_dim_sizes()) - 1  # Trace-like operations
        else:
            return self._calculate_sum_flops()  # Sum over axis

    def _calculate_flops_two_input(self):
        summed_dims = self._get_summed_dims()
        if not summed_dims:
            return self._calculate_outer_product_flops()
        else:
            return self._calculate_matrix_multiply_flops()

    def _calculate_flops_multi_input(self):
        return self._calculate_complex_flops()

    def _calculate_sum_flops(self):
        dim_sizes = self._get_dim_sizes()
        output_size = 1
        for dim in self.output:
            output_size *= dim_sizes[dim]
        total_size = 1
        for dim in self.input_subscripts[0]:
            total_size *= dim_sizes[dim]
        return total_size - output_size  # Subtracting output_size to account for the first assignment

    def _calculate_outer_product_flops(self):
        return self._get_output_size()  # One multiplication per output element

    def _calculate_matrix_multiply_flops(self):
        dim_sizes = self._get_dim_sizes()
        output_size = self._get_output_size()
        summed_dim = list(self._get_summed_dims())[0]
        return output_size * dim_sizes[
            summed_dim] * 2 - output_size  # Multiplications and additions, minus initial assignments

    def _calculate_complex_flops(self):
        dim_sizes = self._get_dim_sizes()
        output_size = self._get_output_size()
        summed_dims = self._get_summed_dims()
        elements_per_sum = 1
        for dim in summed_dims:
            elements_per_sum *= dim_sizes[dim]
        return output_size * elements_per_sum * (2 * len(
            self.matrix_names) - 1) - output_size  # Multiplications and additions, minus initial assignments

    def _get_dim_sizes(self):
        dim_sizes = {}
        for subscripts, shape in zip(self.input_subscripts, self.matrix_shapes_ordered):
            for dim, size in zip(subscripts, shape):
                if dim in dim_sizes and dim_sizes[dim] != size:
                    raise ValueError(f"Inconsistent sizes for dimension '{dim}': {dim_sizes[dim]} vs {size}")
                dim_sizes[dim] = size
        return dim_sizes

    def _get_output_size(self):
        dim_sizes = self._get_dim_sizes()
        output_size = 1
        for dim in self.output:
            output_size *= dim_sizes[dim]
        return output_size

    def _get_dim_sizes_single_input(self):
        dim_sizes = {}
        subscripts = self.input_subscripts[0]
        shape = self.matrix_shapes_ordered[0]
        for dim, size in zip(subscripts, shape):
            dim_sizes[dim] = size
        return dim_sizes

    def _calculate_output_shape(self):
        output_shape = []
        dim_sizes = self._get_dim_sizes()
        for dim in self.output:
            if dim not in dim_sizes:
                raise ValueError(f"Dimension '{dim}' not found in any input")
            output_shape.append(dim_sizes[dim])
        return output_shape

    def _add_specific_explanations(self):
        explanations = []

        input_dims = set(''.join(self.input_subscripts))
        output_dims = set(self.output)
        summed_dims = input_dims - output_dims

        if not self.output:
            explanations.append("\nNote: This operation results in a scalar (0-dimensional) output.")
        elif summed_dims:
            explanations.append(f"\nNote: This operation involves summation over dimensions {sorted(summed_dims)}.")
        elif len(self.matrix_names) == 1:
            explanations.append("\nNote: This operation rearranges the input tensor without arithmetic computations.")
        elif len(self.matrix_names) > 1:
            explanations.append("\nNote: This operation performs element-wise multiplication without any summation.")
        else:
            explanations.append("\nNote: This operation rearranges the input tensor.")
        return explanations


def test():
    def run_test(name, einsum_expr, matrix_specs):
        print(f"\n{name}:")
        einsum_explainer = EinsumExplainer(einsum_expr, *matrix_specs)
        explanation = einsum_explainer.parse_einsum()
        print(explanation)
        print("=" * 50)

    # 1. Dot product
    run_test("Dot Product",
             "i,i->",
             [('a', [3]), ('b', [3])])

    # 2. Matrix-vector product
    run_test("Matrix-Vector Product",
             "ij,j->i",
             [('matrix', [2, 3]), ('vector', [3])])

    # 3. Matrix multiplication
    run_test("Matrix Multiplication",
             "ik,kj->ij",
             [('matrix1', [2, 3]), ('matrix2', [3, 4])])

    # 4. Outer product
    run_test("Outer Product",
             "i,j->ij",
             [('vector1', [3]), ('vector2', [4])])

    # 5. Transpose
    run_test("Transpose",
             "ij->ji",
             [('matrix', [2, 3])])

    # 6. Sum over axis
    run_test("Sum Over Axis",
             "ij->i",
             [('matrix', [2, 3])])

    # 7. Batch matrix multiplication
    run_test("Batch Matrix Multiplication",
             "bij,bjk->bik",
             [('batch_matrix1', [5, 2, 3]), ('batch_matrix2', [5, 3, 4])])

    # 8. Bilinear transformation
    run_test("Bilinear Transformation",
             "bn,anm,bm->ba",
             [('input', [10, 5]), ('weights', [8, 5, 6]), ('input2', [10, 6])])

    # 9. Tensor contraction
    run_test("Tensor Contraction",
             "abcd,cdef->abef",
             [('tensor1', [2, 3, 4, 5]), ('tensor2', [4, 5, 6, 7])])

    # 10. Diagonal
    run_test("Diagonal",
             "ii->i",
             [('matrix', [3, 3])])

    # 11. Trace
    run_test("Trace",
             "ii->",
             [('matrix', [3, 3])])

    # 12. Complex example (correct)
    run_test("Complex Example (Correct)",
             "bclhn,bcshn,bhcls,bcshp->bclhp",
             [('expanded_output_transformation_weights', [2, 3, 4, 5, 6]),
              ('expanded_state_transformation_weights', [2, 3, 7, 5, 6]),
              ('intra_attention_weights', [2, 5, 3, 4, 7]),
              ('reshaped_input', [2, 3, 7, 5, 8])])

    # 13. Example with incompatible shapes
    run_test("Example with Incompatible Shapes",
             "ab,bc,cd->ad",
             [('matrix_a', [2, 3]),
              ('matrix_b', [3, 4]),
              ('matrix_c', [5, 6])])


if __name__ == '__main__':
    test()
