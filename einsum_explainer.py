import operator
from functools import reduce


class EinsumExplainer:
    def __init__(self, einsum_expression: str, *matrix_specs):
        self.inputs, self.output = einsum_expression.split('->')
        self.input_subscripts = self.inputs.split(',')
        self.matrix_names = [spec[0] for spec in matrix_specs]
        self.matrix_shapes = {spec[0]: spec[1] for spec in matrix_specs}
        self.output_dims = set(self.output)

    def parse_einsum(self):
        explanation = [
            f"Explaining einsum operation:",
            f"result = tensor.einsum('{self.inputs}->{self.output}', {', '.join(self.matrix_names)})\n",
            "Input Matrices:"
        ]

        for matrix_name, subscripts in zip(self.matrix_names, self.input_subscripts):
            expected_shape = self.matrix_shapes[matrix_name]
            shape_str = f"{', '.join(f'{dim}:{size}' for dim, size in zip(subscripts, expected_shape))}"
            explanation.append(f"* {matrix_name}[{shape_str}]")

        explanation.append("\nOperations:")
        try:
            explanation.extend(self._explain_recursive(self.input_subscripts, self.matrix_names, self.matrix_shapes))

            output_shape = self._calculate_output_shape()
            output_str = f"{', '.join(f'{dim}:{size}' for dim, size in zip(self.output, output_shape))}" if output_shape else "scalar"
            explanation.append(f"\nOutput: result[{output_str}]")

            explanation.extend(self._add_specific_explanations())

        except ValueError as e:
            explanation.append(f"\nError: {str(e)}")

        return "\n".join(explanation)

    def _explain_recursive(self, input_subscripts, matrix_names, matrix_shapes):
        explanation = []
        remaining_subscripts = list(input_subscripts)
        remaining_matrices = list(matrix_names)
        remaining_shapes = [matrix_shapes[name] for name in matrix_names]
        total_flops = 0

        while len(remaining_matrices) > 1:
            num_inputs = min(3, len(remaining_matrices))  # Handle up to 3 inputs at once
            current_subscripts = remaining_subscripts[:num_inputs]
            current_matrices = remaining_matrices[:num_inputs]
            current_shapes = remaining_shapes[:num_inputs]

            new_subscripts, summed_dims = self._calculate_output_subscripts(*current_subscripts)

            try:
                self._validate_shapes(current_subscripts, current_shapes, summed_dims)
            except ValueError as e:
                explanation.append(f"Error: {str(e)}")
                return explanation

            new_shape = self._calculate_intermediate_shape(new_subscripts, current_subscripts, current_shapes)

            input_str = ", ".join([f"{matrix}[{', '.join(f'{dim}:{size}' for dim, size in zip(subs, shape))}]"
                                   for matrix, subs, shape in
                                   zip(current_matrices, current_subscripts, current_shapes)])
            new_str = f"intermediate[{', '.join(f'{dim}:{size}' for dim, size in zip(new_subscripts, new_shape))}]"

            operation_description = f"{len(explanation) + 1}. Tensor contraction({input_str}) -> {new_str}"
            if summed_dims:
                operation_description += f" (summing over {', '.join(sorted(summed_dims))})"
            else:
                operation_description += " (element-wise operation, no summation)"

            # Calculate FLOPs for this operation
            flops = self._calculate_flops(current_shapes, new_shape, summed_dims)
            total_flops += flops
            operation_description += f" [FLOPs: {flops:,}]"

            explanation.append(operation_description)

            remaining_matrices = ['intermediate'] + remaining_matrices[num_inputs:]
            remaining_subscripts = [new_subscripts] + remaining_subscripts[num_inputs:]
            remaining_shapes = [new_shape] + remaining_shapes[num_inputs:]

        # Handle single-input operations
        if len(remaining_matrices) == 1:
            input_subscripts = remaining_subscripts[0]
            output_subscripts = self.output
            input_shape = remaining_shapes[0]
            output_shape = self._calculate_output_shape()
            summed_dims = set(input_subscripts) - set(output_subscripts)

            flops = self._calculate_flops([input_shape], output_shape, summed_dims)
            total_flops += flops

            operation_description = f"{len(explanation) + 1}. Single-input operation: {remaining_matrices[0]} -> result"
            if summed_dims:
                operation_description += f" (summing over {', '.join(sorted(summed_dims))})"
            operation_description += f" [FLOPs: {flops:,}]"

            if flops == 0:
                operation_description += " (This operation is a simple reshape or view, requiring no arithmetic operations)"
            elif flops < len(input_shape) * 2:
                operation_description += " (This operation involves selective element access or simple reduction)"

            explanation.append(operation_description)

        explanation.append(f"\nTotal FLOPs: {total_flops:,}")
        return explanation

    def _add_specific_explanations(self):
        explanations = []

        input_dims = set(''.join(self.input_subscripts))
        output_dims = set(self.output)
        summed_dims = input_dims - output_dims

        if not self.output:
            explanations.append("\nNote: This operation results in a scalar (0-dimensional) output.")
        elif self.inputs == self.output:
            explanations.append("\nNote: This operation does not change the structure of the input.")
        elif len(self.input_subscripts) == 1 and len(self.input_subscripts[0]) == 2 and len(self.output) == 2 and \
                self.input_subscripts[0] != self.output:
            explanations.append("\nNote: This operation transposes the input matrix, swapping its dimensions. This is a memory operation and doesn't involve arithmetic computations.")
        elif input_dims and not output_dims:
            explanations.append(f"\nNote: This operation sums over all dimensions {sorted(input_dims)}, resulting in a scalar output.")
        elif len(self.input_subscripts) == 1 and len(set(self.input_subscripts[0])) == 1 and len(self.output) == 1:
            explanations.append("\nNote: This operation extracts the diagonal elements of the input matrix. This involves selective memory access rather than arithmetic operations.")
        elif len(self.input_subscripts) == 1 and len(set(self.input_subscripts[0])) == 1 and not self.output:
            explanations.append("\nNote: This operation computes the trace of the input matrix (sum of diagonal elements).")
        elif ',' in self.inputs and input_dims == output_dims:
            explanations.append("\nNote: This operation performs element-wise multiplication without any summation.")
        elif ',' in self.inputs and input_dims != output_dims:
            explanations.append(f"\nNote: This operation involves both element-wise multiplication and summation over dimensions {sorted(summed_dims)}.")
        elif input_dims != output_dims:
            explanations.append(f"\nNote: This operation involves summation over dimensions {sorted(summed_dims)}.")

        if len(self.input_subscripts) > 2:
            explanations.append("\nNote: This is a multi-step operation involving multiple tensor contractions.")

        return explanations

    def _calculate_flops(self, shapes, output_shape, summed_dims):
        input_sizes = [reduce(operator.mul, shape, 1) for shape in shapes]
        output_size = reduce(operator.mul, output_shape, 1)

        if not summed_dims:
            return max(sum(input_sizes), output_size)

        summed_size = 1
        for dim in summed_dims:
            dim_size = None
            for subscripts, shape in zip(self.input_subscripts, shapes):
                if dim in subscripts:
                    dim_index = subscripts.index(dim)
                    if dim_index < len(shape):
                        dim_size = shape[dim_index]
                        break
            if dim_size is None:
                # If the dimension is not found in any input, assume it has size 1
                dim_size = 1
            summed_size *= dim_size

        return 2 * output_size * summed_size

    def _calculate_output_subscripts(self, *input_subscripts):
        all_dims = set(''.join(input_subscripts))
        common_dims = set.intersection(*map(set, input_subscripts))

        summed_dims = common_dims - self.output_dims
        dims_to_keep = (all_dims - summed_dims) | (self.output_dims & all_dims)

        original_order = self.output + ''.join(set(''.join(input_subscripts)) - set(self.output))
        intermediate_dims = ''.join(dim for dim in original_order if dim in dims_to_keep)

        return intermediate_dims, summed_dims

    def _calculate_intermediate_shape(self, new_subscripts, input_subscripts, input_shapes):
        new_shape = []
        for dim in new_subscripts:
            for subscripts, shape in zip(input_subscripts, input_shapes):
                if dim in subscripts:
                    new_shape.append(shape[subscripts.index(dim)])
                    break
        return new_shape

    def _calculate_output_shape(self):
        output_shape = []
        for dim in self.output:
            size = None
            for subscripts, shape in zip(self.input_subscripts, self.matrix_shapes.values()):
                if dim in subscripts:
                    dim_size = shape[subscripts.index(dim)]
                    if size is None:
                        size = dim_size
                    elif size != dim_size:
                        raise ValueError(f"Inconsistent sizes for dimension '{dim}': found {size} and {dim_size}")
            if size is None:
                raise ValueError(f"Dimension '{dim}' not found in any input")
            output_shape.append(size)
        return output_shape

    def _validate_shapes(self, subscripts, shapes, summed_dims):
        dim_sizes = {}
        for subs, shape in zip(subscripts, shapes):
            for dim, size in zip(subs, shape):
                if dim in dim_sizes and dim_sizes[dim] != size:
                    raise ValueError(f"Incompatible sizes for dimension '{dim}': {dim_sizes[dim]} != {size}")
                dim_sizes[dim] = size

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
              ('matrix_c', [5, 6])])  # This should be [4, 6] to be compatible


if __name__ == '__main__':
    test()