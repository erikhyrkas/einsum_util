from unittest import expectedFailure


class EinsumGeneratorUtil:
    def __init__(self):
        pass

    class Matrix:
        def __init__(self, name: str, dims: list, shape: list):
            """
            Initialize a matrix with a name, dimension names (dims), and shape.
            :param name: Name of the matrix (for use in the einsum expression)
            :param dims: List of dimension names (e.g., ['i', 'j'])
            :param shape: List of corresponding dimension sizes (e.g., [3, 4])
            """
            self.name = name
            self.dims = dims  # List of dimension names
            self.shape = shape  # List of dimension sizes

        def validate_shape(self, other_matrix, common_dims):
            """
            Validate that shared dimensions have the same size.
            """
            for dim in common_dims:
                if dim in self.dims and dim in other_matrix.dims:
                    self_dim_size = self.shape[self.dims.index(dim)]
                    other_dim_size = other_matrix.shape[other_matrix.dims.index(dim)]
                    if self_dim_size != other_dim_size:
                        raise ValueError(
                            f"Dimension size mismatch: {dim} has sizes {self_dim_size} and {other_dim_size}")

    def dot_product(self, matrix_a, matrix_b):
        """Dot product of two vectors."""
        matrix_a.validate_shape(matrix_b, 'i')
        einsum_expression = f"{''.join(matrix_a.dims)},{''.join(matrix_b.dims)}->"
        return f"torch.einsum('{einsum_expression}', [{matrix_a.name}, {matrix_b.name}])"

    def matrix_vector_product(self, matrix, vector):
        """Matrix-vector multiplication."""
        common_dim = 'j'  # Common dimension for multiplication
        matrix.validate_shape(vector, common_dim)
        einsum_expression = f"{''.join(matrix.dims)},{''.join(vector.dims)}->{matrix.dims[0]}"
        return f"torch.einsum('{einsum_expression}', [{matrix.name}, {vector.name}])"

    def matrix_multiplication(self, matrix_a, matrix_b):
        """Matrix-matrix multiplication."""
        common_dim = 'j'  # Shared dimension
        matrix_a.validate_shape(matrix_b, common_dim)
        einsum_expression = f"{''.join(matrix_a.dims)},{''.join(matrix_b.dims)}->{matrix_a.dims[0]}{matrix_b.dims[1]}"
        return f"torch.einsum('{einsum_expression}', [{matrix_a.name}, {matrix_b.name}])"

    def outer_product(self, vector_a, vector_b):
        """Outer product of two vectors."""
        einsum_expression = f"{''.join(vector_a.dims)},{''.join(vector_b.dims)}->{''.join(vector_a.dims)}{''.join(vector_b.dims)}"
        return f"torch.einsum('{einsum_expression}', [{vector_a.name}, {vector_b.name}])"

    def transpose(self, matrix, new_dims):
        """Transpose a matrix."""
        if len(new_dims) != len(matrix.dims):
            raise ValueError("New dimensions must match the number of dimensions in the matrix.")
        einsum_expression = f"{''.join(matrix.dims)}->{''.join(new_dims)}"
        return f"torch.einsum('{einsum_expression}', [{matrix.name}])"

    def sum_over_axis(self, matrix, axis):
        """Sum a matrix over a specific axis."""
        dims_without_axis = ''.join([dim for i, dim in enumerate(matrix.dims) if i != axis])
        einsum_expression = f"{''.join(matrix.dims)}->{dims_without_axis}"
        return f"torch.einsum('{einsum_expression}', [{matrix.name}])"

    def multi_einsum(self, einsum_expression, *matrices):
        """Handle multi-dimensional einsum expressions."""
        matrix_names = ', '.join([matrix.name for matrix in matrices])
        return f"torch.einsum('{einsum_expression}', [{matrix_names}])"


def test_dot_product():
    einsum_util = EinsumGeneratorUtil()
    a = einsum_util.Matrix('a', ['i'], [3])
    b = einsum_util.Matrix('b', ['i'], [3])
    result = einsum_util.dot_product(a, b)
    expected = "torch.einsum('i,i->', [a, b])"
    print(result, expected)
    assert result == expected


def test_matrix_vector_product():
    einsum_util = EinsumGeneratorUtil()
    A = einsum_util.Matrix('A', ['i', 'j'], [3, 4])
    x = einsum_util.Matrix('x', ['j'], [4])
    result = einsum_util.matrix_vector_product(A, x)
    expected = "torch.einsum('ij,j->i', [A, x])"
    print(result, expected)
    assert result == expected


def test_matrix_multiplication():
    einsum_util = EinsumGeneratorUtil()
    A = einsum_util.Matrix('A', ['i', 'j'], [3, 4])
    B = einsum_util.Matrix('B', ['j', 'k'], [4, 5])
    result = einsum_util.matrix_multiplication(A, B)
    expected = "torch.einsum('ij,jk->ik', [A, B])"
    print(result, expected)
    assert result == expected


def test_outer_product():
    einsum_util = EinsumGeneratorUtil()
    x = einsum_util.Matrix('x', ['i'], [3])
    y = einsum_util.Matrix('y', ['j'], [4])
    result = einsum_util.outer_product(x, y)
    expected = "torch.einsum('i,j->ij', [x, y])"
    print(result, expected)
    assert result == expected


def test_transpose():
    einsum_util = EinsumGeneratorUtil()
    A = einsum_util.Matrix('A', ['i', 'j'], [3, 4])
    result = einsum_util.transpose(A, ['j', 'i'])
    expected = "torch.einsum('ij->ji', [A])"
    print(result, expected)
    assert result == expected


def test_sum_over_axis():
    einsum_util = EinsumGeneratorUtil()
    A = einsum_util.Matrix('A', ['i', 'j'], [3, 4])
    result = einsum_util.sum_over_axis(A, 1)
    expected = "torch.einsum('ij->i', [A])"
    print(result, expected)
    assert result == expected


def test_intra_block_output():
    einsum_util = EinsumGeneratorUtil()
    expanded_output_transformation_weights = einsum_util.Matrix('expanded_output_transformation_weights',
                                                                ['b', 'c', 'l', 'h', 'n'], [2, 3, 4, 5, 6])
    expanded_state_transformation_weights = einsum_util.Matrix('expanded_state_transformation_weights',
                                                               ['b', 'c', 's', 'h', 'n'], [2, 3, 7, 5, 6])
    intra_attention_weights = einsum_util.Matrix('intra_attention_weights', ['b', 'h', 'c', 'l', 's'], [2, 5, 3, 4, 7])
    reshaped_input = einsum_util.Matrix('reshaped_input', ['b', 'c', 's', 'h', 'p'], [2, 3, 7, 5, 6])

    result = einsum_util.multi_einsum("bclhn,bcshn,bhcls,bcshp->bclhp",
                                      expanded_output_transformation_weights,
                                      expanded_state_transformation_weights,
                                      intra_attention_weights,
                                      reshaped_input)

    expected = "torch.einsum('bclhn,bcshn,bhcls,bcshp->bclhp', [expanded_output_transformation_weights, expanded_state_transformation_weights, intra_attention_weights, reshaped_input])"
    print(result, expected)
    assert result == expected


def test_inter_block_states():
    einsum_util = EinsumGeneratorUtil()
    decay_chunk = einsum_util.Matrix('decay_chunk', ['b', 'h', 'z', 'c'], [2, 4, 5, 6])
    states = einsum_util.Matrix('states', ['b', 'c', 'h', 'p', 'n'], [2, 3, 6, 7, 5])

    result = einsum_util.multi_einsum("bhzc,bchpn->bzhpn", decay_chunk, states)

    expected = "torch.einsum('bhzc,bchpn->bzhpn', [decay_chunk, states])"
    print(result, expected)
    assert result == expected


def test_QK_scores():
    einsum_util = EinsumGeneratorUtil()
    block = einsum_util.Matrix('block', ['b', 't', 'e'], [2, 3, 4])
    C = einsum_util.Matrix('C', ['e', 'd'], [4, 5])
    B = einsum_util.Matrix('B', ['e', 'd'], [4, 5])

    # Q calculation
    Q_result = einsum_util.multi_einsum('bte,ed->btd', block, C)
    Q_expected = "torch.einsum('bte,ed->btd', [block, C])"
    print(Q_result, Q_expected)
    assert Q_result == Q_expected

    # K calculation
    K_result = einsum_util.multi_einsum('bte,ed->btd', block, B)
    K_expected = "torch.einsum('bte,ed->btd', [block, B])"
    print(K_result, K_expected)
    assert K_result == K_expected

    # Scores calculation
    Q = einsum_util.Matrix('Q', ['b', 't', 'h', 'c'], [2, 3, 4, 5])
    K = einsum_util.Matrix('K', ['b', 's', 'h', 'c'], [2, 3, 4, 5])
    scores_result = einsum_util.multi_einsum('bthc,bshc->btsh', Q, K)
    scores_expected = "torch.einsum('bthc,bshc->btsh', [Q, K])"
    print(scores_result, scores_expected)
    assert scores_result == scores_expected


def test_block_proj():
    einsum_util = EinsumGeneratorUtil()
    block = einsum_util.Matrix('block', ['b', 't', 'e'], [2, 3, 4])
    B_proj = einsum_util.Matrix('B_proj', ['e', 'h', 'd'], [4, 5, 6])

    result = einsum_util.multi_einsum('bte,ehd->bthd', block, B_proj)

    expected = "torch.einsum('bte,ehd->bthd', [block, B_proj])"
    print(result, expected)
    assert result == expected


def test_block_z_y():
    einsum_util = EinsumGeneratorUtil()
    block = einsum_util.Matrix('block', ['b', 't', 'e'], [2, 3, 4])
    B = einsum_util.Matrix('B', ['e', 'h', 'c'], [4, 5, 6])
    C = einsum_util.Matrix('C', ['e', 'h', 'c'], [4, 5, 6])
    h = einsum_util.Matrix('h', ['b', 't', 'h', 'c'], [2, 3, 5, 6])

    # z calculation
    z_result = einsum_util.multi_einsum('bte,ehc->bthc', block, B)
    z_expected = "torch.einsum('bte,ehc->bthc', [block, B])"
    print(z_result, z_expected)
    assert z_result == z_expected

    # y calculation
    y_result = einsum_util.multi_einsum('ehc,bthc->bte', C, h)
    y_expected = "torch.einsum('ehc,bthc->bte', [C, h])"
    print(y_result, y_expected)
    assert y_result == y_expected

def run_generator_tests():
    test_dot_product()
    test_matrix_vector_product()
    test_matrix_multiplication()
    test_outer_product()
    test_transpose()
    test_sum_over_axis()
    test_intra_block_output()
    test_inter_block_states()
    test_QK_scores()
    test_block_proj()
    test_block_z_y()

if __name__ == '__main__':
    run_generator_tests()