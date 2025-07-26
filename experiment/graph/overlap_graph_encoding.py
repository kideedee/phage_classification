import numpy as np


def _convert_nucleotide_to_number(sequence: str):
    result = sequence.replace("A", "0").replace("C", "1").replace("G", "2").replace("T", "3")
    return ''.join(list(filter(str.isdigit, result)))


def _gen_encoded_kmers(sequence: str, k: int):
    encoded_kmers = []
    for i in range(0, len(sequence) - k + 1):
        encoded_kmers.append(int(sequence[i:i + k], 4))

    return encoded_kmers


def convert_sequence_to_matrix(k: int, encoded_kmers: list, shift_list: list):
    matrix = np.zeros((4 ** k, 4 ** k))
    num_transition = 0
    for shift in shift_list:
        for i in range(0, len(encoded_kmers) - shift):
            from_mer = encoded_kmers[i]
            to_mer = encoded_kmers[i + shift]
            matrix[from_mer][to_mer] += 1
        num_transition += len(encoded_kmers) - 2 * k - (k - shift) + 1

    matrix = matrix / num_transition
    new_matrix = matrix.flatten()
    return new_matrix


def gpg_convert_sequence_to_matrix(encoded_kmers: list, dis_list: list, k: int):
    matrix = np.zeros((4 ** k, 4 ** k))
    num = 0
    length = len(encoded_kmers)
    for dis in dis_list:
        for i in range(0, len(encoded_kmers) - k - dis):
            matrix[encoded_kmers[i]][encoded_kmers[i + k + dis]] += 1
        num = num + (length - 2 * k - dis + 1.0)

    matrix = matrix / num

    new_matrix = matrix.flatten()

    return new_matrix


def gpg_create_matrix_feature(tuple):
    sequence, k, d = tuple
    encoded_sequence = _convert_nucleotide_to_number(sequence)
    encoded_kmers = _gen_encoded_kmers(encoded_sequence, k)

    dis = [
        list(range(0, 1)),
        list(range(1, 2)),
        list(range(2, 3)),
        list(range(3, 5)),
        list(range(5, 9)),
        list(range(9, 17)),
        list(range(17, 33)),
        list(range(33, 65))
    ]
    if d == 1:
        feature = np.hstack((gpg_convert_sequence_to_matrix(encoded_kmers, list(range(0, 1)), k)))

    elif d == 2:
        feature = np.hstack((
            gpg_convert_sequence_to_matrix(encoded_kmers, list(range(0, 1)), k),
            gpg_convert_sequence_to_matrix(encoded_kmers, list(range(1, 2)), k)))
    else:
        feature = np.hstack((
            gpg_convert_sequence_to_matrix(encoded_kmers, list(range(0, 1)), k),
            gpg_convert_sequence_to_matrix(encoded_kmers, list(range(1, 2)), k)))
        for i in range(2, d):
            feature = np.hstack((feature, gpg_convert_sequence_to_matrix(encoded_kmers, dis[i], k)))

    return feature * 100


def create_matrix_feature(tuple):
    sequence, k, _ = tuple
    encoded_sequence = _convert_nucleotide_to_number(sequence)
    encoded_kmers = _gen_encoded_kmers(encoded_sequence, k)

    shift = [
        # list(range(0, 1)),
        list(range(1, 2)),
        list(range(2, 3)),
        list(range(3, 5)),
        list(range(5, 9)),
        list(range(9, 17)),
        list(range(17, 33)),
        list(range(33, 65))
    ]

    if k == 2:
        feature = convert_sequence_to_matrix(k, encoded_kmers, shift[0])
    elif k == 3:
        feature_1 = convert_sequence_to_matrix(k, encoded_kmers, shift[0])
        feature_2 = convert_sequence_to_matrix(k, encoded_kmers, shift[1])
        feature = np.hstack([feature_1, feature_2])
    else:
        feature_1 = convert_sequence_to_matrix(k, encoded_kmers, shift[0])
        feature_2 = convert_sequence_to_matrix(k, encoded_kmers, shift[1])
        feature = np.hstack([feature_1, feature_2])
        for i in range(3, k):
            feature_i = convert_sequence_to_matrix(k, encoded_kmers, shift[i])
            feature = np.hstack([feature, feature_i])

    return feature * 100

# if __name__ == '__main__':
#     dna = "ATCGAAG"
#     k = 3
#     flattened_matrix = create_matrix_feature(dna, k)
#     print(flattened_matrix)
#     temp = flattened_matrix.reshape(-1, k - 1, 4 ** (k * 2))
#     print(temp.shape)
