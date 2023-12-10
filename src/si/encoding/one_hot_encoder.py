import numpy as np


class OneHotEncoder:
    """
    OneHotEncoder class for encoding and decoding sequences using one-hot encoding.

    Parameters
    ----------
    padder : str, optional
        The character to perform padding with, by default ' '.
    max_length : int, optional
        Maximum length of the sequences, by default None.

    Attributes
    ----------
    padder : str
        The character used for padding.
    max_length : int
        Maximum length of the sequences.
    alphabet : list
        The unique characters in the sequences.
    char_to_index : dict
        Dictionary mapping characters in the alphabet to unique integers.
    index_to_char : dict
        Reverse of char_to_index, mapping integers to characters.

    Methods
    -------
    fit(data)
        Fits the encoder to the data (learns the alphabet, char_to_index, and index_to_char).
    transform(data)
        Encodes the sequence to one-hot encoding.
    fit_transform(data)
        Runs fit and then transform.
    inverse_transform(data)
        Converts one-hot-encoded sequences back to sequences.
    """

    def __init__(self, padder=' ', max_length=None):
        self.padder = padder
        self.max_length = max_length
        self.alphabet = None
        self.char_to_index = None
        self.index_to_char = None

    def fit(self, data):
        """
        Fit the encoder to the data.

        Parameters
        ----------
        data : list of str
            List of sequences (strings) to learn from.
        """
        # Learn the alphabet, char_to_index, and index_to_char
        sequences = [list(seq) for seq in data]
        self.alphabet = list(set([char for seq in sequences for char in seq]))
        self.char_to_index = {char: index for index, char in enumerate(self.alphabet)}
        self.index_to_char = {index: char for index, char in enumerate(self.alphabet)}

        # Set max_length if not defined
        if self.max_length is None:
            self.max_length = max(len(seq) for seq in sequences)

    def transform(self, data):
        """
        Encode the sequence to one-hot encoding.

        Parameters
        ----------
        data : list of str
            Data to encode.

        Returns
        -------
        list of numpy.ndarray
            List of one-hot encoded matrices.
        """
        # Trim, pad, and encode the data to one-hot encoding
        sequences = [list(seq)[:self.max_length] for seq in data]
        sequences = [seq + [self.padder] * (self.max_length - len(seq)) for seq in sequences]
        encoded_sequences = []

        for seq in sequences:
            encoded_seq = np.zeros((len(seq), len(self.alphabet)))
            for i, char in enumerate(seq):
                if char in self.char_to_index:
                    encoded_seq[i, self.char_to_index[char]] = 1
            encoded_sequences.append(encoded_seq)

        return encoded_sequences

    def fit_transform(self, data):
        """
        Run fit and then transform.

        Parameters
        ----------
        data : list of str
            List of sequences (strings) to learn from.

        Returns
        -------
        list of numpy.ndarray
            List of one-hot encoded matrices.
        """
        # Run fit and then transform
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        """
        Convert one-hot encoded matrices back to sequences.

        Parameters
        ----------
        data : list of numpy.ndarray
            Data to decode (one-hot encoded matrices).

        Returns
        -------
        list of str
            List of decoded sequences.
        """
        # Convert one-hot encoded matrices back to sequences
        decoded_sequences = []

        for encoded_seq in data:
            decoded_seq = [self.index_to_char[index] for index in np.argmax(encoded_seq, axis=1)]
            decoded_sequences.append(''.join(decoded_seq))

        return decoded_sequences


if __name__ == '__main__':
    # Example
    data = ['abc', 'aabb', 'ghi']
    encoder = OneHotEncoder(padder=' ', max_length=5)
    encoded_data = encoder.fit_transform(data)
    print("Encoded Data:")
    print(encoded_data)

    decoded_data = encoder.inverse_transform(encoded_data)
    print("\nDecoded Data:")
    print(decoded_data)
