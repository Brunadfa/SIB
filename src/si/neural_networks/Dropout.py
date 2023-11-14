import numpy as np

from si.neural_networks.layers import Layer


class Dropout(Layer):

    def __init__(self, probability: float):
        """
        Initialize the dropout layer.

        Parameters
        ----------
        probability: float
            The dropout rate, between 0 and 1.
        """
        super().__init__()
        self.probability = probability
        self.mask = None
        self.input = None
        self.output = None

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        if training:
            # Compute scaling factor
            scaling_factor = 1.0 / (1.0 - self.probability)
            # Compute mask using binomial distribution
            self.mask = (np.random.rand(*input.shape) > self.probability).astype(float)
            # Compute output with dropout
            self.output = input * self.mask * scaling_factor
            return self.output
        else:
            # Inference mode, return the input as is
            return input

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: numpy.ndarray
            The output error of the layer.

        Returns
        -------
        numpy.ndarray
            The input error of the layer.
        """
        # Multiply the output error by the mask
        input_error = output_error * self.mask
        return input_error

    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        return self.input_shape()

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return 0


if __name__ == '__main__':
    # Create a Dropout layer with a probability of 0.5 (you can adjust this)
    dropout_layer = Dropout(probability=0.5)
    # Generate a random input for testing
    random_input = np.random.rand(5, 5)
    # Perform forward propagation during training mode
    output_training = dropout_layer.forward_propagation(random_input, training=True)
    # Perform forward propagation during inference mode
    output_inference = dropout_layer.forward_propagation(random_input, training=False)

    print("Random Input:")
    print(random_input)
    print("\nOutput during Training:")
    print(output_training)
    print("\nOutput during Inference:")
    print(output_inference)