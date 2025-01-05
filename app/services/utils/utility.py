from keras import backend as K
from keras import layers

class FixedDropout(layers.Dropout):
    """
    This class implements a custom version of the Keras `Dropout` layer called `FixedDropout`.
    
    It overrides the `_get_noise_shape` method to customize the noise shape computation for the dropout layer.
    
    Attributes:
        noise_shape (tuple): The shape of the binary dropout mask that will be multiplied with the input.
                              It is used to specify the dimensions for which dropout should be applied.
    """
    
    def _get_noise_shape(self, inputs) -> tuple:
        """
        Customizes the dropout mask shape based on the provided `noise_shape` argument.
        
        This method calculates the noise shape for the input tensor based on the dimensions of the input
        and the specified `noise_shape`. It overrides the base class method to provide a flexible 
        implementation where each dimension can either keep its shape or be `None` to indicate a drop.
        
        :param inputs: The input tensor to which the dropout mask will be applied.
        :type inputs: Keras tensor
        :return: The noise shape tuple that will be used to generate the dropout mask.
        :rtype: tuple
        :raises TypeError: If the `noise_shape` is not set correctly.
        """
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)
