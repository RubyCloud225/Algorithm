from encoding_process import NN_encoder

class SmoothQuantArgs:
    def __init__(self, num_bits_weight: int, num_bits_activation: int, smooth_quant: bool, quant_range: int):
        """
        Initialize the Smooth Quant Args instance

        Args: 
            num_bits_weight:The number of bits to use for weight quantization.
            num_bits_activation: The number of bits to use for activation quantization.
            smooth_quant: Whether to use smooth quantization or not.
        """
        self.num_bits_weight = num_bits_weight
        self.num_bits_activation = num_bits_activation
        self.smooth_quant = smooth_quant
        self.quant_range = quant_range

class SmoothQuant:
    def __init__(self, args: SmoothQuantArgs):
        self.args = args

    def quantize(self, nn: NN_encoder) -> 'NN_encoder':
        """
        Quantize the weights and activations of the neural network using smooth quantization.

        Args:
            nn: The custom neural network to quantize

        Returns: 
        The quantized neural network
        """

        for layer in nn.layers:
            self.quantize_layer(layer)
        return nn
    
    def quantize_layer(self, layer: NN_encoder) -> None:
        """
        Quantize the input tensor using smooth quantization

        Args:
            layer: The layer to quantize.
        """
        #Quantize Weights
        weights = layer.weights
        min_val, max_val = weights.min(), weights.max()
        scale = (max_val - min_val) / (self.args.quant_range - 1)
        zero_point = -min_val / scale
        quantized_weights = (weights / scale + zero_point).round().clamp(0, self.args.quant_range - 1)
        layer.weights = quantized_weights

        #Quantize activations
        activations = layer.activations
        min_val, max_val = activations.min(), activations.max()
        scale = (max_val - min_val) / (self.args.quant_range - 1)
        zero_point = -min_val / scale
        quantized_activations = (activations / scale + zero_point).round().clamp(0, self.args.quant_range - 1)
        layer.activations = quantized_activations

    def dequantize(self,  nn: NN_encoder) -> NN_encoder:
        """
        Dequantize the weights and activations of the neural network using smooth dequantization

        args:
            nn: The quantized neural network to dequantize

        Returns: 
            The dequantized Neural Network

        """

        for layer in nn.layers:
            self.dequantize_layer(layer)
        return nn
    
    def dequantize_layer(self, layer: 'NN_encoder') -> None:
        """
        Dequantize the weights and activation of a single layer

        Args: 
            Layer: the layer to dequantize
        """
        # Dequantize weights
        quantized_weights = layer.weights
        scale = layer.weight_quant_scale
        zero_point = layer.weight_quant_zero_point
        dequantized_weights = (quantized_weights - zero_point) * scale
        layer.weights = dequantized_weights

        #Dequantize activation
        quantized_activations = layer.activations
        scale = layer.activation_quant_scale
        zero_point = layer.activation_quant_zero_point
        dequantized_activations = (quantized_activations - zero_point) * scale
        layer.activations = dequantized_activations

"""
# example usage

class CustomNeuralNetwork:
    def __init__(self):
        self.layers = [CustomLayer() for _ in range(5)]

class CustomLayer:
    def __init__(self):
        self.weights = torch.randn(10, 10) # subsitute torch for np to create layers in my case. 
        self.activations = torch.randn(10, 10) # same as above subsitute torch for np to create activations

# Create a custom neural network
nn = CustomNeuralNetwork()

# Create a SmoothQuantArgs instance
args = SmoothQuantArgs(num_bits_weight=8, num_bits_activation=8, smooth_quant=True, quant_range=256)

# Create a SmoothQuant instance
smooth_quant = SmoothQuant(args)

# Quantize the neural network
quantized_nn = smooth_quant.quantize(nn)

# Dequantize the neural network
dequantized_nn = smooth_quant.dequantize(quantized_nn)

"""