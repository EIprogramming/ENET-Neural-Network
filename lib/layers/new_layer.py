class NewLayer:
    LAYER_TYPES = [
        "Input",
        "InputND",
        "Dense",
        "Convolutional",
    ]

    def __init__(self, shape: tuple | int, layer_type: str, **kwargs):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape: tuple = shape
        self.layer_type = layer_type
        self.kernel_params: tuple[int, int, int] | None = kwargs["kernel_params"] if "kernel_params" in kwargs else None
        self.input_shape: tuple | None = kwargs["input_shape"] if "input_shape" in kwargs else None
        self.validate_shape()
        self.validate_type()
    
    def validate_type(self):
        if self.layer_type not in NewLayer.LAYER_TYPES:
            raise ValueError(f"Invalid layer type, got {self.layer_type}, expected: {self.LAYER_TYPES}")
    
    def validate_shape(self):
        if self.layer_type == "Dense" or self.layer_type == "Inputer":
            if len(self.shape) != 1:
                raise TypeError(f"Invalid shape, expected 1 dimension for layer of type {self.layer_type}, but got {self.shape}")
