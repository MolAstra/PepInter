from pepinter.esmc import ESMCConfig


class PepInterConfig(ESMCConfig):
    model_type = "pepinter"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
