from torch import nn
import torch


class LinLayer(nn.Module):
    def __init__(
        self,
        f_in: int,
        f_out: int,
        activation: type[nn.Module],
        use_batchnorm: bool,
        dropout_prob: float,
    ):
        super().__init__()
        self.fc: nn.Module = nn.Linear(f_in, f_out)
        self.act: nn.Module = activation()
        self.dp: nn.Module = nn.Dropout(dropout_prob)
        if use_batchnorm:
            self.bn1: nn.Module = nn.BatchNorm1d(f_out)

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        if hasattr(self, "bn1"):
            x = self.bn1(x)
        x = self.dp(self.act(x))
        return x


# layer_dims = [270, 100, 10]
# layer_activations = [nn.ReLU, nn.ReLU, nn.ReLU]
# layer_bns = [False, False, False]


class PerceptronNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layer_dims: list[int],
        layer_activations: list[type[nn.Module]],
        layer_bns: list[bool],
        dropout_prob: float = 0.0,
        **kwargs
    ):
        """Multilayer perceptron neural network model.

        Example:
        * input_dim = 28*28
        * layer_dims = [256, 128, 10]
        * layer_activations = [nn.ReLU, nn.ReLU, nn.ReLU]
        * layer_bns = [True, True, True]
        * dropout_prob = 0.2

        Will produce a 3-layer MLP for MNIST dataset. First layer would be:
            nn.Linear(768, 256) -> nn.BatchNorm1d(256) -> nn.ReLU() ->
            -> nn.Dropout(0.2) and so on.


        Args:
            input_dim (int): size of input tensor.
            layer_dims (list[int]): list of feature map dimensions for each
                layer.
            layer_activations (list[type[nn.Module]]): list of activation
                function for each layer.
            layer_bns (list[bool]): list of bools whether to use batch
                normalization for each layer.
            dropout_prob (float, Optional): probability of using dropout on
                all layers. Should be value between 0 and 1. Defaults to 0.0.
        """
        super().__init__()
        layer_dims: list[int] = [input_dim] + layer_dims
        _lst_params: list[dict[str, int | type[nn.Module] | bool | float]]
        _lst_params = [
            dict(
                f_in=arg[0],
                f_out=arg[1],
                activation=arg[2],
                use_batchnorm=arg[3],
                dropout_prob=dropout_prob,
            )
            for arg in zip(
                layer_dims[:-1], layer_dims[1:], layer_activations, layer_bns
            )
        ]
        layers: list[LinLayer] = [LinLayer(**kwargs) for kwargs in _lst_params]
        self.mlp: nn.Module = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        out: torch.Tensor = self.mlp(x)
        return out
