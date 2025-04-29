from torchmetrics.functional import auroc
from .torchmetrics_validator import TorchmetricsValidator


class AUCValidator(TorchmetricsValidator):
    """
    Retorna a AUC (Area Under the ROC Curve) utilizando a função
    [auroc do torchmetrics](https://torchmetrics.readthedocs.io/en/latest/references/functional.html#auroc-func).

    Os splits necessários são, por padrão, ["src_val"]. Isso pode ser modificado
    através do parâmetro key_map do BaseValidator.
    """

    def __init__(self, layer="preds", torchmetric_kwargs=None, **kwargs):
        # Define um dicionário padrão para a tarefa binária
        default_kwargs = {"num_classes": 2}
        # Atualiza os valores padrão caso o usuário passe parâmetros adicionais
        if torchmetric_kwargs is not None:
            default_kwargs.update(torchmetric_kwargs)
        super().__init__(layer=layer, torchmetric_kwargs=default_kwargs, **kwargs)

    @property
    def accuracy_fn(self):
        return auroc
