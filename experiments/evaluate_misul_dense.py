"""Use the unchanged final evaluation with the dense architecture constructor."""
from evaluate_misul import main
import evaluate_misul
from transformermodel.misul_dense import DenseModel

evaluate_misul.MisulModel = DenseModel
main()
