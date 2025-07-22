from backbones.baseline import SimpleConcatBaseline
from backbones.baseline import SimpleConcatBaseline
from backbones.our.model import MMNet
from config import MAGNIFICATIONS

def get_all_models(mags=MAGNIFICATIONS):
    """Get all available models for experiments"""
    models = {
        'MMNet': MMNet(),
        'SimpleConcat': SimpleConcatBaseline(),
        'Single400X': SimpleConcatBaseline()
    }
    return models

