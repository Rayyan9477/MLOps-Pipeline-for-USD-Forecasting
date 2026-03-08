"""Models module for training and inference."""


def __getattr__(name):
    if name == "ModelTrainer":
        from src.models.trainer import ModelTrainer
        return ModelTrainer
    if name == "ProductionModelTrainer":
        from src.models.production_trainer import ProductionModelTrainer
        return ProductionModelTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ModelTrainer", "ProductionModelTrainer"]
