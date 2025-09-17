class ModelUtils:
    @staticmethod
    def get_model_type(config):
        return config.model_type if hasattr(config, 'model_type') else 'unknown'

    @staticmethod
    def validate_model_path(model_path):
        import os
        return os.path.isdir(model_path)