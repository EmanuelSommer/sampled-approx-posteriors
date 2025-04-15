"""Top level configuration class."""

import importlib
import inspect
import json
import logging
import logging.config
import logging.handlers
import sys
import time
from dataclasses import field
from pathlib import Path
from typing import Any, Optional

import flax.linen as nn
from dataserious.base import Annotation, BaseConfig, yaml_dump

from src.sai.config.data import DataConfig
from src.sai.config.models.base import ModelConfig
from src.sai.config.training import TrainingConfig
from src.sai.exceptions import MissingConfigError, ModelNotFoundError
from src.sai.inference.evaluation import EvaluationName

models_module = importlib.import_module("src.sai.architectures")


class Config(BaseConfig):
    """Top level configuration class.

    Note:
        - This is the top level configuration class.
        - It contains 3 main Branches of configuration:
            - Data: Configuration for the Dataset `./data.py`.
            - Model: Configuration for the ML Model `./models/*.py`.
            - Training: Configuration for Training / Sampling Proccess `./training.py`.
    """

    saving_dir: str = field(
        metadata={"description": "Directory to save checkpoints and logs."},
    )
    experiment_name: str = field(metadata={"description": "Name of the experiment."})
    data: DataConfig = field(metadata={"description": "Configuration for the Dataset."})
    model: ModelConfig = field(metadata={"description": "Configuration for the Model."})
    training: TrainingConfig = field(
        default_factory=TrainingConfig,
        metadata={"description": "Configuration for Training."},
    )
    rng: int = field(
        default=42, metadata={"description": "RNG Seed.", "searchable": True}
    )
    logging: bool = field(
        default=True, metadata={"description": "Enable logging to file and stdout."}
    )
    logging_level: str = field(
        default="INFO", metadata={"description": "Set logging level."}
    )
    evaluations: list[EvaluationName] = field(
        default_factory=list,
        metadata={"description": "evaluations to perform at the end of the pipeline."},
    )
    evaluation_args: dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "Args for the evals like kappa or nominal_coverage."},
    )

    def __post_init__(self):
        """Ensure that the saving directory exists."""
        super().__post_init__()
        self._modify_field(saving_dir=Path(self.saving_dir).as_posix())

    def __hash__(self):
        """Return the hash of the configuration.

        Note:
            - This might not be the most stable but it is good enough for
            the purpose of keeping unique configurations, when performing
            Grid or Random Search.
        """
        return hash(self.__str__())

    @classmethod
    def to_template(cls, model: str | nn.Module):
        """Return template config for the chosen model."""
        if isinstance(model, str):
            if model not in models_module.__all__:
                raise ModelNotFoundError(
                    f"In order to access model with a string name, it must be "
                    f"contained in {models_module.__name__} module. Got {model} but "
                    f"expected one of: {models_module.__all__}"
                    f"For custom models, pass the model class directly."
                )
            model = getattr(models_module, model)
        cfg_cls = cls._get_model_config_cls(model)
        schema = cls.to_schema()
        schema["model"] = cfg_cls.to_schema()
        return schema

    @classmethod
    def template_to_yaml(cls, path: Path | str, model: str | nn.Module):
        """Write the template to a YAML file.

        Note:
            - See the classmethod `Config.to_template()` for more information.§
        """
        schema = cls.to_template(model)
        yaml_dump(schema, path=path)

    @classmethod
    def template_to_json(cls, path: Path | str, model: str | nn.Module):
        """Write the templat to JSON.

        Note:
            - See the classmethod `Config.to_template()` for more information.
        """
        schema = cls.to_template(model)
        json.dump(schema, open(path, "w"), indent=3)

    @staticmethod
    def list_avaliable_models():
        """List all available pre-defined models."""
        return models_module.__all__

    def get_flax_model(self, **kwargs) -> nn.Module:
        """Return the associated Flax model instance.

        Args:
            **kawrgs:
                Additional keyword Fields other than config
                to pass to the nn.Module constructor.
        """
        return getattr(models_module, self.model.model)(config=self.model, **kwargs)

    def setup_dir(self, folder: Optional[Path] = None):
        """Setups the saving directory and logging for the experiment.

        Notes:
            - If the experiment_name already exists in given saving directory,
                it will append the current timestamp to the name to make it unique
                and avoid overwriting results.
            - Method Recursively creates the saving directory if it does not exist.
            - If logging is enabled, it will setup logging to file and stdout.
        """
        if folder is None:
            if self.experiment_dir.exists():
                name = f"{self.experiment_name}_{time.strftime('%Y%m%d-%H%M%S')}"
                self._modify_field(experiment_name=name)
            self.experiment_dir.mkdir(parents=True)
            self.to_yaml(self.experiment_dir / "config.yaml")
        else:
            if (Path(self.saving_dir) / folder).exists():
                self._modify_field(experiment_name=str(folder))
            else:
                raise FileNotFoundError(f"The folder '{folder}' doesn't exist.")
        self._setup_logging()

    @property
    def n_chains(self):
        """Return the number of chains."""
        return self.training.sampler.n_chains

    @property
    def experiment_dir(self):
        """Return the saving path."""
        return Path(self.saving_dir) / self.experiment_name

    def _setup_logging(self):
        """Setups logging to file and stdout."""
        handlers = []
        handlers.append(logging.FileHandler(self.experiment_dir / "training.log"))
        if self.logging:
            handlers.append(logging.StreamHandler(sys.stdout))
        logging.basicConfig(
            handlers=handlers,
            level=getattr(logging, self.logging_level),
            force=True,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logging.getLogger(__name__).info("Logging successfully setup.")

    @classmethod
    def handle_field_from(cls, annot: Annotation, value) -> Any:
        """Extend the handle_field method to handle ModelConfig fields.

        Notes:
            - We extend the default `handle_field()` method to handle `ModelConfig` fields.
            - When we want to parse the `ModelConfig` field, we need to somehow
                connect the string name to the actual model configuration class.
            - For this Purpose each `ModelConfig` has a `model` field which is a string
                exactly matching the class name of the model it intends to configure.
            - We use this string to get the corresponding `ModelConfig` subclass
                and parse the dictionary accordingly.
        """
        if inspect.isclass(annot) and issubclass(annot, ModelConfig):
            if isinstance(value, dict):
                mapping = annot.get_name_mapping()
                model = mapping.get(value.get("model"))
                if not model:
                    raise ModelNotFoundError(
                        f'Could not find model {value.get("model")}. '
                        f'Avaliable models: {[*mapping.keys()]}'
                    )
                return model.from_dict(value)
        return super().handle_field_from(annot, value)

    def _get_model_config_cls(model: nn.Module) -> type[ModelConfig]:
        """Return the corresponding ModelConfig class for the given model."""
        if not inspect.isclass(model) and not issubclass(model, nn.Module):
            raise ValueError(f"Expected a flax.nn.Module, got {type(model)} instead.")
        if "config" not in model.__dataclass_fields__:
            raise MissingConfigError(
                f"The model {model.__name__} does not have a `config` field "
                "of corresponding pre-defined `ModelConfig` subclass."
            )
        return model.__dataclass_fields__["config"].type
