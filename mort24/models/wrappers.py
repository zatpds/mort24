import logging
from abc import ABC
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import torchmetrics
from sklearn.metrics import log_loss, mean_squared_error, average_precision_score, roc_auc_score

import torch
from torch.nn import MSELoss, CrossEntropyLoss
import torch.nn as nn
from torch import Tensor, FloatTensor
from torch.optim import Optimizer, Adam

import inspect
import gin
import numpy as np
from ignite.exceptions import NotComputableError
from mort24.models.constants import ImputationInit
from mort24.models.custom_metrics import confusion_matrix
from mort24.models.utils import create_optimizer, create_scheduler
from joblib import dump
from pytorch_lightning import LightningModule

from mort24.models.constants import MLMetrics, DLMetrics
from mort24.constants import RunMode

gin.config.external_configurable(nn.functional.nll_loss, module="torch.nn.functional")
gin.config.external_configurable(nn.functional.cross_entropy, module="torch.nn.functional")
gin.config.external_configurable(nn.functional.mse_loss, module="torch.nn.functional")

gin.config.external_configurable(mean_squared_error, module="sklearn.metrics")
gin.config.external_configurable(log_loss, module="sklearn.metrics")
gin.config.external_configurable(average_precision_score, module="sklearn.metrics")
gin.config.external_configurable(roc_auc_score, module="sklearn.metrics")
# gin.config.external_configurable(scorer_wrapper, module="mort24.models.utils")


@gin.configurable("BaseModule")
class BaseModule(LightningModule):
    # DL type models, requires backpropagation
    requires_backprop = False
    # Loss function weight initialization type
    weight = None
    # Metrics to be logged
    metrics = {}
    trained_columns = None
    # Type of run mode
    run_mode = None
    debug = False
    explain_features = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def step_fn(self, batch, step_prefix=""):
        raise NotImplementedError()

    def finalize_step(self, step_prefix=""):
        pass

    def set_metrics(self, *args, **kwargs):
        self.metrics = {}

    def set_trained_columns(self, columns: List[str]):
        self.trained_columns = columns

    def set_weight(self, weight, dataset):
        """Set the weight for the loss function."""

        if isinstance(weight, list):
            weight = FloatTensor(weight).to(self.device)
        elif weight == "balanced":
            weight = FloatTensor(dataset.get_balance()).to(self.device)
        self.loss_weights = weight

    def training_step(self, batch, batch_idx):
        return self.step_fn(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step_fn(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step_fn(batch, "test")

    def on_train_epoch_end(self) -> None:
        self.finalize_step("train")

    def on_validation_epoch_end(self) -> None:
        self.finalize_step("val")

    def on_test_epoch_end(self) -> None:
        self.finalize_step("test")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["class"] = self.__class__
        checkpoint["trained_columns"] = self.trained_columns
        return super().on_save_checkpoint(checkpoint)

    def save_model(self, save_path, file_name, file_extension):
        raise NotImplementedError()

    def check_supported_runmode(self, runmode: RunMode):
        if runmode not in self._supported_run_modes:
            raise ValueError(f"Runmode {runmode} not supported for {self.__class__.__name__}")
        return True


@gin.configurable("DLWrapper")
class DLWrapper(BaseModule, ABC):
    requires_backprop = True
    _metrics_warning_printed = set()
    _supported_run_modes = [RunMode.classification, RunMode.regression, RunMode.imputation]

    def __init__(
        self,
        loss=CrossEntropyLoss(),
        optimizer=Adam,
        run_mode: RunMode = RunMode.classification,
        input_shape=None,
        lr: float = 0.002,
        momentum: float = 0.9,
        lr_scheduler: Optional[str] = None,
        lr_factor: float = 0.99,
        lr_steps: Optional[List[int]] = None,
        epochs: int = 100,
        input_size: Tensor = None,
        initialization_method: str = "normal",
        **kwargs,
    ):
        """General interface for Deep Learning (DL) models."""
        super().__init__()
        self.save_hyperparameters(ignore=["loss", "optimizer"])
        self.loss = loss
        self.optimizer = optimizer
        self.check_supported_runmode(run_mode)
        self.run_mode = run_mode
        self.input_shape = input_shape
        self.lr = lr
        self.momentum = momentum
        self.lr_scheduler = lr_scheduler
        self.lr_factor = lr_factor
        self.lr_steps = lr_steps
        self.epochs = epochs
        self.input_size = input_size
        self.initialization_method = initialization_method
        self.scaler = None

    def on_fit_start(self):
        self.metrics = {
            step_name: {
                metric_name: (metric() if isinstance(metric, type) else metric)
                for metric_name, metric in self.set_metrics().items()
            }
            for step_name in ["train", "val", "test"]
        }
        return super().on_fit_start()

    def on_train_start(self):
        self.metrics = {
            step_name: {
                metric_name: (metric() if isinstance(metric, type) else metric)
                for metric_name, metric in self.set_metrics().items()
            }
            for step_name in ["train", "val", "test"]
        }
        return super().on_train_start()

    def finalize_step(self, step_prefix=""):
        try:
            self.log_dict(
                {
                    f"{step_prefix}/{name}": (
                        np.float32(metric.compute()) if isinstance(metric.compute(), np.float64) else metric.compute()
                    )
                    for name, metric in self.metrics[step_prefix].items()
                    if "_Curve" not in name
                },
                sync_dist=True,
            )
            for metric in self.metrics[step_prefix].values():
                metric.reset()
        except (NotComputableError, ValueError):
            if step_prefix not in self._metrics_warning_printed:
                self._metrics_warning_printed.add(step_prefix)
                logging.warning(f"Metrics for {step_prefix} not computable")
            pass

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""

        if isinstance(self.optimizer, str):
            optimizer = create_optimizer(self.optimizer, self.lr, self.hparams.momentum)
        elif isinstance(self.optimizer, Optimizer):
            # Already set
            optimizer = self.optimizer
        else:
            optimizer = self.optimizer(self.parameters())

        if self.hparams.lr_scheduler is None or self.hparams.lr_scheduler == "":
            return optimizer
        scheduler = create_scheduler(
            self.hparams.lr_scheduler, optimizer, self.hparams.lr_factor, self.hparams.lr_steps, self.hparams.epochs
        )
        optimizers = {"optimizer": optimizer, "lr_scheduler": scheduler}
        logging.info(f"Using: {optimizers}")
        return optimizers

    def on_test_epoch_start(self) -> None:
        self.metrics = {
            step_name: {metric_name: metric() for metric_name, metric in self.set_metrics().items()}
            for step_name in ["train", "val", "test"]
        }
        return super().on_test_epoch_start()

    def save_model(self, save_path, file_name, file_extension=".ckpt"):
        path = save_path / (file_name + file_extension)
        try:
            torch.save(self, path)
            logging.info(f"Model saved to {str(path.resolve())}.")
        except Exception as e:
            logging.error(f"Cannot save model to path {str(path.resolve())}: {e}.")


@gin.configurable("DLPredictionWrapper")
class DLPredictionWrapper(DLWrapper):
    """Interface for Deep Learning models."""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        loss=CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        run_mode: RunMode = RunMode.classification,
        input_shape=None,
        lr: float = 0.002,
        momentum: float = 0.9,
        lr_scheduler: Optional[str] = None,
        lr_factor: float = 0.99,
        lr_steps: Optional[List[int]] = None,
        epochs: int = 100,
        input_size: Tensor = None,
        initialization_method: str = "normal",
        **kwargs,
    ):
        super().__init__(
            loss=loss,
            optimizer=optimizer,
            run_mode=run_mode,
            input_shape=input_shape,
            lr=lr,
            momentum=momentum,
            lr_scheduler=lr_scheduler,
            lr_factor=lr_factor,
            lr_steps=lr_steps,
            epochs=epochs,
            input_size=input_size,
            initialization_method=initialization_method,
            kwargs=kwargs,
        )
        self.output_transform = None
        self.loss_weights = None

    def set_metrics(self, *args):
        """Set the evaluation metrics for the prediction model."""

        def softmax_binary_output_transform(output):
            with torch.no_grad():
                y_pred, y = output
                y_pred = torch.softmax(y_pred, dim=1)
                return y_pred[:, -1], y

        def softmax_multi_output_transform(output):
            with torch.no_grad():
                y_pred, y = output
                y_pred = torch.softmax(y_pred, dim=1)
                return y_pred, y

        # Output transform is not applied for contrib metrics, so we do our own.
        if self.run_mode == RunMode.classification:
            # Binary classification
            if self.logit.out_features == 2:
                self.output_transform = softmax_binary_output_transform
                metrics = DLMetrics.BINARY_CLASSIFICATION
            else:
                # Multiclass classification
                self.output_transform = softmax_multi_output_transform
                metrics = DLMetrics.MULTICLASS_CLASSIFICATION
        # Regression
        elif self.run_mode == RunMode.regression:
            self.output_transform = lambda x: x
            metrics = DLMetrics.REGRESSION
        else:
            raise ValueError(f"Run mode {self.run_mode} not supported.")
        for key, value in metrics.items():
            # Torchmetrics metrics are not moved to the device by default
            if isinstance(value, torchmetrics.Metric):
                value.to(self.device)
        return metrics

    def step_fn(self, element, step_prefix=""):
        """Perform a step in the DL prediction model training loop.

        Args:
            element (object):
            step_prefix (str): Step type, by default: test, train, val.
        """

        if len(element) == 2:
            data, labels = element[0], element[1].to(self.device)
            if isinstance(data, list):
                for i in range(len(data)):
                    data[i] = data[i].float().to(self.device)
            else:
                data = data.float().to(self.device)
            mask = torch.ones_like(labels).bool()

        elif len(element) == 3:
            data, labels, mask = element[0], element[1].to(self.device), element[2].to(self.device)
            if isinstance(data, list):
                for i in range(len(data)):
                    data[i] = data[i].float().to(self.device)
            else:
                data = data.float().to(self.device)
        else:
            raise Exception("Loader should return either (data, label) or (data, label, mask)")
        out = self(data)

        # If aux_loss is present, it is returned as a tuple
        if len(out) == 2 and isinstance(out, tuple):
            out, aux_loss = out
        else:
            aux_loss = 0
        # Get prediction and target
        prediction = torch.masked_select(out, mask.unsqueeze(-1)).reshape(-1, out.shape[-1]).to(self.device)
        target = torch.masked_select(labels, mask).to(self.device)

        if prediction.shape[-1] > 1 and self.run_mode == RunMode.classification:
            # Classification task
            loss = self.loss(prediction, target.long(), weight=self.loss_weights.to(self.device)) + aux_loss
            # Returns torch.long because negative log likelihood loss
        elif self.run_mode == RunMode.regression:
            # Regression task
            loss = self.loss(prediction[:, 0], target.float()) + aux_loss
        else:
            raise ValueError(f"Run mode {self.run_mode} not yet supported. Please implement it.")
        transformed_output = self.output_transform((prediction, target))

        for key, value in self.metrics[step_prefix].items():
            if isinstance(value, torchmetrics.Metric):
                if key == "Binary_Fairness":
                    feature_names = key.feature_helper(self.trainer)
                    value.update(transformed_output[0], transformed_output[1], data, feature_names)
                else:
                    value.update(transformed_output[0], transformed_output[1])
            else:
                value.update(transformed_output)
        self.log(f"{step_prefix}/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss


@gin.configurable("MLWrapper")
class MLWrapper(BaseModule, ABC):
    """Interface for prediction with traditional Scikit-learn-like Machine Learning models."""

    requires_backprop = False
    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(self, *args, run_mode=RunMode.classification, loss=log_loss, patience=10, mps=False, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.scaler = None
        self.check_supported_runmode(run_mode)
        self.run_mode = run_mode
        self.loss = loss
        self.patience = patience
        self.mps = mps
        self.loss_weight = None

    def set_metrics(self, labels):
        if self.run_mode == RunMode.classification:
            # Binary classification
            if len(np.unique(labels)) == 2:
                # if isinstance(self.model, lightgbm.basic.Booster):
                self.output_transform = lambda x: x[:, 1]
                self.label_transform = lambda x: x

                self.metrics = MLMetrics.BINARY_CLASSIFICATION
            # Multiclass classification
            else:
                # Todo: verify multiclass classification
                self.output_transform = lambda x: np.argmax(x, axis=-1)
                self.label_transform = lambda x: x
                self.metrics = MLMetrics.MULTICLASS_CLASSIFICATION

        # Regression
        else:
            if self.scaler is not None:  # We invert transform the labels and predictions if they were scaled.
                self.output_transform = lambda x: self.scaler.inverse_transform(x.reshape(-1, 1))
                self.label_transform = lambda x: self.scaler.inverse_transform(x.reshape(-1, 1))
            else:
                self.output_transform = lambda x: x
                self.label_transform = lambda x: x
            self.metrics = MLMetrics.REGRESSION

    def fit(self, train_dataset, val_dataset):
        """Fit the model to the training data."""
        train_rep, train_label, row_indicators = train_dataset.get_data_and_labels()
        val_rep, val_label, row_indicators = val_dataset.get_data_and_labels()

        self.set_metrics(train_label)

        if "class_weight" in self.model.get_params().keys():  # Set class weights
            self.model.set_params(class_weight=self.weight)

        val_loss = self.fit_model(train_rep, train_label, val_rep, val_label)

        train_pred = self.predict(train_rep)

        logging.debug(f"Model:{self.model}")

        self.log("train/loss", self.loss(train_label, train_pred), sync_dist=True)
        logging.debug(f"Train loss: {self.loss(train_label, train_pred)}")
        self.log("val/loss", val_loss, sync_dist=True)
        logging.debug(f"Val loss: {val_loss}")
        self.log_metrics(train_label, train_pred, "train")

    def fit_model(self, train_data, train_labels, val_data, val_labels):
        """Fit the model to the training data (default SKlearn syntax)"""
        self.model.fit(train_data, train_labels)
        val_loss = 0.0
        return val_loss

    def validation_step(self, val_dataset, _):
        val_rep, val_label, row_indicators = val_dataset.get_data_and_labels()
        val_rep, val_label = torch.from_numpy(val_rep).to(self.device), torch.from_numpy(val_label).to(self.device)
        self.set_metrics(val_label)

        val_pred = self.predict(val_rep)

        self.log_metrics("val/loss", self.loss(val_label, val_pred), sync_dist=True)
        logging.info(f"Val loss: {self.loss(val_label, val_pred)}")
        self.log_metrics(val_label, val_pred, "val")

    def test_step(self, dataset, _):
        test_rep, test_label, pred_indicators = dataset
        test_rep, test_label, pred_indicators = (
            test_rep.squeeze().cpu().numpy(),
            test_label.squeeze().cpu().numpy(),
            pred_indicators.squeeze().cpu().numpy(),
        )
        self.set_metrics(test_label)
        test_pred = self.predict(test_rep)
        if self.debug:
            self._save_model_outputs(pred_indicators, test_pred, test_label)
        if self.explain_features:
            self.explain_model(test_rep, test_label)
        if self.mps:
            self.log("test/loss", np.float32(self.loss(test_label, test_pred)), sync_dist=True)
            self.log_metrics(np.float32(test_label), np.float32(test_pred), "test")
        else:
            self.log("test/loss", self.loss(test_label, test_pred), sync_dist=True)
            self.log_metrics(test_label, test_pred, "test")
        logging.debug(f"Test loss: {self.loss(test_label, test_pred)}")

    def predict(self, features):
        if self.run_mode == RunMode.regression:
            return self.model.predict(features)
        else:  # Classification: return probabilities
            return self.model.predict_proba(features)

    def log_metrics(self, label, pred, metric_type):
        """Log metrics to the PL logs."""
        if "Confusion_Matrix" in self.metrics:
            self.log_dict(confusion_matrix(self.label_transform(label), self.output_transform(pred)), sync_dist=True)
        self.log_dict(
            {
                f"{metric_type}/{name}": (metric(self.label_transform(label), self.output_transform(pred)))
                # For every metric
                for name, metric in self.metrics.items()
                # Filter out metrics that return a tuple (e.g. precision_recall_curve)
                if not isinstance(metric(self.label_transform(label), self.output_transform(pred)), tuple)
                and name != "Confusion_Matrix"
            },
            sync_dist=True,
        )

    def _explain_model(self, test_rep, test_label):
        if self.explainer is not None:
            self.test_shap_values = self.explainer(test_rep)
        else:
            logging.warning("No explainer or explain_features values set.")

    def _save_model_outputs(self, pred_indicators, test_pred, test_label):
        if len(pred_indicators.shape) > 1 and len(test_pred.shape) > 1 and pred_indicators.shape[1] == test_pred.shape[1]:
            pred_indicators = np.hstack((pred_indicators, test_label.reshape(-1, 1)))
            pred_indicators = np.hstack((pred_indicators, test_pred))
            # Save as: id, time (hours), ground truth, prediction 0, prediction 1
            np.savetxt(Path(self.logger.save_dir) / "pred_indicators.csv", pred_indicators, delimiter=",")
            logging.debug(f"Saved row indicators to {Path(self.logger.save_dir) / 'row_indicators.csv'}")
        else:
            logging.warning("Could not save row indicators.")

    def configure_optimizers(self):
        return None

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        del state["label_transform"]
        del state["output_transform"]
        return state

    def save_model(self, save_path, file_name, file_extension=".joblib"):
        path = save_path / (file_name + file_extension)
        try:
            dump(self.model, path)
            logging.info(f"Model saved to {str(path.resolve())}.")
        except Exception as e:
            logging.error(f"Cannot save model to path {str(path.resolve())}: {e}.")

    def set_model_args(self, model, *args, **kwargs):
        """Set hyperparameters of the model if they are supported by the model."""
        signature = inspect.signature(model.__init__).parameters
        possible_hps = list(signature.keys())
        # Get passed keyword arguments
        arguments = locals()["kwargs"]
        # Get valid hyperparameters
        logging.debug(f"Possible hps: {possible_hps}")
        hyperparams = {key: value for key, value in arguments.items() if key in possible_hps}
        logging.debug(f"Creating model with: {hyperparams}.")
        return model(**hyperparams)


@gin.configurable("ImputationWrapper")
class ImputationWrapper(DLWrapper):
    """Interface for imputation models."""

    requires_backprop = True
    _supported_run_modes = [RunMode.imputation]

    def __init__(
        self,
        loss: nn.modules.loss._Loss = MSELoss(),
        optimizer: Union[str, Optimizer] = "adam",
        run_mode: RunMode = RunMode.imputation,
        lr: float = 0.002,
        momentum: float = 0.9,
        lr_scheduler: Optional[str] = None,
        lr_factor: float = 0.99,
        lr_steps: Optional[List[int]] = None,
        input_size: Tensor = None,
        initialization_method: ImputationInit = ImputationInit.NORMAL,
        epochs=100,
        **kwargs: str,
    ) -> None:
        super().__init__(
            loss=loss,
            optimizer=optimizer,
            run_mode=run_mode,
            lr=lr,
            momentum=momentum,
            lr_scheduler=lr_scheduler,
            lr_factor=lr_factor,
            lr_steps=lr_steps,
            epochs=epochs,
            input_size=input_size,
            initialization_method=initialization_method,
            kwargs=kwargs,
        )
        self.check_supported_runmode(run_mode)
        self.run_mode = run_mode
        self.save_hyperparameters(ignore=["loss", "optimizer"])
        self.loss = loss
        self.optimizer = optimizer

    def set_metrics(self):
        return DLMetrics.IMPUTATION

    def init_weights(self, init_type="normal", gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
                if init_type == ImputationInit.NORMAL:
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == ImputationInit.XAVIER:
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == ImputationInit.KAIMING:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
                elif init_type == ImputationInit.ORTHOGONAL:
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError(f"Initialization method {init_type} is not implemented")
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find("BatchNorm2d") != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def on_fit_start(self) -> None:
        self.init_weights(self.hparams.initialization_method)
        for metrics in self.metrics.values():
            for metric in metrics.values():
                metric.reset()
        return super().on_fit_start()

    def step_fn(self, batch, step_prefix=""):
        amputated, amputation_mask, target, target_missingness = batch
        imputated = self(amputated, amputation_mask)
        amputated[amputation_mask > 0] = imputated[amputation_mask > 0]
        amputated[target_missingness > 0] = target[target_missingness > 0]

        loss = self.loss(amputated, target)
        self.log(f"{step_prefix}/loss", loss.item(), prog_bar=True)

        for metric in self.metrics[step_prefix].values():
            metric.update(
                (torch.flatten(amputated.detach(), start_dim=1).clone(), torch.flatten(target.detach(), start_dim=1).clone())
            )
        return loss

    def fit(self, train_dataset, val_dataset):
        raise NotImplementedError()

    def predict_step(self, data, amputation_mask=None):
        return self(data, amputation_mask)

    def predict(self, data):
        self.eval()
        data = data.to(self.device)
        data_missingness = torch.isnan(data).to(torch.float32)
        prediction = self.predict_step(data, data_missingness)
        data[data_missingness.bool()] = prediction[data_missingness.bool()]
        return data


@gin.configurable("DomainAdaptiveWrapper")
class DomainAdaptiveWrapper(DLPredictionWrapper):
    """Wrapper for domain adaptation models using adversarial training.

    This wrapper extends DLPredictionWrapper to support domain adversarial training
    for multi-domain datasets. It computes both task loss and domain adversarial loss.

    Args:
        domain_loss_weight: Weight for domain adversarial loss (default: 1.0)
        grl_schedule: Schedule for gradient reversal layer alpha:
            - "constant": Keep alpha constant
            - "linear": Linearly increase alpha
            - "exponential": Exponentially increase following DANN paper
        grl_low: Initial alpha value (default: 0.0)
        grl_high: Maximum alpha value (default: 1.0)
        track_domain_metrics: Whether to track domain-specific metrics (default: True)
        separate_domain_evaluation: Evaluate source and target domains separately (default: True)
        **kwargs: Additional arguments passed to DLPredictionWrapper
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        domain_loss_weight: float = 1.0,
        grl_schedule: str = "linear",
        grl_low: float = 0.0,
        grl_high: float = 1.0,
        track_domain_metrics: bool = True,
        separate_domain_evaluation: bool = True,
        **kwargs,
    ):
        """Initialize the domain adaptive wrapper."""
        super().__init__(**kwargs)

        self.domain_loss_weight = domain_loss_weight
        self.grl_schedule = grl_schedule
        self.grl_low = grl_low
        self.grl_high = grl_high
        self.track_domain_metrics = track_domain_metrics
        self.separate_domain_evaluation = separate_domain_evaluation

        # Domain loss function (cross-entropy for domain classification)
        self.domain_loss_fn = torch.nn.CrossEntropyLoss()

        logging.info(
            f"Initialized DomainAdaptiveWrapper with domain_loss_weight={domain_loss_weight}, "
            f"grl_schedule={grl_schedule}"
        )

    def set_metrics(self, *args):
        """Set evaluation metrics including domain-specific metrics."""
        # Get base metrics from parent class
        metrics = super().set_metrics(*args)

        if self.track_domain_metrics:
            # Add domain discriminator accuracy
            metrics["Domain_Accuracy"] = torchmetrics.Accuracy(
                task="binary" if self.run_mode == RunMode.classification else "multiclass",
                num_classes=2,
            ).to(self.device)

        return metrics

    def on_train_epoch_start(self) -> None:
        """Update GRL alpha at the start of each epoch."""
        if hasattr(self, "grl_schedule") and self.grl_schedule != "constant":
            from mort24.models.domain_adaptation.gradient_reversal import get_grl_alpha

            alpha = get_grl_alpha(
                epoch=self.current_epoch,
                max_epoch=self.hparams.epochs,
                schedule=self.grl_schedule,
                low=self.grl_low,
                high=self.grl_high,
            )

            # Update model's GRL if it has one
            if hasattr(self, "set_grl_alpha"):
                self.set_grl_alpha(alpha)

            self.log("train/grl_alpha", alpha, on_step=False, on_epoch=True, sync_dist=True)
            logging.debug(f"Updated GRL alpha to {alpha:.4f} for epoch {self.current_epoch}")

        return super().on_train_epoch_start()

    def step_fn(self, element, step_prefix=""):
        """Perform a training/validation/test step with domain adaptation.

        Args:
            element: Batch data, expected to be (data, labels, mask, domain_ids)
            step_prefix: Step type (train/val/test)

        Returns:
            Total loss (task_loss + domain_loss_weight * domain_loss)
        """
        # Unpack batch data
        if len(element) == 4:
            data, labels, mask, domain_ids = element
            data = data.float().to(self.device)
            labels = labels.to(self.device)
            mask = mask.to(self.device)
            domain_ids = domain_ids.to(self.device)
        elif len(element) == 3:
            # Fallback for non-domain-adaptation datasets
            data, labels, mask = element
            data = data.float().to(self.device)
            labels = labels.to(self.device)
            mask = mask.to(self.device)
            domain_ids = None
        else:
            raise ValueError(f"Expected element to have 3 or 4 components, got {len(element)}")

        # Forward pass through model
        # Model should return (task_output, domain_output) if it's a DANN model
        output = self(data)

        if isinstance(output, tuple) and len(output) == 2:
            task_output, domain_output = output
            has_domain_output = True
        else:
            task_output = output
            domain_output = None
            has_domain_output = False

        # Compute task loss (same as parent class)
        prediction = torch.masked_select(task_output, mask.unsqueeze(-1)).reshape(-1, task_output.shape[-1])
        target = torch.masked_select(labels, mask)

        if prediction.shape[-1] > 1 and self.run_mode == RunMode.classification:
            task_loss = self.loss(prediction, target.long(), weight=self.loss_weights.to(self.device))
        elif self.run_mode == RunMode.regression:
            task_loss = self.loss(prediction[:, 0], target.float())
        else:
            raise ValueError(f"Run mode {self.run_mode} not supported")

        # Compute domain loss if we have domain outputs
        if has_domain_output and domain_ids is not None and domain_output is not None:
            # Flatten domain predictions and targets
            domain_pred = torch.masked_select(domain_output, mask.unsqueeze(-1)).reshape(-1, domain_output.shape[-1])
            domain_target = domain_ids.unsqueeze(1).expand(-1, mask.shape[1])
            domain_target = torch.masked_select(domain_target, mask).long()

            domain_loss = self.domain_loss_fn(domain_pred, domain_target)

            # Total loss
            total_loss = task_loss + self.domain_loss_weight * domain_loss

            # Log domain loss
            self.log(f"{step_prefix}/domain_loss", domain_loss, on_step=False, on_epoch=True, sync_dist=True)

            # Update domain accuracy metric if tracking
            if self.track_domain_metrics and "Domain_Accuracy" in self.metrics[step_prefix]:
                domain_pred_labels = torch.argmax(domain_pred, dim=1)
                self.metrics[step_prefix]["Domain_Accuracy"].update(domain_pred_labels, domain_target)
        else:
            total_loss = task_loss

        # Update task metrics (same as parent class)
        transformed_output = self.output_transform((prediction, target))
        for key, value in self.metrics[step_prefix].items():
            if key == "Domain_Accuracy":
                continue  # Already updated above
            if isinstance(value, torchmetrics.Metric):
                value.update(transformed_output[0], transformed_output[1])
            else:
                value.update(transformed_output)

        # Log losses
        self.log(f"{step_prefix}/task_loss", task_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{step_prefix}/loss", total_loss, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss
