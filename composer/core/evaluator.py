# Copyright 2021 MosaicML. All Rights Reserved.

"""A wrapper for a dataloader to include metrics that apply to a specific dataset."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, List, Optional, Union

from torchmetrics import Metric, MetricCollection

from composer.core.data_spec import DataSpec as DataSpec
from composer.core import Callback

if TYPE_CHECKING:
    from composer.core.types import DataLoader

__all__ = ["Evaluator"]

log = logging.getLogger(__name__)


class Evaluator(Callback):
    """A wrapper for a dataloader to include metrics that apply to a specific dataset.

    For example, :class:`~.nlp_metrics.CrossEntropyLoss` metric for NLP models.

    .. doctest::

       >>> from torchmetrics.classification.accuracy import Accuracy
       >>> eval_evaluator = Evaluator(label="myEvaluator", dataloader=eval_dataloader, metrics=Accuracy())
       >>> trainer = Trainer(
       ...     model=model,
       ...     train_dataloader=train_dataloader,
       ...     eval_dataloader=eval_evaluator,
       ...     optimizers=optimizer,
       ...     max_duration="1ep",
       ... )

    .. testcleanup::

        trainer.engine.close()


    Args:
        label (str): Name of the Evaluator
        dataloader (Union[DataSpec, DataLoader]): DataLoader/DataSpec for evaluation data
        metrics (Metric | MetricCollection): :class:`torchmetrics.Metric` to log. ``metrics`` will be deep-copied to ensure
            that each evaluator updates only its ``metrics``.
        summary (Optional[str]): Optionally, a summary statistic for each metric that is logged to WandB.
    """

    def __init__(self,
                 *,
                 label: str,
                 dataloader: Union[DataSpec, DataLoader],
                 metrics: Union[Metric, MetricCollection],
                 summary: Optional[List[str]] = None):
        self.label = label
        if isinstance(dataloader, DataSpec):
            self.dataloader = dataloader
        else:
            self.dataloader = DataSpec(dataloader)

        # Forcing metrics to be a MetricCollection simplifies logging results
        metrics = copy.deepcopy(metrics)
        if isinstance(metrics, Metric):
            self.metrics = MetricCollection([metrics])
        else:
            self.metrics = metrics
        self.summary = summary

    def init(self, state: State, logger: Logger) -> None:
        # add metric summary to WandB metrics
        if self.summary is not None:
            try:
                import wandb
            except ImportError:
                log.warning(f"WandB not installed so {label} summary '{self.summary}' will not be logged.")

            if wandb.run is None:
                raise ValueError("wandb must be initialized before serialization.")

            if len(self.metrics.keys()) != len(self.summary):
                raise ValueError("There must be a summary statistic for every metric.")

            for metric_index, metric_name in enumerate(self.metrics.keys()):
                wandb.define_metric(name=f'metrics/{self.label}/{metric_name}', summary=self.summary[metric_index])
