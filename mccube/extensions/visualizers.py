import abc
import os
import time
from queue import Queue
from threading import Thread

import jax.numpy as jnp
from jaxtyping import Float, Array
from mccube.cubature import MCCubatureState
from mccube.metrics import cubature_target_error
from mccube.utils import no_operation
from tensorboardX import SummaryWriter


class AbstractMCCubatureVisualizer:
    """Defines a Markov chain cubature visualization step."""

    @abc.abstractmethod
    def __call__(self, state: MCCubatureState):
        ...


class NonblockingVisualizer(AbstractMCCubatureVisualizer):
    """Defines a nonblocking MCCubature visualizer.

    Attributes:
        logging_interval: interval at which to perform visualization/logging.
        target_mean: mean against which to compare the MCCubature mean.
        target_cov: covariance against which to compare the MCCubature covariance.
    """

    def __init__(
        self,
        logging_interval: int = 1,
        target_mean: Float[Array, " d"] = None,
        target_cov: Float[Array, "d d"] = None,
    ):
        """Initialise a cubature visualizer.

        Args:
            logging_interval: interval at which to perform visualization/logging.
            target_mean: mean against which to compare the MCCubature mean.
            target_cov: covariance against which to compare the MCCubature covariance.
        """
        self.logging_interval = logging_interval
        self.target_mean = target_mean
        self.target_cov = target_cov
        self._queue = Queue()

    def __call__(self, state: MCCubatureState, *args, **kwargs):
        """Visulization callback to be passed to the MCCubatureStep."""
        epoch = state.time
        particles = state.particles
        current_time = time.time()
        if not hasattr(self, "start_time_"):
            self.start_time_ = current_time
        if epoch % self.logging_interval:
            return
        self.elapsed_time_ = current_time - self.start_time_
        self._queue.put((particles, epoch, current_time))

    def __enter__(self):
        self.start_visualization()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop_visualization()

    def _inprocess_visualization(self, visualization_function=no_operation):
        """Visualization operations to perform on queued cubature steps."""
        while True:
            particles, epoch, walltime = self._queue.get()
            self.epoch_ = epoch
            visualization_function(particles, epoch, walltime)
            if self.target_mean and self.target_cov:
                self.error_ = cubature_target_error(
                    particles, self.target_mean, self.target_cov
                )
            self._queue.task_done()

    def _postprocess_visualization(self):
        """Visualization to perform after the final cubature epoch has terminated."""
        string = f"""
        Visualization Complete
        Experiment Time: {self.elapsed_time_}
        """
        if self.target_mean and self.target_cov:
            normalized_mean_error = self.error_[0].l2 / jnp.linalg.norm(
                self.target_mean
            )
            normalized_variance_error = self.error_[1].l2 / jnp.linalg.norm(
                jnp.diag(self.target_cov)
            )
            string += f"""
            Error Mean: {self.error_[0].l2} ({100 * 1 - normalized_mean_error} %)
            Error Var: {self.error_[1].l2} ({100 * 1 - normalized_variance_error} %)
            """
        print(string)

    def start_visualization(self):
        """Start the visualization thread."""
        self._visualizer_thread = Thread(
            target=self._inprocess_visualization, daemon=True
        )
        self._visualizer_thread.start()
        # May need a finalizer for the case where stop_visualizer isnt called?

    def stop_visualization(self):
        """Finish processing items in the queue and stop the visualization thread."""
        if self._queue.not_empty:
            print("Visualiser is finishing up...")
            self._queue.join()
        self._postprocess_visualization()


class TensorboardVisualizer(NonblockingVisualizer):
    """Tensorboard compatible cubature visualizer.

    Attributes:
        logdir: directory in which to store the tensorboard logs.
        logging_interval: interval at which to perform visualization/logging.
        hparams: list of cubature hyper-parameters.
        plot_figures: indicates if to plot intermediate cubature step particles.
    """

    def __init__(
        self,
        logdir=None,
        logging_interval=1,
        hparams=None,
        experiment_name=str(time.time()),
        **kwargs,
    ):
        """Initialises a Tensorboard cubature visualizer.

        Args:
            logdir: directory in which to store the tensorboard logs.
            logging_interval: interval at which to perform visualization/logging.
            hparams: list of cubature hyper-parameters.
            experiment_name: name of the experimental run.
            plot_figures: indicates if to plot intermediate cubature step particles.
        """
        self.logdir = os.path.join(logdir or os.getcwd(), experiment_name)
        self.logging_interval = logging_interval
        self.hparams = hparams
        super().__init__(logging_interval=logging_interval, **kwargs)

    def _inprocess_visualization(self):
        def visualisation_function(particles, epoch, walltime):
            if self.target_mean and self.target_cov:
                self.error_ = cubature_target_error(
                    particles, self.target_mean, self.target_cov
                )
                self._writer.add_scalar(
                    tag="Error/Mean",
                    scalar_value=self.error_[0].l2,
                    global_step=epoch,
                    walltime=walltime,
                )
                self._writer.add_scalar(
                    tag="Error/Variance",
                    scalar_value=self.error_[1].l2,
                    global_step=epoch,
                    walltime=walltime,
                )
            self._writer.add_scalar(
                tag="Speed/Time_per_epoch",
                scalar_value=self.elapsed_time_ / (epoch or 1),
                global_step=epoch,
                walltime=walltime,
            )
            self._writer.add_histogram(
                tag="Distribution/ProjectedHistogram",
                values=particles,
                global_step=epoch,
                walltime=walltime,
            )

        self._writer = SummaryWriter(self.logdir)
        super()._inprocess_visualization(visualization_function=visualisation_function)

    def _postprocess_visualization(self):
        if self.hparams:
            metric_dict = {
                "Speed/Elapsed_time": self.elapsed_time_,
                "Speed/Time_per_epoch": self.elapsed_time_ / (self.epoch_ or 1),
            }
            if self.target_mean and self.target_cov:
                metric_dict.update(
                    {
                        "Error/Mean": self.error_[0].l2,
                        "Error/Variance": self.error_[1].l2,
                    }
                )
            self._writer.add_hparams(
                hparam_dict=self.hparams,
                metric_dict=metric_dict,
                name=".",
                global_step=self.epoch_,
            )
        self._writer.close()
        super()._postprocess_visualization()
