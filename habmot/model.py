from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

import biorbd
import biobuddy
import numpy as np
import scipy
from scipy.optimize import minimize

from .config import Config, TrialConfig
from .trial import Trial

_logger = logging.getLogger(__name__)


class ReconstructMethods(Enum):
    KALMAN = "kalman"
    GLOBAL_OPTIMIZATION = "global_optimization"


def _reconstruct_with_kalman(biorbd_model: biorbd.Model, targets: list[np.ndarray]) -> np.ndarray:
    _logger.info("\t\tReconstruct with Kalman filter")
    frame_count = targets[0].shape[-1]

    nq = biorbd_model.nbQ()
    kalman_params = biorbd.KalmanParam(frequency=100, noiseFactor=1e-10, errorFactor=1e-5)
    kalman = biorbd.KalmanReconsIMU(biorbd_model, kalman_params)

    # Prepare temporary variables and output
    q_out = biorbd.GeneralizedCoordinates(biorbd_model)
    qd_out = biorbd.GeneralizedVelocity(biorbd_model)
    qdd_out = biorbd.GeneralizedAcceleration(biorbd_model)
    q_recons = np.ndarray((nq, frame_count))

    for time_index in range(frame_count):
        target = np.hstack([val[:3, :3, time_index].T.reshape((-1,)) for val in targets])
        kalman.reconstructFrame(biorbd_model, target, q_out, qd_out, qdd_out)

        q_recons[:, time_index] = q_out.to_array()

    _logger.info("\t\tReconstruction done")
    return q_recons


def _reconstruct_with_global_optimization(biorbd_model: biorbd.Model, targets: list[np.ndarray]) -> np.ndarray:
    _logger.info("\t\tReconstruct with global optimization")
    frame_count = targets[0].shape[-1]

    def objective_function(q: np.ndarray) -> float:
        imus_model = [imu.to_array() for imu in biorbd_model.IMU(q)]

        defects = np.array(
            [
                targets[imu_index][:3, :3, time_index] - imus_model[imu_index][:3, :3]
                for imu_index in range(biorbd_model.nbIMUs())
            ]
        ).reshape((-1,))
        return np.linalg.norm(defects)

    q_recons = np.ndarray((biorbd_model.nbQ(), frame_count))
    q_init = np.zeros((biorbd_model.nbQ(),))
    for time_index in range(frame_count):
        if time_index % 200 == 0:
            _logger.info(f"\t\t\tTime index: {time_index}..")
        q_init = minimize(objective_function, q_init).x
        q_recons[:, time_index] = q_init

    _logger.info("\t\tReconstruction done")
    return q_recons


@dataclass(frozen=True)
class Model:
    _biomodel: biorbd.Model

    def reconstruct_kinematics(
        self, trial_config: TrialConfig, methods: ReconstructMethods | list[ReconstructMethods]
    ) -> dict[str, np.ndarray]:
        trial = Trial.from_trial_config(trial_config)

        axis_names = ["Roll", "Pitch", "Yaw"]
        for trial_time, data in zip(trial.time_stamps, trial.data):
            # Convert Roll, Pitch, Yaw of IMUs to homogenous matrix (in the same order as the model)
            targets: list[np.ndarray] = [
                _to_xsens_homogenous_matrix(
                    euler=data[imu][:, [axis_names.index(axis) for axis in trial_config.header if axis in axis_names]],
                )
                for imu in [imu.to_string() for imu in self._biomodel.IMUsNames()]
            ]

            if isinstance(methods, ReconstructMethods):
                methods = [methods]

            out = {}
            if ReconstructMethods.KALMAN in methods:
                out[ReconstructMethods.KALMAN] = _reconstruct_with_kalman(self._biomodel, targets)
            if ReconstructMethods.GLOBAL_OPTIMIZATION in methods:
                out[ReconstructMethods.GLOBAL_OPTIMIZATION] = _reconstruct_with_global_optimization(
                    self._biomodel, targets
                )

            if ReconstructMethods.KALMAN in methods and ReconstructMethods.GLOBAL_OPTIMIZATION in methods:
                mean_rmse = np.mean(
                    np.sqrt(
                        np.mean(
                            (out[ReconstructMethods.KALMAN] - out[ReconstructMethods.GLOBAL_OPTIMIZATION]) ** 2, axis=0
                        )
                    )
                )
                _logger.info(f"\t\tMean RMSE between Kalman and Global optimization is: {mean_rmse}")
            return out

    @staticmethod
    def from_biomod(file_path: str) -> "Model":
        return Model(_biomodel=biorbd.Model(file_path))

    @staticmethod
    def from_config(config: Config, models_folder: str, save_folder: str) -> "Model":
        model = biobuddy.BiomodModelParser(Path(models_folder) / config.model_filepath).to_real()

        axes_required = ["Roll", "Pitch", "Yaw"]
        static = Trial.from_trial_config(config.static)
        if any(axis not in static.header for axis in axes_required):
            raise NotImplementedError("Roll, Pitch, Yaw are required when loading the model")

        # Calibrate the model with the static trial
        for imu, data in static.concatenated_data.items():
            euler = data[:, [axes_required.index(axis) for axis in static.header if axis in axes_required]]

            if imu not in model.segments.keys():
                raise ValueError(f"Segment {imu} not found in the model. Available segments: {model.segments.keys()}")

            imu_in_global = biobuddy.utils.linear_algebra.mean_homogenous_matrix(
                _to_xsens_homogenous_matrix(euler=euler)
            )

            current_segment = model.segments[imu]
            rt_to_global = current_segment.segment_coordinate_system.scs[:, :, 0]
            while current_segment.parent_name:
                current_segment = model.segments[current_segment.parent_name]
                rt_to_global = current_segment.segment_coordinate_system.scs[:, :, 0] @ rt_to_global
            rt_to_global_transposed = biobuddy.SegmentCoordinateSystemReal(scs=rt_to_global).transpose.scs[:, :, 0]

            scs_in_local = np.eye(4)
            scs_in_local[:3, :3] = (rt_to_global_transposed @ imu_in_global)[:3, :3]
            model.segments[imu].imus.append(
                biobuddy.InertialMeasurementUnitReal(name=imu, parent_name=imu, scs=scs_in_local)
            )

        save_path: Path = Path(save_folder) / "static.bioMod"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.to_biomod(file_path=save_path)

        return Model.from_biomod(file_path=save_path.as_posix())

    def animate(self, q: np.ndarray):
        import bioviz

        b = bioviz.Viz(loaded_model=self._biomodel)
        b.load_movement(q)
        b.exec()


def _to_xsens_homogenous_matrix(euler: np.ndarray) -> np.ndarray:

    seq = "XYZ"
    euler = euler[:, [1, 2, 0]]

    scs = np.repeat(np.eye(4)[:, :, None], euler.shape[0], axis=2)
    scs[:3, :3, :] = np.einsum(
        "ijk->jki", scipy.spatial.transform.Rotation.from_euler(seq, euler, degrees=True).as_matrix()
    )

    return scs

    # CANDIDATES (Same direction)
    # seq = "XYZ" / euler = euler[:, [1, 2, 0]]
    # seq = "XZY" / euler = euler[:, [1, 2, 0]]
    # seq = "YZX" / euler = euler[:, [2, 0, 1]]
    # seq = "ZYX" / euler = euler[:, [2, 0, 1]]
    # seq = "YXZ" / euler = euler[:, [2, 1, 0]]
    # seq = "ZXY" / euler = euler[:, [2, 1, 0]]
    # seq = "yxz" / euler = euler[:, [0, 1, 2]]
    # seq = "zxy" / euler = euler[:, [0, 1, 2]]
    # seq = "yzx" / euler = euler[:, [0, 2, 1]]
    # seq = "zyx" / euler = euler[:, [0, 2, 1]]
    # seq = "xyz" / euler = euler[:, [1, 0, 2]]
    # seq = "xzy" / euler = euler[:, [1, 0, 2]]

    # CANDIDATES (Opposite direction)
    # seq = "YXZ" / euler = euler[:, [1, 0, 2]]
    # seq = "ZXY" / euler = euler[:, [1, 0, 2]]
    # seq = "yzx" / euler = euler[:, [1, 2, 0]]
    # seq = "zyx" / euler = euler[:, [1, 2, 0]]
    # seq = "yxz" / euler = euler[:, [2, 0, 1]]
    # seq = "zxy" / euler = euler[:, [2, 0, 1]]
    # seq = "yzx" / euler = euler[:, [2, 1, 0]]
    # seq = "zyx" / euler = euler[:, [2, 1, 0]]
