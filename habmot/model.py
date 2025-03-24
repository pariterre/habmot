from dataclasses import dataclass
from pathlib import Path

import biorbd
import biobuddy
import numpy as np
import scipy

from .config import Config, TrialConfig
from .trial import Trial


_xsens_euler_sequence: str = "zyx"


def _reconstruct_with_kalman(biorbd_model: biorbd.Model, targets: list[np.ndarray]) -> np.ndarray:
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

        import bioviz

    b = bioviz.Viz(loaded_model=biorbd_model)
    b.load_movement(q_recons)
    b.exec()


def _reconstruct_global_optimization(biorbd_model: biorbd.Model, targets: list[np.ndarray]) -> np.ndarray:
    from scipy.optimize import minimize


@dataclass(frozen=True)
class Model:
    _biomodel: biorbd.Model

    def reconstruct_kinematics(self, trial_config: TrialConfig):
        trial = Trial.from_trial_config(trial_config)

        axis_names = ["Roll", "Pitch", "Yaw"]
        for trial_time, data in zip(trial.time_stamps, trial.data):
            # Convert Roll, Pitch, Yaw of IMUs to homogenous matrix (in the same order as the model)
            targets: list[np.ndarray] = [
                _to_homogenous_matrix(
                    euler=data[imu][:, [axis_names.index(axis) for axis in trial_config.header if axis in axis_names]],
                    seq=_xsens_euler_sequence,
                )
                for imu in [imu.to_string() for imu in self._biomodel.IMUsNames()]
            ]

            _reconstruct_with_kalman(self._biomodel, targets)

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
                _to_homogenous_matrix(euler=euler, seq=_xsens_euler_sequence)
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


def _to_homogenous_matrix(euler: np.ndarray, seq: str) -> np.ndarray:
    # Reorder the euler angles to match the seq (e.g. zyx -> euler[:, [2, 1, 0]])
    euler = euler[:, [seq.index(axis) for axis in "xyz"]]
    euler[:, [0, 2]] = -euler[:, [0, 2]]

    # Change the seq for intrinsic rotation
    seq = seq.upper()

    scs = np.repeat(np.eye(4)[:, :, None], euler.shape[0], axis=2)
    scs[:3, :3, :] = np.einsum(
        "ijk->jki", scipy.spatial.transform.Rotation.from_euler(seq, euler, degrees=True).as_matrix()
    )
    return scs
