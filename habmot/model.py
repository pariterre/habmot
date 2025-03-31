from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

import biorbd
import biobuddy
import numpy as np
import scipy

from .config import Config, TrialConfig
from .trial import Trial

_logger = logging.getLogger(__name__)


def _reconstruct_with_kalman(biorbd_model: biorbd.Model, targets: dict[str, np.ndarray]) -> np.ndarray:
    _logger.info("\t\tReconstruct with Kalman filter")
    frame_count = targets[list(targets.keys())[0]].shape[-1]

    nq = biorbd_model.nbQ()
    kalman_params = biorbd.KalmanParam(frequency=100, noiseFactor=1e-10, errorFactor=1e-5)
    kalman = biorbd.KalmanReconsIMU(biorbd_model, kalman_params)

    # Prepare temporary variables and output
    q_out = biorbd.GeneralizedCoordinates(biorbd_model)
    qd_out = biorbd.GeneralizedVelocity(biorbd_model)
    qdd_out = biorbd.GeneralizedAcceleration(biorbd_model)
    q_recons = np.ndarray((nq, frame_count))

    defects = np.ndarray((frame_count, biorbd_model.nbIMUs() * 9))
    for time_index in range(frame_count):
        target = []
        for segment in biorbd_model.segments():
            name = segment.name().to_string()
            if name in targets:
                target.append(targets[name][:3, :3, time_index].T.reshape((-1,)))
        target = np.hstack(target)
        kalman.reconstructFrame(biorbd_model, target, q_out, qd_out, qdd_out)

        q_recons[:, time_index] = q_out.to_array()

        imus_model = np.hstack(
            [imu.to_array()[:3, :3].T.reshape((-1,)) for imu in biorbd_model.IMU(q_recons[:, time_index])]
        )
        defects[time_index, :] = target - imus_model

    rmse = np.sqrt(np.mean(defects**2))
    _logger.info(f"\t\t\tKalman RMSE: {rmse}")

    _logger.info("\t\tReconstruction done")
    return q_recons


def _realign_vertical_rt(root_in_global: np.ndarray, reference_imu_in_global: np.ndarray) -> np.ndarray:
    def best_vertical_rotation(angle: np.array) -> biobuddy.SegmentReal:
        root = scipy.spatial.transform.Rotation.from_euler("Z", angle[0]).as_matrix() @ root_in_global[:3, :3]

        saggital_error = np.dot(root[:3, root_saggital], -reference_imu_in_global[:3, imu_saggital]) - 1
        frontal_error = np.dot(root[:3, root_frontal], -reference_imu_in_global[:3, imu_frontal]) - 1
        return saggital_error**2 + frontal_error**2

    def find_axes_indices(matrix: np.ndarray) -> tuple[int, int, int]:
        saggital_axis = None
        frontal_axis = None
        vertical_axis = None

        for index, _ in enumerate("xyz"):
            if np.abs(np.dot(matrix[:3, index], np.array([1, 0, 0]))) > 0.85:
                saggital_axis = index
                break
        if saggital_axis is None:
            raise ValueError("No saggital axis found")

        for index, _ in enumerate("xyz"):
            if np.abs(np.dot(matrix[:3, index], np.array([0, 1, 0]))) > 0.85:
                frontal_axis = index
                break
        if frontal_axis is None:
            raise ValueError("No frontal axis found")

        for index, _ in enumerate("xyz"):
            if np.abs(np.dot(matrix[:3, index], np.array([0, 0, 1]))) > 0.85:
                vertical_axis = index
                break
        if vertical_axis is None:
            raise ValueError("No vertical axis found")

        return saggital_axis, frontal_axis, vertical_axis

    # Find the axes of the matrices
    root_saggital, root_frontal, root_vertical = find_axes_indices(root_in_global[:3, :3])
    imu_saggital, imu_frontal, imu_vertical = find_axes_indices(reference_imu_in_global[:3, :3])

    # # If the z-axis of a matrix is pointing down, flip the x-axis (effectively rotating the matrix 180 degrees)
    if np.dot(root_in_global[:3, root_vertical], [0, 0, 1]) < 0:
        root_in_global = root_in_global.copy()
        root_in_global[:, root_saggital] *= -1
    if np.dot(reference_imu_in_global[:3, imu_vertical], [0, 0, 1]) < 0:
        reference_imu_in_global = reference_imu_in_global.copy()
        reference_imu_in_global[:, imu_saggital] *= -1

    value = scipy.optimize.minimize(best_vertical_rotation, 0).x[0]
    root_rt = np.eye(4)
    root_rt[:3, :3] = scipy.spatial.transform.Rotation.from_euler("Z", value).as_matrix()
    return root_rt


@dataclass(frozen=True)
class Model:
    _biomodel: biorbd.Model
    _imu_seq: str = "ZYX"

    def reconstruct_kinematics(
        self, trial_config: TrialConfig, animate: bool = False, save_folder: str = None
    ) -> list[np.ndarray]:
        trial = Trial.from_trial_config(trial_config)

        out: list[np.ndarray] = []
        axis_names = ["Roll", "Pitch", "Yaw"]
        for i, data in enumerate(trial.data):
            # Convert Roll, Pitch, Yaw of IMUs to homogenous matrix (in the same order as the model)
            targets: dict[str, np.ndarray] = {
                imu: _to_homogenous_matrix(
                    euler=data[imu][:, [axis_names.index(axis) for axis in trial_config.header if axis in axis_names]],
                    seq=Model._imu_seq,
                )
                for imu in [imu.to_string() for imu in self._biomodel.IMUsNames()]
            }

            all_q = _reconstruct_with_kalman(self._biomodel, targets)

            if animate:
                self.animate(all_q, targets)

            if save_folder is not None:
                header = [name.to_string() for name in self._biomodel.nameDof()]
                multiplier = np.ones(len(header))
                for name in header:
                    if "Rot" in name:
                        multiplier[header.index(name)] = 180 / np.pi
                file_path = Path(save_folder) / f"{trial_config.name}_{i + 1}.csv"
                to_write = all_q.T * multiplier
                np.savetxt(file_path, to_write, delimiter=",", header=",".join(header), fmt="%.6f")
                _logger.info(f"\t\tSave reconstructed data to {file_path}")

            out.append(all_q)
        return out

    @staticmethod
    def from_biomod(file_path: str) -> "Model":
        return Model(_biomodel=biorbd.Model(file_path))

    @staticmethod
    def from_config(config: Config, models_folder: str, save_folder: str, show_static: bool = False) -> "Model":
        model = biobuddy.BiomodModelParser(Path(models_folder) / config.model_filepath).to_real()

        axis_names = ["Roll", "Pitch", "Yaw"]
        static = Trial.from_trial_config(config.static)

        if any(axis not in static.header for axis in axis_names):
            raise NotImplementedError("Roll, Pitch, Yaw are required when loading the model")
        data_indices = [axis_names.index(axis) for axis in static.header if axis in axis_names]

        # Do a prealignment of the root segment with its corresponding IMU
        root_segment = model.segments[0]
        root_segment_rt = root_segment.segment_coordinate_system.scs[:, :, 0]
        root_imu = biobuddy.utils.linear_algebra.mean_homogenous_matrix(
            _to_homogenous_matrix(
                euler=static.concatenated_data[root_segment.name][:, data_indices], seq=Model._imu_seq
            )
        )
        root_correction = _realign_vertical_rt(root_in_global=root_segment_rt, reference_imu_in_global=root_imu)
        root_segment.segment_coordinate_system.scs[:, :, 0] = root_correction @ root_segment_rt
        root_segment.segment_coordinate_system.is_in_global = False

        targets: dict[str, np.ndarray] = {
            imu: _to_homogenous_matrix(euler=static.concatenated_data[imu][:, data_indices], seq=Model._imu_seq)
            for imu in static.concatenated_data.keys()
        }

        # Calibrate the model with the static trial
        for imu, data in targets.items():
            if imu not in model.segments.keys():
                raise ValueError(f"Segment {imu} not found in the model. Available segments: {model.segments.keys()}")

            imu_in_global = biobuddy.utils.linear_algebra.mean_homogenous_matrix(data)

            current_segment = model.segments[imu]
            rt_to_global = current_segment.segment_coordinate_system.scs[:, :, 0]
            while current_segment.parent_name:
                current_segment = model.segments[current_segment.parent_name]
                rt_to_global = current_segment.segment_coordinate_system.scs[:, :, 0] @ rt_to_global
            rt_to_global_T = biobuddy.SegmentCoordinateSystemReal(scs=rt_to_global).transpose.scs[:, :, 0]

            scs_in_local = np.eye(4)
            scs_in_local[:3, :3] = (rt_to_global_T @ imu_in_global)[:3, :3]
            model.segments[imu].imus.append(
                biobuddy.InertialMeasurementUnitReal(name=imu, parent_name=imu, scs=scs_in_local)
            )

        save_path: Path = Path(save_folder) / "static.bioMod"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.to_biomod(file_path=save_path)
        if show_static:
            _animate(biorbd.Model(save_path.as_posix()), imu_targets=targets)

        return Model.from_biomod(file_path=save_path.as_posix())

    def animate(self, q: np.ndarray, imu_targets: dict[str, np.ndarray] = None) -> None:
        _animate(self._biomodel, q, imu_targets)


def _animate(biorbd_model: biorbd.Model, q: np.ndarray = None, imu_targets: dict[str, np.ndarray] = None) -> None:
    import bioviz

    b = bioviz.Viz(loaded_model=biorbd_model, show_imus=True, show_local_ref_frame=True)
    if q is not None:
        b.load_movement(q)

    if imu_targets is not None:
        b.load_experimental_imus(imu_targets)
    b.exec()


def _to_homogenous_matrix(euler: np.ndarray, seq: str) -> np.ndarray:
    # Change the seq for intrinsic rotation
    seq = seq.upper()

    # Reorder the euler angles to match the seq (e.g. zyx -> euler[:, [2, 1, 0]])
    euler = euler[:, [seq.index(axis) for axis in "XYZ"]]

    scs = np.repeat(np.eye(4)[:, :, None], euler.shape[0], axis=2)
    scs[:3, :3, :] = np.einsum(
        "ijk->jki", scipy.spatial.transform.Rotation.from_euler(seq, euler, degrees=True).as_matrix()
    )
    return scs
