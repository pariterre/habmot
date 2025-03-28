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


class ReconstructMethods(Enum):
    NO_MODEL = "no_model"
    KALMAN = "kalman"
    GLOBAL_OPTIMIZATION = "global_optimization"


def _reconstruct_no_model(biorbd_model: biorbd.Model, targets: list[np.ndarray]) -> np.ndarray:
    frame_count = targets[0].shape[-1]

    q_recons = np.zeros((biorbd_model.nbQ(), frame_count))
    imu_names = [imu.to_string() for imu in biorbd_model.IMUsNames()]
    for imu in imu_names:
        # Find the parent of the IMU
        imu_idx = imu_names.index(imu)
        segment = biorbd_model.segment(imu)
        local_imu = biorbd_model.IMU(imu_idx).to_array()[:3, :3]

        angle_indices = range(
            segment.getFirstDofIndexInGeneralizedCoordinates(biorbd_model) + segment.nbDofTrans(),
            segment.getLastDofIndexInGeneralizedCoordinates(biorbd_model) + 1,
        )
        angle_sequence = segment.seqR().to_string()

        projected_imu = np.ndarray((frame_count, 3, 3))
        for time_index in range(frame_count):
            # Use the previously placed IMU as basis for computing the base IMU. This assume the kinematics
            # chain is declared with parent IMU before child IMU
            base_imus_in_global = biorbd_model.IMU(q_recons[:, time_index], imu_idx).to_array()[:3, :3]
            projected_imu[time_index, :, :] = local_imu.T @ (
                base_imus_in_global.T @ targets[imu_idx][:3, :3, time_index]
            )

            scipy.spatial.transform.Rotation.from_matrix(
                local_imu.T @ (base_imus_in_global.T @ targets[imu_idx][:3, :3, time_index])
            ).as_euler("xyz", degrees=True).T
        q_recons[angle_indices, :] = (
            scipy.spatial.transform.Rotation.from_matrix(projected_imu).as_euler(angle_sequence).T
        )

        # time_index = 0
        # rts = biorbd_model.globalJCS(np.zeros(biorbd_model.nbQ()), 0).to_array()
        # base_imus = biorbd_model.IMU(np.zeros(biorbd_model.nbQ()), imu_idx).to_array()[:3, :3]
        # local_imus = biorbd_model.IMU(imu_idx).to_array()[:3, :3]

        # scipy.spatial.transform.Rotation.from_matrix(base_imus).as_euler("xyz", degrees=True).T
        # scipy.spatial.transform.Rotation.from_matrix(targets[imu_idx][:3, :3, time_index]).as_euler(
        #     "xyz", degrees=True
        # ).T
        # scipy.spatial.transform.Rotation.from_matrix(base_imus.T @ targets[imu_idx][:3, :3, time_index]).as_euler("xyz", degrees=True).T
        # scipy.spatial.transform.Rotation.from_matrix(local_imus @ (base_imus.T @ targets[imu_idx][:3, :3, time_index])).as_euler("xyz", degrees=True).T

    return q_recons


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

    # Compute reconstruction error
    defects = np.ndarray((frame_count, biorbd_model.nbIMUs() * 9))
    for time_index in range(frame_count):
        imus_model = [imu.to_array() for imu in biorbd_model.IMU(q_recons[:, time_index])]
        defects[time_index, :] = np.array(
            [
                targets[imu_index][:3, :3, time_index] - imus_model[imu_index][:3, :3]
                for imu_index in range(biorbd_model.nbIMUs())
            ]
        ).reshape((-1,))
    rmse = np.sqrt(np.mean(defects**2))
    _logger.info(f"\t\t\tKalman RMSE: {rmse}")

    _logger.info("\t\tReconstruction done")
    return q_recons


def _reconstruct_with_global_optimization(biorbd_model: biorbd.Model, targets: list[np.ndarray]) -> np.ndarray:
    _logger.info("\t\tReconstruct with global optimization")
    frame_count = targets[0].shape[-1]

    def objective_function(q: np.ndarray) -> float:
        imus_model = [imu.to_array() for imu in biorbd_model.IMU(q)]

        defects = np.array(
            [
                targets[imu_index][:3, :3, time_index].T @ imus_model[imu_index][:3, :3] - np.eye(3)
                for imu_index in range(biorbd_model.nbIMUs())
            ]
        ).reshape((-1,))
        return np.linalg.norm(defects)

    q_recons = np.ndarray((biorbd_model.nbQ(), frame_count))
    q_init = np.zeros((biorbd_model.nbQ(),))
    for time_index in range(frame_count):
        if time_index % 200 == 0:
            _logger.info(f"\t\t\tTime index: {time_index}..")
        q_init = scipy.optimize.least_squares(objective_function, q_init).x
        q_recons[:, time_index] = q_init
        if time_index == 1:
            break

    _logger.info("\t\tReconstruction done")
    return q_recons


def _realign_vertical_rt(root_in_global: np.ndarray, reference_imu_in_global: np.ndarray) -> np.ndarray:
    def best_vertical_rotation(angle: np.array) -> biobuddy.SegmentReal:
        root = scipy.spatial.transform.Rotation.from_euler("Z", angle[0]).as_matrix() @ root_in_global[:3, :3]

        saggital_error = np.dot(root[:3, root_x], reference_imu_in_global[:3, imu_x]) - 1
        frontal_error = np.dot(root[:3, root_y], reference_imu_in_global[:3, imu_y]) - 1
        return saggital_error**2 + frontal_error**2

    def find_axes_indices(matrix: np.ndarray) -> tuple[int, int, int]:
        for index, _ in enumerate("xyz"):
            if np.abs(np.dot(matrix[:3, index], np.array([0, 0, 1]))) > 0.9:
                z_axis = index
                x_axis = (z_axis + 1) % 3
                y_axis = (z_axis + 2) % 3
                return x_axis, y_axis, z_axis
        raise ValueError("No vertical axis found")

    # Find the axes of the matrices
    root_x, root_y, root_z = find_axes_indices(root_in_global[:3, :3])
    imu_x, imu_y, imu_z = find_axes_indices(reference_imu_in_global[:3, :3])

    # If the z-axis of a matrix is pointing down, flip the x-axis (effectively rotating the matrix 180 degrees)
    if np.dot(root_in_global[:3, root_z], [0, 0, 1]) < 0:
        root_in_global = root_in_global.copy()
        root_in_global[:, root_x] *= -1
    if np.dot(reference_imu_in_global[:3, imu_z], [0, 0, 1]) < 0:
        reference_imu_in_global = reference_imu_in_global.copy()
        reference_imu_in_global[:, imu_x] *= -1

    value = scipy.optimize.minimize(best_vertical_rotation, 0).x[0]
    root_rt = np.eye(4)
    root_rt[:3, :3] = scipy.spatial.transform.Rotation.from_euler("Z", value).as_matrix()
    return root_rt


@dataclass(frozen=True)
class Model:
    _biomodel: biorbd.Model

    def reconstruct_kinematics(
        self, trial_config: TrialConfig, methods: ReconstructMethods | list[ReconstructMethods]
    ) -> dict[str, np.ndarray]:
        trial = Trial.from_trial_config(trial_config)

        axis_names = ["Roll", "Pitch", "Yaw"]
        for data in trial.data:
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
            if ReconstructMethods.NO_MODEL in methods:
                out[ReconstructMethods.NO_MODEL] = _reconstruct_no_model(self._biomodel, targets)
            if ReconstructMethods.KALMAN in methods:
                out[ReconstructMethods.KALMAN] = _reconstruct_with_kalman(self._biomodel, targets)
            if ReconstructMethods.GLOBAL_OPTIMIZATION in methods:
                out[ReconstructMethods.GLOBAL_OPTIMIZATION] = _reconstruct_with_global_optimization(
                    self._biomodel, targets
                )

            all_methods = list(out.keys())
            for method_index, method1 in enumerate(all_methods[:-1]):
                for method2 in all_methods[(method_index + 1) :]:
                    mean_rmse = np.mean(np.sqrt(np.mean((out[method1] - out[method2]) ** 2, axis=0)))
                    _logger.info(f"\t\tMean RMSE between {method1} and {method2} is: {mean_rmse}")

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
        data_indices = [axes_required.index(axis) for axis in static.header if axis in axes_required]

        # Do a prealignment of the root segment with its corresponding IMU
        root_segment = model.segments[0]
        root_segment_rt = root_segment.segment_coordinate_system.scs[:, :, 0]
        root_imu = biobuddy.utils.linear_algebra.mean_homogenous_matrix(
            _to_xsens_homogenous_matrix(euler=static.concatenated_data[root_segment.name][:, data_indices])
        )
        root_correction = _realign_vertical_rt(root_in_global=root_segment_rt, reference_imu_in_global=root_imu)
        root_segment.segment_coordinate_system.scs[:, :, 0] = root_correction @ root_segment_rt

        # Calibrate the model with the static trial
        for imu, data in static.concatenated_data.items():
            euler = data[:, data_indices]

            # # Validity check
            # to_matrix = _to_homogenous_matrix(euler=euler, seq="ZYX")[:3, :3, :]
            # provided_matrix = data[:, 3:].reshape((-1, 3, 3)).T
            # for i in range(to_matrix.shape[-1]):
            #     if not np.allclose(to_matrix[:, :, i], provided_matrix[:, :, i], atol=1e-5):
            #         raise ValueError(f"Provided matrix for {imu} is not close to the calculated matrix")

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
            rt_to_global_T = biobuddy.SegmentCoordinateSystemReal(scs=rt_to_global).transpose.scs[:, :, 0]

            scs_in_local = np.eye(4)
            scs_in_local[:3, :3] = (rt_to_global_T @ imu_in_global)[:3, :3]
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


def _to_xsens_homogenous_matrix(euler: np.ndarray) -> np.ndarray:
    seq = "ZYX"
    euler = euler[:, [2, 1, 0]]

    scs = np.repeat(np.eye(4)[:, :, None], euler.shape[0], axis=2)
    scs[:3, :3, :] = np.einsum(
        "ijk->jki", scipy.spatial.transform.Rotation.from_euler(seq, euler, degrees=True).as_matrix()
    )

    return scs

    # CANDIDATES (Same direction)
    # seq = "ZYX"
    # euler = euler[:, [1, 0, 2]]
    # # euler[:, 0] = -euler[:, 0]
    # euler[:, 1] = -euler[:, 1]
    # # euler[:, 2] = -euler[:, 2]
