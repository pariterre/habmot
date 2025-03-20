from dataclasses import dataclass
from pathlib import Path

import biorbd
import biobuddy
import numpy as np
import scipy

from .config import Config, TrialConfig
from .trial import Trial


@dataclass(frozen=True)
class Model:
    _biomodel: biorbd.Model

    def reconstruct_kinematics(self, trial_config: TrialConfig):
        nq = self._biomodel.nbQ()
        trial = Trial.from_trial_config(trial_config)
        for time, data in zip(trial.time_stamps, trial.data):
            frame_count = len(time)

            kalman_params = biorbd.KalmanParam(100)
            kalman = biorbd.KalmanReconsIMU(self._biomodel, kalman_params)

            # Convert Roll, Pitch, Yaw of IMUs to homogenous matrix (in the same order as the model)
            model_imus = [imu.to_string() for imu in self._biomodel.IMUsNames()]
            targets = [_to_homogenous_matrix(euler=data[imu], seq="xyz") for imu in model_imus]

            # Prepare temporary variables and output
            q_out = biorbd.GeneralizedCoordinates(self._biomodel)
            qd_out = biorbd.GeneralizedVelocity(self._biomodel)
            qdd_out = biorbd.GeneralizedAcceleration(self._biomodel)
            q_recons = np.ndarray((nq, frame_count))
            for time_index in range(frame_count):
                target = np.hstack([val[:3, :3, time_index].T.reshape((-1,)) for val in targets])
                kalman.reconstructFrame(self._biomodel, target, q_out, qd_out, qdd_out)
                q_recons[:, time_index] = q_out.to_array()

            import bioviz

            b = bioviz.Viz(loaded_model=self._biomodel)
            b.load_movement(q_recons)
            b.exec()

    @staticmethod
    def from_biomod(file_path: str) -> "Model":
        return Model(_biomodel=biorbd.Model(file_path))

    @staticmethod
    def from_config(config: Config, models_folder: str, save_folder: str) -> "Model":
        model = biobuddy.BiomodModelParser(Path(models_folder) / config.model_filepath).to_real()

        static = Trial.from_trial_config(config.static)
        if static.header != ["Roll", "Pitch", "Yaw"]:
            raise NotImplementedError("Only Roll, Pitch, Yaw are supported for static")

        # Calibrate the model with the static trial
        for imu, data in static.concatenated_data.items():
            if imu not in [segment.name for segment in model.segments]:
                raise ValueError(f"IMU {imu} not found in the model")

            scs_in_global = biobuddy.utils.linear_algebra.mean_homogenous_matrix(
                _to_homogenous_matrix(euler=data, seq="xyz")
            )

            current_segment = model.segments[imu]
            rt_to_global = current_segment.segment_coordinate_system.scs[:, :, 0]
            while current_segment.parent_name:
                current_segment = model.segments[current_segment.parent_name]
                rt_to_global = current_segment.segment_coordinate_system.scs[:, :, 0] @ rt_to_global
            rt_to_global_transposed = biobuddy.SegmentCoordinateSystemReal(scs=rt_to_global).transpose.scs[:, :, 0]
            scs_in_local = np.eye(4)
            scs_in_local[:3, :3] = (rt_to_global_transposed @ scs_in_global)[:3, :3]
            model.segments[imu].imus.append(
                biobuddy.InertialMeasurementUnitReal(name=imu, parent_name=imu, scs=scs_in_local)
            )

        save_path: Path = Path(save_folder) / "static.bioMod"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.to_biomod(file_path=save_path)

        return Model.from_biomod(file_path=save_path.as_posix())


def _to_homogenous_matrix(euler: np.ndarray, seq: str) -> np.ndarray:
    scs = np.repeat(np.eye(4)[:, :, None], euler.shape[0], axis=2)
    scs[:3, :3, :] = np.einsum(
        "ijk->jik", scipy.spatial.transform.Rotation.from_euler(seq, euler, degrees=True).as_matrix().T
    )
    return scs
