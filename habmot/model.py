from dataclasses import dataclass
from pathlib import Path

import biorbd
import biobuddy
import numpy as np
import scipy

from .config import Config
from .trial import Trial


@dataclass(frozen=True)
class Model:
    biomodel: biorbd.Model

    @staticmethod
    def from_biomod(file_path: str) -> "Model":
        return Model(biomodel=biorbd.Model(file_path))

    @staticmethod
    def from_config(config: Config, models_folder: str, save_folder: str) -> "Model":
        model = biobuddy.BiomodModelParser(Path(models_folder) / config.model_filepath).to_real()

        static = Trial.from_trial_config(config.static)
        if static.header != ["OriInc_q0", "OriInc_q1", "OriInc_q2", "OriInc_q3"]:
            raise NotImplementedError("Only quaternions are supported for now")

        for imu, data in static.concatenated_data.items():
            if imu not in [segment.name for segment in model.segments]:
                raise ValueError(f"IMU {imu} not found in the model")

            scs = np.repeat(np.eye(4)[:, :, None], data.shape[0], axis=2)
            scs[:3, :3, :] = np.einsum(
                "ijk->jik", scipy.spatial.transform.Rotation.from_quat(data[:, [1, 2, 3, 0]]).as_matrix().T
            )

            model.segments[imu].imus.append(
                biobuddy.InertialMeasurementUnitReal(
                    name=imu, parent_name=imu, scs=biobuddy.utils.linear_algebra.mean_homogenous_matrix(scs)
                )
            )

        save_path: Path = Path(save_folder) / "static.bioMod"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.to_biomod(file_path=save_path)

        return Model.from_biomod(file_path=save_path.as_posix())
