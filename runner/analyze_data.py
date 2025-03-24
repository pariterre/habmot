import logging
from pathlib import Path

import habmot

_logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    current_folder = Path(__file__).parent
    data_folder = current_folder / ".." / "data"
    models_folder = current_folder / "models"
    results_folder = current_folder / ".." / "results"
    subjects = [
        "Jenn",
    ]

    # Load the config files
    configs: dict[str, habmot.Config] = {}
    for subject in subjects:
        _logger.info(f"Load subject: {subject}")
        configs[subject] = habmot.Config.from_config_file(Path(data_folder) / subject)

        # Load the static trial
        _logger.info(f"    Generate model from static")
        model = habmot.Model.from_config(
            config=configs[subject], models_folder=models_folder, save_folder=results_folder / subject
        )

        # model.reconstruct_kinematics(trial_config=configs[subject].static)

        trials: dict[str, habmot.Trial] = {}
        for key in configs[subject].trials:
            _logger.info(f"    Reconstruct trial {key}")
            model.reconstruct_kinematics(trial_config=configs[subject].trials[key])

        # TODO: Confirm with simple movement so the axes and right/left hand-side matrix are dealt with


if __name__ == "__main__":
    main()
