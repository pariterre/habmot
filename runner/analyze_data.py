from pathlib import Path

import habmot
import biobuddy


def main():
    data_folder = Path(__file__).parent / ".." / "data"
    subjects = [
        "Jenn",
    ]

    # Load the config files
    configs: dict[str, habmot.Config] = {}
    for subject in subjects:
        configs[subject] = habmot.Config.from_config_file(Path(data_folder) / subject)

        # Create the static model
        model = biobuddy.BiomodModelParser(
            (Path(__file__).parent / configs[subject].model_filepath).as_posix()
        ).to_real()

        # TODO: Fetch the static trial
        # TODO: Replace the imu by the static values
        # TODO: Fetch data from trials
        # TODO: Reconstruct the kinematics


if __name__ == "__main__":
    main()
