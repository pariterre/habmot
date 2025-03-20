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

        # Load the static avatar
        model = biobuddy.BiomodModelParser(
            (Path(__file__).parent / configs[subject].model_filepath).as_posix()
        ).to_real()

        # Load the static trial
        static = habmot.Trial.from_trial_config(configs[subject].static)
        trials = {key: habmot.Trial.from_trial_config(configs[subject].trials[key]) for key in configs[subject].trials}

        print(static)
        print(trials)

        # TODO: Replace the imu by the static values
        # TODO: Reconstruct the kinematics


if __name__ == "__main__":
    main()
