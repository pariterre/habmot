from pathlib import Path

import habmot
import biobuddy


def main():
    data_folder = Path(__file__).parent / ".." / "data"
    subjects = [
        "Jenn",
    ]

    # Load the config files
    configs = {}
    for subject in subjects:
        configs[subject] = habmot.Config.from_config_file(Path(data_folder) / subject)

    # Create the static model


if __name__ == "__main__":
    main()
