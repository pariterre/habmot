from pathlib import Path
import habmot


def main():
    data_folder = Path(__file__).parent / ".." / "data"
    subjects = [
        "Jenn",
    ]

    configs = {}
    for subject in subjects:
        configs[subject] = habmot.Config.from_config_file(Path(data_folder) / subject)


if __name__ == "__main__":
    main()
