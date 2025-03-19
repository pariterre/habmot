from pathlib import Path

import habmot
import numpy.testing as npt


def test_version():
    assert habmot.__version__ == "0.1.0"


def test_parse_config_file():
    config = habmot.Config.from_config_file(Path(__file__).parent)

    assert config.model_filepath == "models/mymodel.biomod"
    assert config.imus == {"left_ankle": "first_imu", "left_thigh": "second_imu"}
    assert config.static_files == ["Tpose*/Tpose-*-first_imu.txt", "Tpose*/Tpose-*-second_imu.txt"]
    npt.assert_equal(config.static_frames, [[10, 20]])
    assert config.trials_files == {
        "first": ["first*/first*-*-first_imu.txt", "first*/first*-*-second_imu.txt"],
        "second": ["second*/second*-*-first_imu.txt", "second*/second*-*-second_imu.txt"],
    }
    assert "first" in config.trials_frames
    assert "second" in config.trials_frames
    npt.assert_equal(config.trials_frames["first"], [[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]])
    npt.assert_equal(config.trials_frames["second"], [[100, 200], [300, 400], [500, 600], [700, 800], [900, 1000]])
