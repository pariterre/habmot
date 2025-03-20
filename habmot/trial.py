from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import TrialConfig


def _read_data_file(file: str) -> tuple[np.ndarray, np.ndarray]:
    file: Path = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"File {file} does not exist")

    with file.open("r") as f:
        content = f.read()

    # Break the content into lines, remove comments and remove empty lines
    lines = content.splitlines()
    lines = [lines[line_index].split("//")[0].strip() for line_index in range(len(lines))]
    lines = [line for line in lines if line]

    # Sanity check, the first line should be the header starting with "PacketCounter"
    if not lines[0].startswith("PacketCounter"):
        raise ValueError(f"Invalid data file {file}, missing header")
    header = lines[0].split("\t")

    frame_count = len(lines) - 1
    column_count = len(header) - 2  # Subtract the first two columns (PacketCounter and TimeStamp)

    # Parse the data
    frame_stamps = np.ndarray(frame_count, dtype=int)
    data = np.ndarray((frame_count, column_count), dtype=float)

    # Read the data
    for frame_index in range(frame_count):
        frame = lines[frame_index + 1].split("\t")
        frame_stamps[frame_index] = int(frame[0])
        # Skip time stamps as it is often not provided
        data[frame_index, :] = [
            np.nan if col + 2 >= len(frame) or frame[col + 2] == "" else float(frame[col + 2])
            for col in range(column_count)
        ]

    return frame_stamps, data


@dataclass(frozen=True)
class Trial:
    time_stamps: list[np.ndarray]
    data: list[dict[str, np.ndarray]]

    @staticmethod
    def from_trial_config(config: TrialConfig) -> "Trial":
        joints = {}
        time_stamps = None
        for key, file in config.files.items():
            time_stamps_tp, data = _read_data_file(file)
            if time_stamps is None:
                time_stamps = time_stamps_tp
            else:
                if not np.array_equal(time_stamps, time_stamps_tp):
                    raise ValueError(f"Time stamps do not match for joint {key}")

            joints[key] = data

        # Cut the data to fit the TrialConfig
        time_stamps_out = []
        data_out = []
        starts_frame = config.frames[:, 0]
        ends_frame = config.frames[:, 1]
        for start_frame, end_frames in zip(starts_frame, ends_frame):
            index_start = np.where(time_stamps == start_frame)[0][0]
            index_end = np.where(time_stamps == end_frames)[0][0]
            if index_end <= index_start:
                raise ValueError(f"Invalid frame range {start_frame} to {end_frames}")
            time_stamps_out.append(time_stamps[index_start:index_end])
            data_out.append({key: joints[key][index_start:index_end] for key in joints.keys()})
        return Trial(time_stamps=time_stamps_out, data=data_out)
