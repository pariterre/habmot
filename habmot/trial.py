from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable

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
    header = lines[0].split("\t")[2:]  # Remove the first two columns (PacketCounter and TimeStamp)
    lines = lines[1:]  # We don't need the header anymore

    frame_count = len(lines)
    column_count = len(header)

    # Parse the data
    # We must np.arange the time stamps because they wrap after 65535
    first_frame_count = int(lines[0].split("\t")[0])
    frame_stamps = np.arange(frame_count, dtype=int) + first_frame_count
    data = np.ndarray((frame_count, column_count), dtype=float)

    # Read the data
    for frame_index in range(frame_count):
        frame = lines[frame_index].split("\t")[2:]  # Remove the first two columns (PacketCounter and TimeStamp)
        data[frame_index, :] = [
            np.nan if col >= len(frame) or frame[col] == "" else float(frame[col]) for col in range(column_count)
        ]

    return header, frame_stamps, data


@dataclass(frozen=True)
class Trial:
    header: list[str]
    time_stamps: list[np.ndarray]
    data: list[dict[str, np.ndarray]]

    @cached_property
    def concatenated_time_stamps(self) -> np.ndarray:
        return np.concatenate(self.time_stamps)

    @cached_property
    def concatenated_data(self) -> dict[str, np.ndarray]:
        out = {}
        for key in self.data[0].keys():
            out[key] = np.concatenate([data[key] for data in self.data])
        return out

    def plot(self, data_keys: str | Iterable[str] = None, merged: bool = False, show_now: bool = True):
        import matplotlib.pyplot as plt

        def draw_graph(title: str, key: str, time_stamps: np.ndarray, data: np.ndarray):
            plt.figure(title)
            plt.title(title)
            plt.plot(time_stamps, data)
            plt.xlabel("Time (frame)")
            plt.ylabel(key)

        if data_keys is None:
            self.plot(data_keys=self.concatenated_data.keys(), merged=merged, show_now=show_now)
            return

        if isinstance(data_keys, str):
            data_keys = [data_keys]

        for key in data_keys:
            if merged:
                draw_graph(
                    title=key, key=key, time_stamps=self.concatenated_time_stamps, data=self.concatenated_data[key]
                )
            else:
                for data, time_stamps in zip(self.data, self.time_stamps):
                    draw_graph(
                        title=f"{key} from {time_stamps[0]} to {time_stamps[-1]}",
                        key=key,
                        time_stamps=time_stamps,
                        data=data[key],
                    )

        if show_now:
            plt.show()

    @staticmethod
    def from_trial_config(config: TrialConfig) -> "Trial":
        headers = None
        time_stamps = None
        joints = {}
        for key, file in config.files.items():
            header, time_stamps_tp, data = _read_data_file(file)
            if headers is None:
                headers = header
            if not np.array_equal(headers, header):
                raise ValueError(f"Header does not match for joint {key}")

            if time_stamps is None:
                time_stamps = time_stamps_tp
            if not np.array_equal(time_stamps, time_stamps_tp):
                raise ValueError(f"Time stamps do not match for joint {key}")

            cols = [i for i, col in enumerate(header) if col in config.header]
            joints[key] = data[:, cols]

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
        return Trial(header=config.header, time_stamps=time_stamps_out, data=data_out)
