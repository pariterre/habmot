from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ImuConfig:
    names: list[str]


@dataclass(frozen=True)
class TrialConfig:
    files: dict[str, str]  # imu_name: file_path
    header: list[str]
    frames: npt.NDArray[np.int_]


@dataclass(frozen=True)
class Config:
    model_filepath: str

    imus: ImuConfig

    static: TrialConfig
    trials: dict[str, TrialConfig]

    @staticmethod
    def from_config_file(subject_folder: str) -> "Config":
        """
        Load the config file from the subject folder and return a Config object.

        Parameters
        ----------
        subject_folder : str
            The absolute path to the subject folder.
        """
        # Load the content of the file
        config = json.loads((Path(subject_folder) / "conf.json").read_text())

        # Read down the dict and replace any [value] with the value in the dict.
        def expand_str(item) -> tuple[list[str], bool]:
            def get_current_expand_iterator(current_expand):
                if isinstance(current_expand, list):
                    return range(len(current_expand))
                elif isinstance(current_expand, dict):
                    return current_expand.keys()
                else:
                    raise ValueError(f"Unknown type {type(current_expand)}")

            def replace_with_value(to_replace, value):
                if isinstance(to_replace, dict):
                    return {key: replace_with_value(to_replace[key], value) for key in to_replace}
                elif isinstance(to_replace, list):
                    return [replace_with_value(tp, value) for tp in to_replace]
                elif isinstance(to_replace, str):
                    if isinstance(value, dict):
                        return {key: to_replace.replace(pattern, v) for key, v in value.items()}
                    elif isinstance(value, list):
                        return [to_replace.replace(pattern, v) for v in value]
                    elif isinstance(value, str):
                        return to_replace.replace(pattern, value)
                    else:
                        raise ValueError(f"Unknown type {type(value)}")
                else:
                    raise ValueError(f"Unknown type {type(to_replace)}")

            def check_if_fully_expanded(current_expand) -> bool:
                if isinstance(current_expand, dict):
                    return all(check_if_fully_expanded(val) for val in current_expand.values())
                elif isinstance(current_expand, list):
                    return all(check_if_fully_expanded(val) for val in current_expand)
                elif isinstance(current_expand, str):
                    return not re.search(replacable_tag, current_expand)
                else:
                    raise ValueError(f"Unknown type {type(current_expand)}")

            def collapse_len_one_lists(to_replace):
                if isinstance(to_replace, dict):
                    for key in to_replace:
                        to_replace[key] = collapse_len_one_lists(to_replace[key])
                elif isinstance(to_replace, list) and len(to_replace) == 1:
                    return to_replace[0]
                else:
                    return to_replace

            # First change the current key into a list. This is useful to deal with (*) elements.
            current_expand = [item]

            # Find all the [...] but not the \[...\] elements (i.e. with an escape caracter)
            replacable_tag = r"(?<!\\)\[.*?\]"
            # replacable_tag = r"\[.*?\]"
            patters = set(re.findall(replacable_tag, item))
            # sort the patters by alphabetical order to ensure that the replacement is done in the right order.
            for pattern in patters:
                value = config  # Search the value to replace the element with
                for element in pattern[1:-1].split("."):
                    if element == "keys(*)":
                        value = [key for key in value.keys()]
                    elif element == "values(*)":
                        value = {key: item for key, item in value.items()}
                    elif element.endswith("(*)"):
                        value = value[element[:-3]]
                    else:
                        value = value[element]

                current_expand = replace_with_value(current_expand, value)

            could_fully_expand = check_if_fully_expanded(current_expand)
            current_expand = collapse_len_one_lists(current_expand)
            return current_expand, could_fully_expand

        def replace_config_values(config_item, need_repass: bool = None) -> bool:
            if need_repass is None:
                need_repass = False

            if isinstance(config_item, list):
                for index, item in enumerate(config_item):
                    if isinstance(item, dict) or isinstance(item, list):
                        need_repass = replace_config_values(item) or need_repass
                    elif isinstance(item, str):
                        config_item[index], could_fully_expand = expand_str(item)
                        if not could_fully_expand:
                            need_repass = True
                    elif isinstance(item, float) or isinstance(item, int):
                        pass
                    else:
                        raise ValueError(f"Unknown type {type(item)}")
            elif isinstance(config_item, dict):
                for key, item in config_item.items():
                    if isinstance(item, dict) or isinstance(item, list):
                        need_repass = replace_config_values(item) or need_repass
                    elif isinstance(item, str):
                        config_item[key], could_fully_expand = expand_str(item)
                        if not could_fully_expand:
                            need_repass = True
                    else:
                        raise ValueError(f"Unknown type {type(item)}")
            else:
                # This will raise for str too, but we don't want to change the str as it will not change the underlying
                # object.
                raise ValueError(f"Unknown type {type(config_item)}")

            return need_repass

        def collapse_escape_characters(config_item):
            if isinstance(config_item, dict):
                for key in config_item:
                    config_item[key] = collapse_escape_characters(config_item[key])
            elif isinstance(config_item, list):
                for index in range(len(config_item)):
                    config_item[index] = collapse_escape_characters(config_item[index])
            elif isinstance(config_item, str):
                config_item = config_item.replace("\\[", "[").replace("\\]", "]")
            elif isinstance(config_item, float) or isinstance(config_item, int):
                pass
            else:
                raise ValueError(f"Unknown type {type(config_item)}")
            return config_item

        # Do the replacement until we can't do it anymore.
        while replace_config_values(config):
            pass
        collapse_escape_characters(config)

        # Do a sanity check of the config
        if "model" not in config:
            raise ValueError('Missing "model" in the config file.')
        if "file" not in config["model"]:
            raise ValueError('Missing "file" in the config["model"] file.')
        if not isinstance(config["model"]["file"], str):
            raise ValueError('The "model" in the config file is not a string.')

        if "imus" not in config["model"]:
            raise ValueError('Missing "imus" in the config["model"] file.')
        if not isinstance(config["model"]["imus"], dict):
            raise ValueError('The "imus" in the config file is not a dict.')
        for key, value in config["model"]["imus"].items():
            if not isinstance(key, str):
                raise ValueError('The key in the "imus" in the config file is not a string.')
            if not isinstance(value, str):
                raise ValueError('The value in the "imus" in the config file is not a string.')

        if "data" not in config:
            raise ValueError('Missing "data" in the config file.')

        if "static" not in config["data"]:
            raise ValueError('Missing "static" in the config["data"] file.')
        if not isinstance(config["data"]["static"], dict):
            raise ValueError('The "static" in the config file is not a dict.')

        if "files" not in config["data"]["static"]:
            raise ValueError('Missing "files" in the config["data"]["static"] file.')
        if not isinstance(config["data"]["static"]["files"], dict):
            raise ValueError('The "files" in the config["data"]["static"] file is not a dict of files.')
        for key in config["data"]["static"]["files"]:
            if not isinstance(key, str):
                raise ValueError('The keys in the "files" in the config["data"]["static"] file is not a string.')

        if "header" not in config["data"]["static"]:
            raise ValueError('Missing "header" in the config["data"]["static"] file.')
        if not isinstance(config["data"]["static"]["header"], list):
            raise ValueError('The "header" in the config["data"]["static"] file is not a list of strings.')

        if "frames" not in config["data"]["static"]:
            raise ValueError('Missing "frames" in the config["data"]["static"] file.')
        if not isinstance(config["data"]["static"]["frames"], list):
            raise ValueError('The "frames" in the config["data"]["static"] file is not a list.')
        try:
            config["data"]["static"]["frames"] = np.array(config["data"]["static"]["frames"])
        except ValueError:
            raise ValueError('The "frames" in the config["data"]["static"] file is not a list of ints.')
        if config["data"]["static"]["frames"].dtype != np.int_:
            raise ValueError('The "frames" in the config["data"]["static"] file is not a list of ints.')
        if len(config["data"]["static"]["frames"].shape) == 1:
            config["data"]["static"]["frames"] = config["data"]["static"]["frames"][None, ...]
        if len(config["data"]["static"]["frames"].shape) != 2:
            raise ValueError('The "frames" in the config["data"]["static"] file is not a list of pairs of ints.')
        if config["data"]["static"]["frames"].shape[1] != 2:
            raise ValueError('The "frames" in the config["data"]["static"] file is not a list of pairs of ints.')

        if "trials" not in config["data"]:
            raise ValueError('Missing "trials" in the config["data"] file.')

        if "prefixes" not in config["data"]["trials"]:
            raise ValueError('Missing "prefixes" in the config["data"]["trials"] file.')
        if isinstance(config["data"]["trials"]["prefixes"], str):
            config["data"]["trials"]["prefixes"] = [config["data"]["trials"]["prefixes"]]
        if not isinstance(config["data"]["trials"]["prefixes"], list):
            raise ValueError('The "prefixes" in the config["data"]["trials"] file is not a list.')
        for prefix in config["data"]["trials"]["prefixes"]:
            if not isinstance(prefix, str):
                raise ValueError('The prefix in the "prefixes" in the config["data"]["trials"] file is not a string.')

        if "files" not in config["data"]["trials"]:
            raise ValueError('Missing "files" in the config["data"]["trials"] file.')
        if isinstance(config["data"]["trials"]["files"], dict):
            # It was parsed in a reversed order so we need to reverse it back.
            files = deepcopy(config["data"]["trials"]["files"])
            file_count = None
            for index in files:
                if file_count is None:
                    file_count = len(files[index])
                if file_count != len(files[index]):
                    raise ValueError(
                        'The number of files in the "files" in the config["data"]["trials"] file is not the same for '
                        "all the joints."
                    )
            config["data"]["trials"]["files"] = [
                {key: value[i] for key, value in files.items()} for i in range(file_count)
            ]
        if not isinstance(config["data"]["trials"]["files"], list):
            raise ValueError('The "files" in the config["data"]["trials"] file is not a list of dicts of files.')
        for files in config["data"]["trials"]["files"]:
            if not isinstance(files, dict):
                raise ValueError('The file in the "files" in the config["data"]["trials"] file is not a dict.')
            for key in files:
                if not isinstance(files[key], str):
                    raise ValueError('The file in the "files" in the config["data"]["trials"] file is not a string.')
        if len(config["data"]["trials"]["files"]) != len(config["data"]["trials"]["prefixes"]):
            raise ValueError(
                'The number of prefixes in the "prefixes" in the config["data"]["trials"] file is not the same as the '
                'number of files in the "files" in the config["data"]["trials"] file.'
            )
        config["data"]["trials"]["files"] = {
            prefix: files
            for prefix, files in zip(config["data"]["trials"]["prefixes"], config["data"]["trials"]["files"])
        }

        if "header" not in config["data"]["trials"]:
            raise ValueError('Missing "header" in the config["data"]["trials"] file.')
        if not isinstance(config["data"]["trials"]["header"], list):
            raise ValueError('The "header" in the config["data"]["trials"] file is not a list of strings.')

        if "frames" not in config["data"]["trials"]:
            raise ValueError('Missing "frames" in the config["data"]["trials"] file.')
        if not isinstance(config["data"]["trials"]["frames"], dict):
            raise ValueError('The "frames" in the config["data"]["trials"] file is not a dict.')
        for key, value in config["data"]["trials"]["frames"].items():
            if not isinstance(key, str):
                raise ValueError('The key in the "frames" in the config["data"]["trials"] file is not a string.')
            if not isinstance(value, list):
                raise ValueError('The value in the "frames" in the config["data"]["trials"] file is not a list.')
            try:
                config["data"]["trials"]["frames"][key] = np.array(value)
            except ValueError:
                raise ValueError(
                    'The value in the "frames" in the config["data"]["trials"] file is not a list of ints.'
                )
            if config["data"]["trials"]["frames"][key].dtype != np.int_:
                raise ValueError(
                    'The value in the "frames" in the config["data"]["trials"] file is not a list of ints.'
                )
            if len(config["data"]["trials"]["frames"][key].shape) == 1:
                config["data"]["trials"]["frames"][key] = config["data"]["trials"]["frames"][key][None, ...]
            if len(config["data"]["trials"]["frames"][key].shape) != 2:
                raise ValueError(
                    'The value in the "frames" in the config["data"]["trials"] file is not a list of pairs of ints.'
                )
            if config["data"]["trials"]["frames"][key].shape[1] != 2:
                raise ValueError(
                    'The value in the "frames" in the config["data"]["trials"] file is not a list of pairs of ints.'
                )

        # Get all the files in the subject folder
        subject_folder = Path(subject_folder)
        static_trial_config = TrialConfig(
            files={
                key: _collapse_filepath(base_folder=subject_folder, filepath=file)
                for key, file in config["data"]["static"]["files"].items()
            },
            header=config["data"]["static"]["header"],
            frames=config["data"]["static"]["frames"],
        )

        # Get all the trial files in the subject folder
        trial_configs = {
            prefix: TrialConfig(
                files={
                    key: _collapse_filepath(base_folder=subject_folder, filepath=file)
                    for key, file in config["data"]["trials"]["files"][prefix].items()
                },
                header=config["data"]["trials"]["header"],
                frames=config["data"]["trials"]["frames"][prefix],
            )
            for prefix in config["data"]["trials"]["prefixes"]
        }

        # Dispatch the config dict into the Config object
        return Config(
            model_filepath=config["model"]["file"],
            imus=ImuConfig(names=list(config["model"]["imus"].keys())),
            static=static_trial_config,
            trials=trial_configs,
        )


def _collapse_filepath(base_folder: str, filepath: str) -> str:
    filepath: Path = Path(filepath)
    filepaths: list[Path] = list(Path(base_folder).glob(filepath.as_posix()))
    if len(filepaths) == 0:
        raise ValueError(f"Could not find the file {filepath.as_posix()} in the folder {base_folder}")
    elif len(filepaths) > 1:
        raise ValueError(f"Found multiple files {filepath.as_posix()} in the folder {base_folder}")
    return filepaths[0].resolve().as_posix()
