from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import numpy.typing as npt


@dataclass
class Config:
    model_filename: str
    imus: dict[str, str]

    static_files: list[str]
    static_frames: npt.NDArray[np.int_]

    trials_files: dict[str, list[str]]
    trials_frames: dict[str, npt.NDArray[np.int_]]

    @staticmethod
    def from_config_file(subject_folder: str) -> "Config":
        # Load the content of the file
        config = json.loads((Path(subject_folder) / "conf.json").read_text())

        # Read down the dict and replace any [value] with the value in the dict.
        def expand_str(item) -> tuple[list[str], bool]:
            # First change the current key into a list. This is useful to deal with (*) elements.
            current_expand = [item]

            # Find all the [...]
            replacable_tag = r"\[.*?\]"
            loops = re.findall(replacable_tag, item)
            for loop in set(loops):
                value = config  # Search the value to replace the element with
                for element in loop[1:-1].split("."):
                    if element == "keys(*)":
                        value = list(value.keys())
                    elif element == "values(*)":
                        value = list(value.values())
                    elif element.endswith("(*)"):
                        value = value[element[:-3]]
                    else:
                        value = value[element]

                if isinstance(value, list):
                    tp = []
                    for i in range(len(current_expand)):
                        for v in value:
                            tp.append(current_expand[i].replace(loop, v))
                    current_expand = tp
                elif isinstance(value, str):
                    for i in range(len(current_expand)):
                        current_expand[i] = current_expand[i].replace(loop, value)
                else:
                    raise ValueError(f"Unknown type {type(value)}")

            could_fully_expand = True
            for val in current_expand:
                if re.search(replacable_tag, val):
                    could_fully_expand = False
                    break
            return ((current_expand[0] if len(current_expand) == 1 else current_expand), could_fully_expand)

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

        # Do the replacement until we can't do it anymore.
        while replace_config_values(config):
            pass

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
        if not isinstance(config["data"]["static"]["files"], list):
            raise ValueError('The "files" in the config["data"]["static"] file is not a string.')
        for file in config["data"]["static"]["files"]:
            if not isinstance(file, str):
                raise ValueError('The file in the "files" in the config["data"]["static"] file is not a string.')

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
        if not isinstance(config["data"]["trials"]["prefixes"], list):
            raise ValueError('The "prefixes" in the config["data"]["trials"] file is not a list.')
        for prefix in config["data"]["trials"]["prefixes"]:
            if not isinstance(prefix, str):
                raise ValueError('The prefix in the "prefixes" in the config["data"]["trials"] file is not a string.')

        if "files" not in config["data"]["trials"]:
            raise ValueError('Missing "files" in the config["data"]["trials"] file.')
        if not isinstance(config["data"]["trials"]["files"], list):
            raise ValueError('The "files" in the config["data"]["trials"] file is not a string.')
        for file in config["data"]["trials"]["files"]:
            if not isinstance(file, str):
                raise ValueError('The file in the "files" in the config["data"]["trials"] file is not a string.')
        dispatched_trials_files = {}
        for prefix in config["data"]["trials"]["prefixes"]:
            dispatched_trials_files[prefix] = []
            for file in config["data"]["trials"]["files"]:
                if re.match(f"^{prefix}.*\\/.*$", file):
                    dispatched_trials_files[prefix].append(file)
        config["data"]["trials"]["files"] = dispatched_trials_files

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

        # Dispatch the config dict into the Config object
        return Config(
            model_filename=config["model"]["file"],
            imus=config["model"]["imus"],
            static_files=config["data"]["static"]["files"],
            static_frames=config["data"]["static"]["frames"],
            trials_files=config["data"]["trials"]["files"],
            trials_frames=config["data"]["trials"]["frames"],
        )
