import yaml


def load_config(config_file=None, default_file="configs/default.yaml"):
    with open(default_file, "r") as f:
        config = yaml.safe_load(f)

    if config_file is not None:
        with open(config_file, "r") as f:
            user_config = yaml.safe_load(f)
        config.update(user_config)

    return config
