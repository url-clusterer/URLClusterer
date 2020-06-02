import yaml


class ConfigLoader:
    def __init__(self):
        with open("config.yml") as stream:
            self.__configs = yaml.safe_load(stream)

    def get_configs(self):
        return self.__configs

    configs = None


def get_configs():
    if ConfigLoader.configs is None:
        ConfigLoader.configs = ConfigLoader().get_configs()
    return ConfigLoader.configs
