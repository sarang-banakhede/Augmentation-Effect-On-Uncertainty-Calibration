import yaml


class Config(dict):
    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value


def load_config(path):
    with open(path) as handle:
        return Config(yaml.safe_load(handle))


def to_plain(obj):
    if isinstance(obj, dict):
        return {key: to_plain(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(value) for value in obj]
    return obj


def save_config(cfg, path):
    with open(path, "w") as handle:
        yaml.safe_dump(to_plain(cfg), handle, sort_keys=False)


def apply_overrides(cfg, strategies=None, seeds=None, output_root=None):
    if strategies:
        cfg["experiment"]["strategies"] = list(strategies)
    if seeds:
        cfg["experiment"]["seeds"] = [int(s) for s in seeds]
    if output_root:
        cfg["paths"]["output_root"] = str(output_root)
    return cfg
