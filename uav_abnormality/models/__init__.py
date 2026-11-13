def build_model(name, n_sources, n_features, window_size, config=None):
    config = dict(config or {})
    if name == "prbtd":
        from .prbtd import PRBTD
        return PRBTD(n_sources, n_features, window_size, **config)
    if name in ("tsadc", "tsad-c"):
        from .tsadc import TSADC
        return TSADC(n_sources, n_features, window_size, **config)
    raise ValueError("Unknown algorithm: %s" % name)
