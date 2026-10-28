from scripts.leading_silence.publish import _configs, checkpoint_name, stale_vectors
from scripts.leading_silence.weights import weights_sha256


def test_vectors_are_named_after_the_checkpoint_not_the_language_alias():
    assert checkpoint_name(["english", "english_2026-10"]) == "english_2026-10"
    assert checkpoint_name(["french"]) == "french"
    assert checkpoint_name(["english_2026-09", "english_2026-09_2"]) == "english_2026-09"


def test_every_shipped_vectors_file_fits_its_configs_weights():
    # Upstream moves aliases such as `english` to new weights: a config left pointing at the
    # old file would load vectors trained on another model.
    configs = {
        name: config for name, config in _configs().items() if config.get("leading_silence_vectors")
    }
    assert configs
    for name, config in configs.items():
        reason = stale_vectors(config, weights_sha256(config["weights_path"]))
        assert reason is None, f"{name}: {reason}"
