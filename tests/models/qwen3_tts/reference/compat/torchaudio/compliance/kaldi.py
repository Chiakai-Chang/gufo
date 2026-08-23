"""Fail-fast guard for the unused upstream 25 Hz Kaldi frontend."""


def fbank(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "torchaudio Kaldi features are unavailable in the 12 Hz oracle shell"
    )
