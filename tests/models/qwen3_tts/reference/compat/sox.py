"""Import guard for the unused upstream 25 Hz tokenizer dependency.

The official package imports its 25 Hz tokenizer modules while loading the
12 Hz model. Nixpkgs does not package the Python ``sox`` wrapper for Python
3.13, and the selected 12 Hz CustomVoice path never instantiates it.
"""


class Transformer:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "the Python sox wrapper is unavailable in the 12 Hz oracle shell"
        )
