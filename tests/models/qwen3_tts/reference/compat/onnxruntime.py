"""Import guard for the unused upstream 25 Hz x-vector dependency."""


class GraphOptimizationLevel:
    ORT_ENABLE_ALL = object()


class SessionOptions:
    def __init__(self):
        self.graph_optimization_level = None
        self.intra_op_num_threads = 0


class InferenceSession:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "ONNX Runtime is unavailable in the 12 Hz Qwen3-TTS oracle shell"
        )
