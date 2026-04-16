import sys


def _patch_torch_load_for_cpu():
    """Force CPU mapping and disable weights_only so CUDA-saved checkpoints
    (e.g. omnipose models) load on CPU-only machines. Must run before any
    cellpose/omnipose import. Overrides caller kwargs — setdefault is not
    enough because cellpose passes weights_only=True explicitly."""
    try:
        import torch
    except ImportError:
        return

    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        if not torch.cuda.is_available():
            kwargs["map_location"] = torch.device("cpu")
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load


_patch_torch_load_for_cpu()

# Force UTF-8 stdout/stderr on Windows so emoji prints don't crash on cp1252.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
