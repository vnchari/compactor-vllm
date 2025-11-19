from collections.abc import Callable

import torch


def maybe_execute_in_stream(
    fn: Callable, *args, STORE_STREAM: torch.cuda.Stream = None, **kwargs
):
    if STORE_STREAM is not None:
        tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
        tensors += [val for val in kwargs.values() if isinstance(val, torch.Tensor)]
        obj = getattr(fn, "__self__", None)
        if isinstance(obj, torch.Tensor):
            tensors.append(obj)
        STORE_STREAM.wait_stream(torch.cuda.default_stream())
        with STORE_STREAM:
            output = fn(*args, **kwargs)
        for t in tensors:
            t.record_stream(STORE_STREAM)
        if isinstance(output, tuple):
            for o in output:
                if isinstance(o, torch.Tensor):
                    o.record_stream(torch.cuda.default_stream())
        elif isinstance(output, torch.Tensor):
            output.record_stream(torch.cuda.default_stream())
        return output
    else:
        return fn(*args, **kwargs)
