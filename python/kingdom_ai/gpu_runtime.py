"""Compile small CUDA kernels with NVRTC and launch on PyTorch's stream.

No nvcc, C++ host compiler, extension build, or third-party CUDA Python package
is needed. PyTorch owns the context, stream and tensor allocations. The driver
receives pointers only to caller-owned CUDA tensors, never CPU tensor storage.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from functools import lru_cache
import os
from pathlib import Path
from threading import Lock

import torch


_COMPILE_LOCK = Lock()
_DLL_DIRECTORIES = []


def cuda_device(device="cuda") -> torch.device:
    """Resolve a CUDA index and reject an unavailable or CPU backend."""
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("GPU rules and PUCT require a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install CUDA-enabled PyTorch and a compatible NVIDIA driver")
    index = torch.cuda.current_device() if resolved.index is None else resolved.index
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError("CUDA device index is unavailable")
    return torch.device("cuda", index)


def _bind(library, name, arguments, result=ctypes.c_int):
    function = getattr(library, name)
    function.argtypes = arguments
    function.restype = result
    return function


def _nvrtc_library():
    root = Path(torch.__file__).resolve().parent
    directories = [root / "lib", root.parent / "nvidia" / "cuda_nvrtc" / "lib"]
    if os.environ.get("CUDA_PATH"):
        toolkit = Path(os.environ["CUDA_PATH"])
        directories += [toolkit / "bin", toolkit / "lib64"]
    pattern = "nvrtc64_*.dll" if os.name == "nt" else "libnvrtc.so*"
    for directory in directories:
        candidates = sorted(directory.glob(pattern), reverse=True)
        for path in candidates:
            if ".alt." in path.name or "builtins" in path.name:
                continue
            if os.name == "nt":
                # NVRTC loads its builtins DLL lazily while compiling.
                _DLL_DIRECTORIES.append(os.add_dll_directory(str(directory)))
            return ctypes.CDLL(str(path))
    located = ctypes.util.find_library("nvrtc")
    if located:
        return ctypes.CDLL(located)
    raise RuntimeError("NVRTC was not found in PyTorch's CUDA libraries or CUDA_PATH")


class _Libraries:
    def __init__(self):
        self.nvrtc = _nvrtc_library()
        self.driver = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        pointer = ctypes.c_void_p
        int_pointer = ctypes.POINTER(ctypes.c_int)
        size_pointer = ctypes.POINTER(ctypes.c_size_t)
        pointer_pointer = ctypes.POINTER(pointer)
        char_pointers = ctypes.POINTER(ctypes.c_char_p)
        _bind(self.nvrtc, "nvrtcVersion", [int_pointer, int_pointer])
        _bind(self.nvrtc, "nvrtcGetErrorString", [ctypes.c_int], ctypes.c_char_p)
        _bind(self.nvrtc, "nvrtcCreateProgram", [pointer_pointer, ctypes.c_char_p, ctypes.c_char_p,
                                               ctypes.c_int, char_pointers, char_pointers])
        _bind(self.nvrtc, "nvrtcCompileProgram", [pointer, ctypes.c_int, char_pointers])
        _bind(self.nvrtc, "nvrtcGetProgramLogSize", [pointer, size_pointer])
        _bind(self.nvrtc, "nvrtcGetProgramLog", [pointer, pointer])
        _bind(self.nvrtc, "nvrtcGetPTXSize", [pointer, size_pointer])
        _bind(self.nvrtc, "nvrtcGetPTX", [pointer, pointer])
        _bind(self.nvrtc, "nvrtcDestroyProgram", [pointer_pointer])
        _bind(self.driver, "cuInit", [ctypes.c_uint])
        _bind(self.driver, "cuGetErrorName", [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)])
        _bind(self.driver, "cuGetErrorString", [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)])
        _bind(self.driver, "cuModuleLoadData", [pointer_pointer, pointer])
        _bind(self.driver, "cuModuleUnload", [pointer])
        _bind(self.driver, "cuModuleGetFunction", [pointer_pointer, pointer, ctypes.c_char_p])
        _bind(self.driver, "cuLaunchKernel", [pointer, *([ctypes.c_uint] * 7), pointer,
                                              pointer_pointer, pointer_pointer])
        self.check_driver(self.driver.cuInit(0), "cuInit")

    def check_nvrtc(self, code, operation):
        if code:
            reason = self.nvrtc.nvrtcGetErrorString(code).decode("utf-8", "replace")
            raise RuntimeError(f"{operation} failed: {reason}")

    def check_driver(self, code, operation):
        if code:
            name, reason = ctypes.c_char_p(), ctypes.c_char_p()
            self.driver.cuGetErrorName(code, ctypes.byref(name))
            self.driver.cuGetErrorString(code, ctypes.byref(reason))
            detail = (reason.value or b"unknown CUDA driver error").decode("utf-8", "replace")
            label = (name.value or str(code).encode()).decode("utf-8", "replace")
            raise RuntimeError(f"{operation} failed: {label}: {detail}")


@lru_cache(maxsize=1)
def _libraries():
    return _Libraries()


@lru_cache(maxsize=8)
def _compile(source: str, architecture: str) -> bytes:
    library = _libraries()
    program = ctypes.c_void_p()
    with _COMPILE_LOCK:
        library.check_nvrtc(library.nvrtc.nvrtcCreateProgram(
            ctypes.byref(program), source.encode("utf-8"), b"kingdom_gpu.cu", 0, None, None
        ), "nvrtcCreateProgram")
        try:
            options = (ctypes.c_char_p * 3)(
                b"--std=c++14", f"--gpu-architecture={architecture}".encode(), b"--fmad=false"
            )
            code = library.nvrtc.nvrtcCompileProgram(program, len(options), options)
            if code:
                length = ctypes.c_size_t()
                library.check_nvrtc(library.nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(length)),
                                    "nvrtcGetProgramLogSize")
                log = ctypes.create_string_buffer(length.value)
                library.check_nvrtc(library.nvrtc.nvrtcGetProgramLog(program, log), "nvrtcGetProgramLog")
                raise RuntimeError("CUDA kernel compilation failed:\n" + log.value.decode("utf-8", "replace"))
            length = ctypes.c_size_t()
            library.check_nvrtc(library.nvrtc.nvrtcGetPTXSize(program, ctypes.byref(length)), "nvrtcGetPTXSize")
            ptx = ctypes.create_string_buffer(length.value)
            library.check_nvrtc(library.nvrtc.nvrtcGetPTX(program, ptx), "nvrtcGetPTX")
            return ptx.raw
        finally:
            library.nvrtc.nvrtcDestroyProgram(ctypes.byref(program))


def nvrtc_version() -> tuple[int, int]:
    major, minor = ctypes.c_int(), ctypes.c_int()
    library = _libraries()
    library.check_nvrtc(library.nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)), "nvrtcVersion")
    return major.value, minor.value


class CudaModule:
    """An NVRTC module loaded in PyTorch's existing CUDA primary context.

    Arguments are CUDA tensors (device pointers), Python int (int32), Python
    float (float32), or explicit ctypes scalar values. Kernels run on the
    current PyTorch stream, preserving ordering with torch operators. A tensor
    used from another stream must follow PyTorch's wait_stream rules.
    """

    def __init__(self, source: str, device="cuda"):
        if not isinstance(source, str) or not source.strip():
            raise ValueError("CUDA source must be a non-empty string")
        self.device = cuda_device(device)
        self._module = ctypes.c_void_p()
        self._functions = {}
        self._library = _libraries()
        with torch.cuda.device(self.device):
            torch.cuda.init()
            # Lazy initialization alone need not activate a driver context.
            # A PyTorch allocation binds its primary context on this thread.
            torch.empty(1, device=self.device)
            capability = torch.cuda.get_device_capability(self.device)
            self.architecture = "compute_%d%d" % capability
            ptx = ctypes.create_string_buffer(_compile(source, self.architecture))
            self._library.check_driver(self._library.driver.cuModuleLoadData(
                ctypes.byref(self._module), ctypes.cast(ptx, ctypes.c_void_p)
            ), "cuModuleLoadData")

    def launch(self, name, *, grid, block, args, shared_memory=0):
        if not self._module.value:
            raise RuntimeError("CUDA module is closed")
        for dimensions in (grid, block):
            if (len(dimensions) != 3 or any(type(value) is not int or value < 1 for value in dimensions)):
                raise ValueError("grid and block must each contain three positive integers")
        if type(shared_memory) is not int or shared_memory < 0:
            raise ValueError("shared_memory must be a non-negative integer")
        values = []
        for argument in args:
            if isinstance(argument, torch.Tensor):
                if argument.device != self.device:
                    raise ValueError("Kernel tensors must be on the module's CUDA device")
                values.append(ctypes.c_void_p(argument.data_ptr()))
            elif type(argument) is int:
                if not -(2 ** 31) <= argument < 2 ** 31:
                    raise ValueError("Integer kernel arguments must fit int32")
                values.append(ctypes.c_int(argument))
            elif type(argument) is float:
                values.append(ctypes.c_float(argument))
            elif isinstance(argument, ctypes._SimpleCData):
                values.append(argument)
            else:
                raise TypeError("Kernel arguments must be CUDA tensors or scalars")
        parameters = (ctypes.c_void_p * len(values))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values)
        )
        with torch.cuda.device(self.device):
            if name not in self._functions:
                function = ctypes.c_void_p()
                self._library.check_driver(self._library.driver.cuModuleGetFunction(
                    ctypes.byref(function), self._module, name.encode("ascii")
                ), "cuModuleGetFunction")
                self._functions[name] = function
            stream = torch.cuda.current_stream(self.device)
            self._library.check_driver(self._library.driver.cuLaunchKernel(
                self._functions[name], *grid, *block, shared_memory,
                ctypes.c_void_p(stream.cuda_stream), parameters, None
            ), f"cuLaunchKernel({name})")
            # Tell the allocator about pointers consumed asynchronously here.
            for argument in args:
                if isinstance(argument, torch.Tensor):
                    argument.record_stream(stream)

    def close(self):
        if self._module.value:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                self._library.check_driver(self._library.driver.cuModuleUnload(self._module), "cuModuleUnload")
            self._module = ctypes.c_void_p()

    def __del__(self):
        # Interpreter shutdown may already have torn down CUDA or ctypes.
        try:
            self.close()
        except Exception:
            pass
