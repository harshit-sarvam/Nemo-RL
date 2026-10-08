# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import tempfile
import time
import warnings
from typing import Any, Callable, Union

import numpy as np
from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class PreservingDataset:
    """A dataset wrapper that preserves original dict structure without None-filling.

    Unlike HuggingFace's Dataset class which enforces schema uniformity across all samples
    (filling missing keys with None), this class maintains the exact structure of each sample.
    This is critical for heterogeneous data like tool calls where different samples may have
    different argument structures.
    """

    def __init__(self, data: list[dict[str, Any]]):
        """Initialize the dataset with a list of dictionaries.

        Args:
            data: List of dictionary samples, each can have different keys
        """
        self.data = data
        self.features = None  # For compatibility with HF Dataset interface

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: Union[int, slice, list]
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        """Support integer indexing, slicing, and list indexing."""
        if isinstance(idx, slice):
            return [self.data[i] for i in range(*idx.indices(len(self.data)))]
        elif isinstance(idx, int):
            # Handle negative indices
            if idx < 0:
                idx = len(self.data) + idx
            if idx < 0 or idx >= len(self.data):
                raise IndexError(
                    f"Index {idx} out of range for dataset of size {len(self.data)}"
                )
            return self.data[idx]
        elif isinstance(idx, list):
            return [self.data[i] for i in idx]
        else:
            raise TypeError(
                f"Indices must be integers, slices, or lists, not {type(idx)}"
            )

    def __iter__(self):
        return iter(self.data)

    def map(self, function: Callable, *args, **kwargs) -> "PreservingDataset":
        """Apply a function to each sample in the dataset.

        Args:
            function: Function to apply to each sample
            with_indices: If True, pass index as second argument to function

        Returns:
            New PreservingDataset with transformed samples
        """
        if kwargs.get("with_indices", False):
            mapped_data = [function(item, i) for i, item in enumerate(self.data)]
        else:
            mapped_data = [function(item) for item in self.data]
        return PreservingDataset(mapped_data)


def _print_scan_progress(
    pos: int, file_size: int, n_lines: int, start: float
) -> None:
    rate = pos / max(time.perf_counter() - start, 1e-9)
    print(
        f"  indexed {pos / (1 << 30):7.1f} / {file_size / (1 << 30):.1f} GiB"
        f" ({100 * pos / max(file_size, 1):5.1f}%)"
        f" | {rate / (1 << 20):5.0f} MiB/s"
        f" | {n_lines:,} lines"
        f" | ETA {(file_size - pos) / rate / 60:5.1f} min",
        flush=True,
    )


def _build_offset_index(data_path: str) -> np.ndarray:
    """Scan the file once and record the byte offset of every non-blank line.

    Pure I/O plus newline scanning; no JSON parsing, so it runs at storage
    speed. Prints progress every 8 GiB scanned.
    """
    file_size = os.path.getsize(data_path)
    start = time.perf_counter()
    next_report = report_every = 8 << 30
    offsets = []
    pos = 0
    with open(data_path, "rb", buffering=1 << 24) as f:
        for line in f:
            if line.strip():
                offsets.append(pos)
            pos += len(line)
            if pos >= next_report:
                next_report += report_every
                _print_scan_progress(pos, file_size, len(offsets), start)
    _print_scan_progress(pos, file_size, len(offsets), start)
    return np.asarray(offsets, dtype=np.int64)


def _load_or_build_offset_index(data_path: str) -> np.ndarray:
    """Return the line-offset index, using a cache file next to the data.

    The cache is keyed by file size + mtime and rebuilt automatically when
    the data file changes. Cache write failures (e.g. read-only directory)
    degrade to rebuilding the index on every run.
    """
    stat = os.stat(data_path)
    cache_path = data_path + ".offsets.npz"

    try:
        with np.load(cache_path) as cache:
            if (
                int(cache["file_size"]) == stat.st_size
                and int(cache["mtime_ns"]) == stat.st_mtime_ns
            ):
                print(
                    f"Loaded line-offset index from cache: {cache_path}", flush=True
                )
                return cache["offsets"]
        print(
            f"Offset cache is stale (data file changed), rebuilding: {cache_path}",
            flush=True,
        )
    except FileNotFoundError:
        pass
    except Exception as e:  # corrupt/unreadable cache: rebuild from scratch
        warnings.warn(
            f"Could not read offset cache {cache_path} ({e!r}); rebuilding.",
            stacklevel=2,
        )

    print(
        f"Building line-offset index for {data_path} (one-time sequential scan)...",
        flush=True,
    )
    offsets = _build_offset_index(data_path)

    try:
        # Write atomically so a killed run or a concurrent job never leaves a
        # partial cache behind.
        fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(cache_path), suffix=".npz.tmp"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(
                    f,
                    offsets=offsets,
                    file_size=np.int64(stat.st_size),
                    mtime_ns=np.int64(stat.st_mtime_ns),
                )
            os.replace(tmp_path, cache_path)
        except BaseException:
            os.unlink(tmp_path)
            raise
        print(f"Saved line-offset index to {cache_path}", flush=True)
    except OSError as e:
        warnings.warn(
            f"Could not write offset cache {cache_path} ({e!r}); "
            "the index will be rebuilt on the next run.",
            stacklevel=2,
        )

    return offsets


class LazyJSONLDataset:
    """Map-style dataset that parses JSONL lines on access instead of upfront.

    Only a numpy array of line-start byte offsets (8 bytes per sample) is held
    in memory; ``__getitem__`` seeks to the line and parses it on demand. With
    a torch DataLoader, parsing therefore runs inside the DataLoader worker
    processes, overlapped with training, instead of stalling startup while a
    single core parses the whole file into RAM.

    Like ``PreservingDataset``, samples keep their exact structure with no
    schema unification or None-filling.
    """

    def __init__(self, data_path: str, format_fn: Callable[[dict], dict]):
        self.data_path = os.path.abspath(data_path)
        self.format_fn = format_fn
        self.offsets = _load_or_build_offset_index(self.data_path)
        self.features = None  # For compatibility with HF Dataset interface
        # One file handle per process: DataLoader workers each get their own,
        # so seek positions never collide.
        self._fh = None
        self._fh_pid = None

    def _file(self):
        if self._fh is None or self._fh_pid != os.getpid():
            self._fh = open(self.data_path, "rb")
            self._fh_pid = os.getpid()
        return self._fh

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(
        self, idx: Union[int, slice, list]
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        elif isinstance(idx, list):
            return [self[i] for i in idx]
        elif isinstance(idx, (int, np.integer)):
            n = len(self.offsets)
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise IndexError(f"Index {idx} out of range for dataset of size {n}")
            f = self._file()
            f.seek(int(self.offsets[idx]))
            return self.format_fn(json.loads(f.readline()))
        else:
            raise TypeError(
                f"Indices must be integers, slices, or lists, not {type(idx)}"
            )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getstate__(self):
        # File handles can't be pickled (and must not be shared across
        # processes); each process reopens lazily via _file().
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_fh_pid"] = None
        return state


class OpenAIFormatDataset(RawDataset):
    """This class is used to load an SFT dataset in the OpenAI format.

    The dataset should be in the following format:
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
    }
    Please refer to https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/sft.md#openai-format-datasets-with-tool-calling-support for more details.

    Args:
        data_path: Path to the dataset JSON file
        chat_key: Key for the messages list in the dataset (default: "messages")
        system_key: Optional key for system prompt in the dataset
        system_prompt: Optional system prompt to add if not in the dataset
        tool_key: Key for tools in the dataset (default: "tools")
        use_preserving_dataset: If True, uses PreservingDataset to maintain
            heterogeneous schemas (e.g., for tool calls with varying argument
            structures). If False, uses standard HuggingFace dataset loading.
            Default is False for backward compatibility.
        use_lazy_dataset: If True, uses LazyJSONLDataset: samples are parsed
            on access (inside DataLoader workers) instead of upfront, using a
            cached byte-offset index. Recommended for very large JSONL files
            where upfront parsing is slow and memory-hungry. Like
            PreservingDataset, it preserves heterogeneous schemas. Takes
            precedence over use_preserving_dataset. Requires data_path to be
            a local JSONL file.

    Notes:
        - system_key and system_prompt are optional. If provided, it will be added
          to the beginning of the dataset.
        - chat_key should be the key of the messages list. Multi-turn conversations
          are supported.
        - The last message in the conversation must be from the assistant.
        - When use_preserving_dataset=True, the dataset preserves the exact structure
          of each sample without None-filling for missing keys, which is useful for
          heterogeneous tool argument schemas.
    """

    def __init__(
        self,
        data_path: str,
        chat_key: str = "messages",
        system_key: str | None = None,
        system_prompt: str | None = None,
        tool_key: str | None = "tools",
        use_preserving_dataset: bool = False,
        use_lazy_dataset: bool = False,
        **kwargs,
    ):
        self.chat_key = chat_key
        self.system_key = system_key
        self.system_prompt = system_prompt
        self.tool_key = tool_key

        self.task_name = "-".join(data_path.split("/")[-2:]).split(".")[0]
        if self.task_name[0] == "-":
            self.task_name = self.task_name[1:]

        if use_lazy_dataset:
            self.dataset = LazyJSONLDataset(data_path, self.format_data)
            print(
                f"Loaded dataset lazily (parsed on access): {len(self.dataset)} samples.",
                flush=True,
            )
        elif not use_preserving_dataset:
            # Use the standard HuggingFace approach (faster and more standard)
            original_dataset = load_dataset("json", data_files=data_path)["train"]
            # Format the dataset
            self.dataset = original_dataset.map(self.format_data)

            print(
                f"Loaded dataset using standard approach: {len(self.dataset)} samples."
            )

            # Warn if tools are present in the dataset
            if self.tool_key and any(
                self.tool_key in sample for sample in self.dataset
            ):
                warnings.warn(
                    "Tools detected in dataset. Set use_preserving_dataset=True to preserve heterogeneous tool schemas. "
                    "Current mode may add None values for missing tool arguments, making samples invalid.",
                    UserWarning,
                    stacklevel=2,
                )

        else:
            # Use custom loading for heterogeneous schemas
            # Issue: When tool calls have varying argument structures across samples,
            # HuggingFace's Dataset.from_list enforces uniform schema by adding None
            # values for missing keys. Example:
            #   Sample 1: {"tools": [{"name": "search", "args": {"query": "x"}}]}
            #   Sample 2: {"tools": [{"name": "calc", "args": {"expr": "y", "precision": 2}}]}
            # Standard loading would add "precision: None" to Sample 1 and "query: None" to Sample 2.
            # PreservingDataset maintains exact structure without None-filling.
            print(
                "Using PreservingDataset to preserve heterogeneous tool argument schemas without None-filling."
            )

            # Load JSON files directly
            with open(data_path, "r") as f:
                original_dataset = [json.loads(line) for line in f]
            # Format the dataset
            formatted_data = [self.format_data(item) for item in original_dataset]
            # Use PreservingDataset to maintain exact structure
            self.dataset = PreservingDataset(formatted_data)

            print(
                f"Loaded dataset using PreservingDataset: {len(self.dataset)} samples."
            )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        messages = [message for message in data[self.chat_key]]
        if self.system_key is not None and self.system_key in data:
            messages = [{"role": "system", "content": data[self.system_key]}] + messages
        elif self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        assert messages[-1]["role"] == "assistant"

        # Preserve tools if they exist in the data
        result = {"messages": messages, "task_name": self.task_name}
        if self.tool_key and self.tool_key in data:
            result["tools"] = data[self.tool_key]

        return result
