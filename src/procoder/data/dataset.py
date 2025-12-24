from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from procoder.utils.structure_parser import parse_structure


class BaseTokenizedDataset:
    """
    Mixin providing shared row parsing, padding, and optional coordinates handling.
    """

    @staticmethod
    def _build_output_from_row(
        row: pd.Series,
        *,
        max_length: int,
        has_coords: bool,
        require_indices: bool = False,
    ) -> dict[str, torch.Tensor | str]:
        """Build output dictionary from a dataset row.

        Args:
            row: DataFrame row containing protein data.
            max_length: Maximum sequence length for padding.
            has_coords: Whether to parse coordinates from the row.
            require_indices: Whether to parse indices column. Defaults to False for MLM.

        Returns:
            Dictionary with pid, seq, and optionally coords.
        """
        pid = row["pid"]
        seq = row["protein_sequence"]

        out: dict[str, torch.Tensor | str] = {
            "pid": pid,
            "seq": seq,
        }

        # parse optional 3D-coordinates only from parquet inputs
        if has_coords:
            raw_coords = row.get("coordinates")
            coords_arr: Optional[np.ndarray]
            # Lazy import to avoid unnecessary dependency at module import time
            import ast as _ast  # type: ignore

            if isinstance(raw_coords, np.ndarray):
                # parquet nested lists can round-trip as object arrays
                if raw_coords.dtype == object:
                    # Deep-normalize two levels to handle arrays-of-arrays objects
                    try:
                        outer = raw_coords.tolist()
                        rows = []
                        for elem in outer:
                            sub = elem.tolist() if hasattr(elem, "tolist") else elem
                            rows.append(
                                [
                                    list(x)
                                    if hasattr(x, "__iter__")
                                    and not isinstance(x, (str, bytes))
                                    else x
                                    for x in sub
                                ]
                            )
                        coords_arr = np.asarray(rows, dtype=np.float32)
                    except Exception:
                        coords_arr = None
                else:
                    coords_arr = raw_coords.astype(np.float32, copy=False)
            elif isinstance(raw_coords, (list, tuple)):
                # Handle list-of-arrays or list-of-lists
                try:
                    rows = []
                    for elem in list(raw_coords):
                        sub = elem.tolist() if hasattr(elem, "tolist") else elem
                        rows.append(
                            [
                                list(x)
                                if hasattr(x, "__iter__")
                                and not isinstance(x, (str, bytes))
                                else x
                                for x in sub
                            ]
                        )
                    coords_arr = np.asarray(rows, dtype=np.float32)
                except Exception:
                    try:
                        coords_arr = np.asarray(raw_coords, dtype=np.float32)
                    except Exception:
                        coords_arr = None
            elif isinstance(raw_coords, float) and pd.isna(raw_coords):
                coords_arr = None
            else:
                coords_arr = None

            # Final generic fallback conversion
            if coords_arr is None and raw_coords is not None:
                obj = (
                    raw_coords.tolist() if hasattr(raw_coords, "tolist") else raw_coords
                )
                try:
                    coords_arr = np.asarray(obj, dtype=np.float32)
                except Exception:
                    try:
                        coords_arr = np.stack(
                            [np.asarray(x, dtype=np.float32) for x in list(obj)],
                            axis=0,
                        )
                    except Exception:
                        coords_arr = None
                # Stringified list fallback (e.g., if serialized as text)
                if (
                    coords_arr is None
                    and isinstance(obj, str)
                    and obj.strip().startswith("[")
                ):
                    try:
                        parsed = _ast.literal_eval(obj)
                        coords_arr = np.asarray(parsed, dtype=np.float32)
                    except Exception:
                        coords_arr = None

            # Coerce to shape [L, 3, 3] if possible
            if coords_arr is not None:
                if coords_arr.ndim == 3 and coords_arr.shape[-2:] == (3, 3):
                    pass
                elif coords_arr.ndim == 3 and coords_arr.shape[:2] == (3, 3):
                    coords_arr = np.transpose(coords_arr, (2, 0, 1))
                elif coords_arr.ndim == 2 and coords_arr.shape == (3, 3):
                    coords_arr = coords_arr[None, ...]
                elif coords_arr.ndim == 2 and (coords_arr.size % 9 == 0):
                    coords_arr = coords_arr.reshape(-1, 3, 3)
                else:
                    coords_arr = None

            if coords_arr is None:
                coords_arr = np.empty((0, 3, 3), dtype=np.float32)

            Lc = int(coords_arr.shape[0])
            copy_len = min(Lc, max_length)
            coords_padded = np.full((max_length, 3, 3), np.nan, dtype=np.float32)
            if copy_len > 0:
                coords_padded[:copy_len] = coords_arr[:copy_len]
            out["coords"] = torch.tensor(coords_padded, dtype=torch.float32)

        return out


class TokenizedDataset(Dataset, BaseTokenizedDataset):
    """Dataset for loading protein sequences from CSV or Parquet files.

    Supports optional coordinate loading from Parquet files for P@L evaluation.

    Args:
        dataset_path: Path to CSV/TSV or Parquet file (or Parquet directory).
        max_length: Maximum sequence length for padding/truncation.
        load_coords: Whether to load 3D coordinates (Parquet only).
        require_indices: Deprecated, kept for backward compatibility. Ignored.
    """

    def __init__(
        self,
        dataset_path: str,
        max_length: int,
        *,
        load_coords: bool = True,
        require_indices: bool = False,
    ):
        p = Path(dataset_path)
        suffix = p.suffix.lower()

        self._is_parquet = False
        if p.is_dir() or suffix in {".parquet", ".parq", ".pq"}:
            self.data = pd.read_parquet(dataset_path)
            self._is_parquet = True
        elif suffix in {".csv"}:
            self.data = pd.read_csv(dataset_path)
        elif suffix in {".tsv", ".tab"}:
            self.data = pd.read_csv(dataset_path, sep="\t")
        else:
            # default to parquet for unknown suffixes/directories
            try:
                self.data = pd.read_parquet(dataset_path)
                self._is_parquet = True
            except Exception as e:
                raise RuntimeError(
                    "Unsupported file format. Provide a CSV/TSV or Parquet file."
                ) from e

        self.max_length = max_length

        # Validate required columns (only pid and protein_sequence needed for MLM)
        required_cols = {"pid", "protein_sequence"}
        missing = required_cols - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Coordinates are only supported from Parquet-backed datasets
        self.has_coords = (
            bool(load_coords)
            and self._is_parquet
            and ("coordinates" in self.data.columns)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        return BaseTokenizedDataset._build_output_from_row(
            row,
            max_length=self.max_length,
            has_coords=self.has_coords,
        )


class DummyMLMDataset(Dataset):
    """Placeholder dataset producing random sequences for MLM smoke tests."""

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int = 32,
    ):
        """Initialize dummy MLM dataset.

        Args:
            num_samples: Number of samples in dataset.
            seq_len: Sequence length for each sample.
            vocab_size: Vocabulary size (default 32 for amino acids).
        """
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        # Amino acid characters (indices 4-23 in DEFAULT_VOCAB)
        self._aa_chars = "LAGVSERTIPDKQNFYMHWC"

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, str]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Dict with 'pid' and 'seq' keys.
        """
        # Generate random amino acid sequence
        seq = "".join(
            self._aa_chars[i] for i in torch.randint(0, 20, (self.seq_len,)).tolist()
        )
        return {"pid": f"dummy_{idx}", "seq": seq}


class IterableTokenizedDataset(IterableDataset, BaseTokenizedDataset):
    """Shard-wise iterable dataset over a directory of Parquet files.

    Loads a single Parquet shard at a time to bound memory use, applies
    deterministic per-epoch shuffling of shards and rows, and partitions
    samples across distributed ranks and dataloader workers.

    Args:
        dataset_path: Path to a directory containing Parquet shard files.
        max_length: Maximum sequence length for padding/truncation.
        shuffle_shards: Whether to shuffle shard order per epoch.
        shuffle_rows: Whether to shuffle selected row indices per shard per epoch.
        seed: Optional base seed for deterministic epoch shuffles.
        load_coords: Whether to load 3D coordinates.
        require_indices: Deprecated, kept for backward compatibility. Ignored.
    """

    def __init__(
        self,
        dataset_path: str,
        max_length: int,
        *,
        shuffle_shards: bool = True,
        shuffle_rows: bool = True,
        seed: int = 0,
        load_coords: bool = True,
        require_indices: bool = False,
    ):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.is_dir():
            raise RuntimeError(
                "IterableTokenizedDataset expects a directory of Parquet files."
            )
        self.max_length = int(max_length)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_rows = bool(shuffle_rows)
        self.seed = int(seed)
        self._epoch = -1
        # user-intent flag for whether to load coordinates at all
        self._load_coords = bool(load_coords)

        # enumerate shard files and stats
        shard_paths = sorted(
            [
                p
                for p in self.dataset_path.iterdir()
                if p.suffix.lower() in {".parquet", ".parq", ".pq"}
            ]
        )
        if len(shard_paths) == 0:
            raise RuntimeError("No Parquet shards found in directory.")

        self._shards: list[Path] = []
        self._rows_per_shard: list[int] = []
        cols_union: set[str] = set()
        for sp in shard_paths:
            pf = pq.ParquetFile(sp.as_posix())
            self._shards.append(sp)
            self._rows_per_shard.append(int(pf.metadata.num_rows))
            try:
                schema = pf.schema_arrow
                cols_union.update([f.name for f in schema])
            except Exception:
                # best-effort
                pass
        self._offsets = np.cumsum([0] + self._rows_per_shard[:-1]).tolist()
        self._total_rows = int(sum(self._rows_per_shard))

        # Track whether the directory has coordinates column
        self.has_coords = ("coordinates" in cols_union) if len(cols_union) > 0 else True

    def __len__(self) -> int:
        # Per-rank sample cap to keep equal sample counts across ranks
        world_size = 1
        try:
            import torch.distributed as dist  # local import to avoid hard dep at import time

            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
        except Exception:
            world_size = 1
        return self._total_rows // max(1, int(world_size))

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # epoch counter for deterministic shuffles
        self._epoch += 1
        seed_base = (0x9E3779B97F4A7C15 ^ self.seed) + (self._epoch * 0x1000003)

        # rank/world from torch.distributed if available
        rank = 0
        world_size = 1
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
        except Exception:
            rank, world_size = 0, 1

        # dataloader workers
        wi = get_worker_info()
        if wi is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = wi.num_workers, wi.id

        # per-epoch shard order
        shard_indices = list(range(len(self._shards)))
        if self.shuffle_shards:
            rng = np.random.RandomState(seed_base & 0xFFFFFFFF)
            rng.shuffle(shard_indices)

        # equalize per-rank sample counts (drop global remainder)
        per_rank_cap = self._total_rows // max(1, world_size)
        emitted = 0

        for s_idx in shard_indices:
            if emitted >= per_rank_cap:
                break
            spath = self._shards[s_idx]
            nrows = int(self._rows_per_shard[s_idx])
            start = int(self._offsets[s_idx])

            # rows assigned to this rank (global striping)
            rank_rows = [
                i for i in range(nrows) if ((start + i) % max(1, world_size)) == rank
            ]
            if not rank_rows:
                continue

            if self.shuffle_rows:
                rng_rows = np.random.RandomState(
                    (seed_base + 1009 + s_idx) & 0xFFFFFFFF
                )
                rng_rows.shuffle(rank_rows)

            # within-rank worker striping
            rank_rows = rank_rows[worker_id :: max(1, num_workers)]
            if not rank_rows:
                continue

            # read only required columns
            want_cols = ["pid", "protein_sequence"]
            use_coords = self.has_coords and self._load_coords
            if use_coords:
                want_cols.append("coordinates")
            df = pd.read_parquet(spath.as_posix(), columns=want_cols)

            for i in rank_rows:
                if emitted >= per_rank_cap:
                    break
                row = df.iloc[i]
                yield BaseTokenizedDataset._build_output_from_row(
                    row,
                    max_length=self.max_length,
                    has_coords=use_coords,
                )
                emitted += 1


class MapAsIterableDataset(IterableDataset):
    """
    Wrap a map-style Dataset to behave like an IterableDataset.

    This is useful when mixing map-style and iterable datasets, because PyTorch
    DataLoader does not allow mixing samplers/shuffle with IterableDatasets.

    Behavior:
      - Each epoch yields `num_samples` samples (default: len(dataset))
      - Sampling is with replacement, uniformly over indices [0, len(dataset))
      - Dataloader worker IDs stripe the sample positions to avoid multiplying
        total emitted samples by `num_workers`.
    """

    def __init__(
        self, dataset: Dataset, *, num_samples: int | None = None, seed: int = 0
    ):
        super().__init__()
        self.dataset = dataset
        self.num_samples = int(num_samples) if num_samples is not None else len(dataset)
        self.seed = int(seed)
        self._epoch = -1

    def __len__(self) -> int:
        return int(self.num_samples)

    def __iter__(self):
        self._epoch += 1
        wi = get_worker_info()
        if wi is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = wi.num_workers, wi.id

        # Deterministic per-epoch, per-worker RNG
        seed_base = (0x9E3779B97F4A7C15 ^ self.seed) + (self._epoch * 0x1000003)
        rng = np.random.RandomState((seed_base + worker_id) & 0xFFFFFFFF)

        n = int(self.num_samples)
        L = int(len(self.dataset))
        if L <= 0 or n <= 0:
            return iter(())

        # worker striping over sample positions (keeps global sample count ~num_samples)
        for _pos in range(worker_id, n, max(1, num_workers)):
            j = int(rng.randint(0, L))
            yield self.dataset[j]


class InterleavedIterableDataset(IterableDataset):
    """
    Interleave samples from multiple IterableDatasets according to fractions.

    Each epoch yields `num_samples` items (default: sum(len(ds)) when available,
    otherwise falls back to 0 which effectively yields nothing).

    When a sub-dataset iterator is exhausted, it is re-initialized so mixing
    continues without prematurely stopping.

    Worker IDs stripe emitted positions so that total yielded items across all
    workers is approximately `num_samples` (not multiplied by num_workers).
    """

    def __init__(
        self,
        datasets: list[IterableDataset],
        fractions: list[float],
        *,
        num_samples: int | None = None,
        seed: int = 0,
    ):
        super().__init__()
        if len(datasets) == 0:
            raise ValueError("InterleavedIterableDataset requires at least 1 dataset")
        if len(datasets) != len(fractions):
            raise ValueError("datasets and fractions must have the same length")

        fr = np.asarray([float(f) for f in fractions], dtype=np.float64)
        if np.any(fr < 0):
            raise ValueError("fractions must be non-negative")
        if float(fr.sum()) <= 0:
            raise ValueError("fractions must sum to a positive value")
        fr = fr / float(fr.sum())

        self.datasets = datasets
        self.fractions = fr.tolist()
        self.seed = int(seed)
        self._epoch = -1

        if num_samples is None:
            # best-effort: sum dataset lengths if available
            total = 0
            for ds in datasets:
                try:
                    total += int(len(ds))  # type: ignore[arg-type]
                except Exception:
                    total = 0
                    break
            num_samples = total
        self.num_samples = int(num_samples)

    def __len__(self) -> int:
        return int(self.num_samples)

    def __iter__(self):
        self._epoch += 1
        wi = get_worker_info()
        if wi is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = wi.num_workers, wi.id

        n = int(self.num_samples)
        if n <= 0:
            return iter(())

        # Deterministic per-epoch, per-worker RNG
        seed_base = (0x9E3779B97F4A7C15 ^ self.seed) + (self._epoch * 0x1000003)
        rng = np.random.RandomState((seed_base + worker_id) & 0xFFFFFFFF)

        # Create iterators; we re-create an iterator when it is exhausted.
        iters = [iter(ds) for ds in self.datasets]
        fr = np.asarray(self.fractions, dtype=np.float64)

        for _pos in range(worker_id, n, max(1, num_workers)):
            # Choose dataset id according to fractions
            ds_idx = int(rng.choice(len(iters), p=fr))
            try:
                yield next(iters[ds_idx])
            except StopIteration:
                iters[ds_idx] = iter(self.datasets[ds_idx])
                yield next(iters[ds_idx])


STRUCTURE_EXTENSIONS = {".pdb", ".ent", ".cif", ".mmcif"}


class StructureFolderDataset(Dataset):
    """Dataset for a folder of PDB/mmCIF structure files.

    This dataset is designed for evaluation with structure-based metrics.
    Each structure file produces one sample with:
      - pid: structure identifier (filename stem)
      - seq: amino acid sequence extracted from structure
      - coords: backbone coordinates [max_length, 3, 3] for N, CA, C atoms
      - masks: boolean mask [max_length] for valid positions
      - nan_masks: same as masks (all backbone atoms present or NaN)

    Note: No VQ indices are provided since these are raw structures without
    pre-computed structure tokens.

    Args:
        folder_path: Path to folder containing PDB/mmCIF files.
        max_length: Maximum sequence length (for padding/truncation).
        chain_id: Specific chain to extract from each file. If None, uses
            first polymer chain found in each structure.
        strict: If True, raise errors on missing backbone atoms.
            If False (default), fill missing atoms with NaN.
        recursive: If True, search subdirectories recursively.

    Raises:
        ValueError: If folder_path is not a directory or contains no
            structure files.

    Example config:
        .. code-block:: yaml

            eval:
              pdb_benchmark:
                path: /path/to/pdb_folder
                format: structure
                chain_id: A
                metrics:
                  only: [lddt, tm_score, rmsd]
    """

    def __init__(
        self,
        folder_path: str | Path,
        max_length: int,
        *,
        chain_id: str | None = None,
        strict: bool = False,
        recursive: bool = False,
    ):
        self.folder_path = Path(folder_path)
        if not self.folder_path.is_dir():
            raise ValueError(f"Not a directory: {folder_path}")

        self.max_length = int(max_length)
        self.chain_id = chain_id
        self.strict = bool(strict)
        self.recursive = bool(recursive)

        # Discover structure files
        if recursive:
            self._files = sorted(
                [
                    p
                    for p in self.folder_path.rglob("*")
                    if p.is_file() and p.suffix.lower() in STRUCTURE_EXTENSIONS
                ]
            )
        else:
            self._files = sorted(
                [
                    p
                    for p in self.folder_path.iterdir()
                    if p.is_file() and p.suffix.lower() in STRUCTURE_EXTENSIONS
                ]
            )

        if len(self._files) == 0:
            raise ValueError(
                f"No structure files found in {folder_path}. "
                f"Supported extensions: {STRUCTURE_EXTENSIONS}"
            )

        # Flags for compatibility with existing dataset code
        self.has_coords = True
        self._is_parquet = False

    def __len__(self) -> int:
        """Return number of structure files in the dataset."""
        return len(self._files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Load and parse a structure file.

        Args:
            idx: Index of the structure file to load.

        Returns:
            Dict with keys:
                - pid (str): Structure identifier
                - seq (str): Amino acid sequence
                - coords (Tensor): [max_length, 3, 3] backbone coordinates
                - masks (Tensor): [max_length] boolean mask for valid positions
                - nan_masks (Tensor): [max_length] same as masks

            Note: 'indices' key is NOT included (not available for raw structures).
        """
        path = self._files[idx]

        # Parse structure
        data = parse_structure(
            path,
            chain_id=self.chain_id,
            strict=self.strict,
        )

        seq_len = len(data.protein_sequence)
        coords = data.coords  # [L, 3, 3]

        # Truncate if needed
        if seq_len > self.max_length:
            seq = data.protein_sequence[: self.max_length]
            coords = coords[: self.max_length]
            seq_len = self.max_length
        else:
            seq = data.protein_sequence

        # Pad coordinates to max_length with NaN
        pad_len = self.max_length - seq_len
        coords_padded = np.full((self.max_length, 3, 3), np.nan, dtype=np.float32)
        coords_padded[:seq_len] = coords

        # Build mask (True for valid positions)
        mask = [True] * seq_len + [False] * pad_len

        out: dict[str, torch.Tensor | str] = {
            "pid": data.pid,
            "seq": seq,
            "coords": torch.tensor(coords_padded, dtype=torch.float32),
            "masks": torch.tensor(mask, dtype=torch.bool),
            "nan_masks": torch.tensor(mask, dtype=torch.bool),
        }

        return out

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        return (
            f"StructureFolderDataset("
            f"folder={self.folder_path}, "
            f"num_files={len(self._files)}, "
            f"max_length={self.max_length})"
        )
