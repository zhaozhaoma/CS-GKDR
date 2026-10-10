# -*- coding: utf-8 -*-
"""REST-meta-MDD fMRI and sMRI datasets.

fMRI: AAL time series -> Fisher-z correlation matrix (node features) and a graph keeping the
top 10% absolute correlations (edges). sMRI: VBM volumes (default wc1), mean-centred per image.
Both are cached on first use. The cache location is os.path.join(CACHE_ROOT, data_dir, ...),
so with an absolute data_dir the cache lives inside data_dir.
"""
import nibabel as nib
import numpy as np
import os
import pandas as pd
import torch

from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch.utils.data import Dataset

NP_TO_TORCH_DTYPES = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}

ALLOWED_SPLITS = {
    # Regex for matching all columns excluding the ones in the dev and test splits
    # using negative pattern matching
    "train": r"(?!(S4|S8|S15|S17)-).*",
    "dev": r"((S4|S8)-)",
    "test": r"((S15|S17)-)",
    "full": r".*",
}


# Dataset for fMRI data that stores everything in memory

# Root directory for preprocessed caches (only used when data_dir is a relative path);
# set from data.cache_root in the configuration by set_cache_root().
CACHE_ROOT = "data/cache"


def set_cache_root(path: str) -> None:
    global CACHE_ROOT
    CACHE_ROOT = path


class RESTfMRIDataset(Dataset):
    def __init__(
        self,
        data_dir,
        metadata_path,
        split="full",
    ):
        super().__init__()
        self.cache_path = os.path.join(CACHE_ROOT, data_dir, split)
        metadata = self._load_metadata(metadata_path, split)

        if self._cache_exists():
            self.edge_indices, self.node_features = self._load_cache()
        else:
            self.edge_indices, self.node_features = self._preprocess_raw_signals(
                metadata, data_dir
            )
            self._create_cache()

        self.num_samples = self.node_features.shape[0]
        self.num_nodes = self.node_features.shape[1]
        self.labels = torch.tensor(metadata.label.values)

    def _load_metadata(self, metadata_path, split):
        metadata = pd.read_csv(metadata_path)
        cond = metadata.subID.str.match(ALLOWED_SPLITS[split])
        return metadata[cond].reset_index(drop=True)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return Data(
            x=self.node_features[idx],
            edge_index=self.edge_indices[idx],
            y=self.labels[idx],
        )

    def _preprocess_raw_signals(self, metadata, data_dir):
        adj_list = []
        node_features_list = []
        for id in metadata.subID:
            # Read the raw signals and create correlation matrix
            filepath = os.path.join(data_dir, f"{id}.npy")
            raw_signals = np.load(filepath)
            corr = self._create_corr(raw_signals)

            # Transform correlation matrix to adjacency matrix
            node_features = torch.tensor(corr, dtype=torch.float32)
            topk = node_features.reshape(-1)
            topk, _ = torch.sort(abs(topk), dim=0, descending=True)
            threshold = topk[int(node_features.shape[0] ** 2 / 20 * 2)]
            # Note: ties at the threshold can change the number of edges of a subject, in which
            # case torch.stack below raises an error (the graphs are stacked as one tensor).
            adj = (torch.abs(node_features) >= threshold).to(int)
            edge_index = dense_to_sparse(adj)[0]

            adj_list.append(edge_index)
            node_features_list.append(node_features)

        return torch.stack(adj_list), torch.stack(node_features_list)

    def _create_corr(self, data):
        eps = 1e-16
        R = np.corrcoef(data)
        R[np.isnan(R)] = 0
        R = R - np.diag(np.diag(R))
        R[R >= 1] = 1 - eps
        corr = 0.5 * np.log((1 + R) / (1 - R))
        return corr

    def _create_cache(self):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        edge_indices_path = os.path.join(self.cache_path, "edge_indices.pt")
        node_features_path = os.path.join(self.cache_path, "node_features.pt")

        torch.save(self.edge_indices, edge_indices_path)
        torch.save(self.node_features, node_features_path)

    def _cache_exists(self):
        edge_indices_path = os.path.join(self.cache_path, "edge_indices.pt")
        if not os.path.exists(edge_indices_path):
            return False

        node_features_path = os.path.join(self.cache_path, "node_features.pt")
        if not os.path.exists(node_features_path):
            return False

        return True

    def _load_cache(self):
        edge_indices_path = os.path.join(self.cache_path, "edge_indices.pt")
        node_features_path = os.path.join(self.cache_path, "node_features.pt")

        edge_indices = torch.load(edge_indices_path, weights_only=True)
        node_features = torch.load(node_features_path, weights_only=True)
        return edge_indices, node_features


class RESTsMRIDataset(Dataset):
    IMAGE_TYPES = [
        "c1",  # Gray matter density in native space
        "c2",  # White matter density in native space
        "c3",  # Cerebrospinal fluid density in native space
        "wc1",  # Gray matter density in MNI space
        "wc2",  # White matter density in MNI space
        "wc3",  # Cerebrospinal fluid density in MNI space
        "mwc1",  # Gray matter volume in MNI space
        "mwc2",  # White matter volume in MNI space
        "mwc3",  # Cerebrospinal fluid volume density in MNI space
    ]

    def __init__(
        self,
        data_dir,
        metadata_path,
        imgtypes=[
            "wc1"
        ],  # Can be any combination of allowed imgtypes (["wc1", "mwc1"...])
        split="full",  # Can be "train", "dev", "test" or "full"
        normalize=True,  # Normalize images
        dtype=np.float32,  # Reduce to save memory
        inmemory=False,  # Load everything in memory
    ):
        super().__init__()
        imgtypes = sorted(imgtypes)
        metadata = self._load_metadata(metadata_path, split)
        self.cache_path = os.path.join(CACHE_ROOT, data_dir, "-".join(imgtypes))
        self.normalize = normalize
        self.dtype = dtype
        self.ids = metadata.subID
        self.labels = torch.tensor(metadata.label.values)
        self.dshape = self._get_data_shape(data_dir, imgtypes)

        if not self._cache_exists():
            self._create_cache(data_dir, imgtypes)

        if inmemory:
            self._load_data_to_memory()
            self.load_data_fn = self._inmemory_load_data
        else:
            self.load_data_fn = self._lazy_load_data

        self.num_samples = len(self.ids)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.load_data_fn(idx)

    def _load_metadata(self, metadata_path, split):
        metadata = pd.read_csv(metadata_path)
        cond = metadata.subID.str.match(ALLOWED_SPLITS[split])
        return metadata[cond].reset_index(drop=True)

    def _cache_exists(self):
        if not os.path.exists(self.cache_path):
            return False

        for subid in self.ids:
            if not os.path.exists(os.path.join(self.cache_path, f"{subid}.npy")):
                return False

        return True

    def _create_cache(self, data_dir, imgtypes):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        for subid in self.ids:
            if os.path.exists(os.path.join(self.cache_path, f"{subid}.npy")):
                continue

            dpoint = np.zeros(self.dshape[1:], dtype=self.dtype)

            for i, split in enumerate(imgtypes):

                filepath = os.path.join(data_dir, split, f"{subid}.nii.gz")
                image = nib.load(filepath).get_fdata()

                if self.normalize:
                    image -= image.mean()

                dpoint[i] = image

            np.save(os.path.join(self.cache_path, f"{subid}.npy"), dpoint)

    def _lazy_load_data(self, idx):
        filename = self.ids[idx]
        data = np.load(os.path.join(self.cache_path, f"{filename}.npy"))
        data_t = torch.tensor(data, dtype=NP_TO_TORCH_DTYPES[self.dtype])
        return data_t, self.labels[idx]

    def _inmemory_load_data(self, idx):
        return self.data[idx], self.labels[idx]

    def _load_data_to_memory(self):
        self.data = torch.zeros(self.dshape, dtype=NP_TO_TORCH_DTYPES[self.dtype])
        for i, subid in enumerate(self.ids):
            filepath = os.path.join(self.cache_path, f"{subid}.npy")
            self.data[i] = torch.tensor(np.load(filepath))

    def _get_data_shape(self, data_dir, imgtypes):
        subid = self.ids[0]
        imgtype = imgtypes[0]
        filepath = os.path.join(data_dir, imgtype, f"{subid}.nii.gz")
        image_shape = nib.load(filepath).get_fdata().shape
        return (len(self.ids), len(imgtypes), *image_shape)
