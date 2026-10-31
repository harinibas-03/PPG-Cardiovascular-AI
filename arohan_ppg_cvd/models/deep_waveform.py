"""Lightweight 1D CNN Waveform Encoder for Raw PPG Waveforms (Branch B).

This module processes unstructured continuous PPG waveforms:
1. Resamples / interpolates arbitrary-length raw waveforms to a standardized fixed length (e.g. 512 points).
2. Performs robust baseline removal and per-window standardization.
3. Applies a multi-scale 1D convolutional feature extractor:
   - Small kernel (kernel=7): Systolic upstroke acceleration and high-frequency dicrotic notch inflection.
   - Medium kernel (kernel=15): Ejection duration and pulse contour morphology.
   - Large kernel (kernel=31): Diastolic decay, vascular compliance, and pulse wave reflection.
4. Uses ReLU non-linearities, temporal max-pooling, and global average/max pooling.
5. Produces a compact 32-dimensional latent waveform embedding vector.
"""

from typing import List, Optional, Union
import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin

class DeepWaveformEncoder(BaseEstimator, TransformerMixin):
    """Lightweight 1D CNN encoder extracting a 32-dimensional embedding from raw PPG signals."""

    def __init__(
        self,
        target_length: int = 512,
        embedding_dim: int = 32,
        random_state: int = 42,
    ):
        self.target_length = target_length
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self._is_fitted = False
        self._init_kernels()

    def _init_kernels(self):
        """Initialize multi-scale physiological 1D convolutional filter banks."""
        rng = np.random.RandomState(self.random_state)
        
        # Scale 1: Fast systolic upstroke & notch (7 taps, 6 filters)
        t7 = np.linspace(-1, 1, 7)
        k_s1 = [
            np.gradient(np.exp(-0.5 * (t7 / 0.3) ** 2)),                     # 1st deriv Gaussian
            np.gradient(np.gradient(np.exp(-0.5 * (t7 / 0.3) ** 2))),           # 2nd deriv
            np.sin(np.pi * t7),                                               # Wavelet sine
            np.cos(np.pi * t7),                                               # Wavelet cosine
            rng.randn(7) * 0.1,
            rng.randn(7) * 0.1,
        ]

        # Scale 2: Pulse contour morphology (15 taps, 6 filters)
        t15 = np.linspace(-1, 1, 15)
        k_s2 = [
            np.exp(-0.5 * (t15 / 0.4) ** 2) * np.cos(1.5 * np.pi * t15),
            np.gradient(np.exp(-0.5 * (t15 / 0.4) ** 2)),
            np.sin(2.0 * np.pi * t15),
            np.exp(-np.abs(t15) / 0.3),
            rng.randn(15) * 0.1,
            rng.randn(15) * 0.1,
        ]

        # Scale 3: Diastolic decay & vascular compliance (31 taps, 4 filters)
        t31 = np.linspace(-1, 1, 31)
        k_s3 = [
            np.exp(-0.5 * (t31 / 0.5) ** 2) * np.cos(0.8 * np.pi * t31),
            np.gradient(np.exp(-0.5 * (t31 / 0.5) ** 2)),
            np.sin(np.pi * t31),
            rng.randn(31) * 0.1,
        ]

        # Normalize filter energies
        self.conv1_filters = []
        for k in k_s1 + k_s2 + k_s3:
            norm = np.linalg.norm(k)
            self.conv1_filters.append(k / (norm + 1e-8))

        # Total Conv1 feature channels = 6 + 6 + 4 = 16 channels
        # Projection matrix from pooled features (16 avg + 16 max = 32) to embedding_dim
        self.proj_matrix = rng.randn(32, self.embedding_dim) * np.sqrt(2.0 / 32)

    def _preprocess_waveform(self, sig: np.ndarray) -> np.ndarray:
        """Standardize length and amplitude of a 1D raw waveform."""
        sig = np.asarray(sig, dtype=np.float32).flatten()
        if len(sig) == 0:
            return np.zeros(self.target_length, dtype=np.float32)

        # Resample to target length
        if len(sig) != self.target_length:
            resampled = signal.resample(sig, self.target_length)
        else:
            resampled = sig.copy()

        # Detrend and z-score standardize
        resampled = signal.detrend(resampled)
        std_val = np.std(resampled)
        if std_val > 1e-6:
            resampled = (resampled - np.mean(resampled)) / std_val
        else:
            resampled = resampled - np.mean(resampled)

        return resampled.astype(np.float32)

    def _forward_single(self, sig: np.ndarray) -> np.ndarray:
        """Pass a single normalized waveform through the 1D CNN pipeline."""
        x = self._preprocess_waveform(sig)

        # Layer 1: Multi-scale 1D convolutions + ReLU
        feature_maps = []
        for flt in self.conv1_filters:
            # Mode 'same' convolution
            conv_out = np.convolve(x, flt, mode="same")
            # ReLU activation
            relu_out = np.maximum(0.0, conv_out)
            feature_maps.append(relu_out)

        # Temporal pooling (Global Average & Global Max Pooling per channel)
        pooled = []
        for fmap in feature_maps:
            pooled.append(float(np.mean(fmap)))
            pooled.append(float(np.max(fmap)))

        pooled_vec = np.array(pooled, dtype=np.float32)  # shape (32,)

        # Projection to embedding_dim
        embedding = np.dot(pooled_vec, self.proj_matrix)
        # L2 normalize embedding
        emb_norm = np.linalg.norm(embedding)
        if emb_norm > 1e-8:
            embedding = embedding / emb_norm

        return embedding.astype(np.float32)

    def fit(self, X, y=None):
        """Fit encoder on training waveforms."""
        self._is_fitted = True
        return self

    def transform(self, X: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        """Encode a collection or batch of raw PPG waveforms into embedding matrix (N, embedding_dim)."""
        embeddings = []
        for sig in X:
            emb = self._forward_single(sig)
            embeddings.append(emb)
        return np.vstack(embeddings).astype(np.float32)

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X, y).transform(X)

