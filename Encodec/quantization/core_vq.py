# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Core vector quantization implementation."""

import typing as tp
import warnings

from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F

import distrib


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    """ema update parameter. moving_avg = moving_avg + (1-decay) * new
    Args:
        moving_avg (_type_): 
        new (_type_): update parameter
        decay (float): update rate
    """
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(
            means, "c d -> () c d"
        )
        dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        # distrib.broadcast_tensors(self.buffers()) # FIXME: this is not working for some reason

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        # distrib.broadcast_tensors(self.buffers()) # FIXME: this is not working for some reason

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        ) # get the distance between x and embed
        embed_ind = dist.max(dim=-1).indices # get the index of the closest embed
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x) # [2,32,128] -> [64,128]

        self.init_embed_(x) # to better initialize the codebook

        embed_ind = self.quantize(x) # get the index of the closest embed
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if self.training: # update the codebook
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


# ---------------------------------------------------------------------------
# Q²D² — Two-Dimensional Quantisation with Fixed Geometric Lattice
# ---------------------------------------------------------------------------

def _build_square_lattice(levels: int) -> torch.Tensor:
    """Return a (levels*levels, 2) tensor of square-grid coordinates, normalised to [-1, 1]."""
    xs = torch.linspace(-1.0, 1.0, levels)
    ys = torch.linspace(-1.0, 1.0, levels)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    return coords  # [levels*levels, 2]


def _build_rhombic_lattice(levels: int) -> torch.Tensor:
    """Return a (levels*levels, 2) tensor of rhombic-lattice coordinates, normalised to [-1, 1].

    Basis vectors b1=[1,0] and b2=[0.5, √3/2] give the optimal 2-D sphere packing.
    This is the grid type reported as best in the Q²D² paper (arXiv:2512.01537).
    """
    b1 = torch.tensor([1.0, 0.0])
    b2 = torch.tensor([0.5, 3.0 ** 0.5 / 2.0])
    coords = [k1 * b1 + k2 * b2 for k1 in range(levels) for k2 in range(levels)]
    pts = torch.stack(coords)          # [levels², 2]
    pts = pts - pts.mean(dim=0)
    scale = pts.abs().max()
    if scale > 0:
        pts = pts / scale
    return pts  # normalised to [-1, 1]


def _build_hex_lattice(levels: int) -> torch.Tensor:
    """Return a (levels*levels, 2) tensor of hexagonal-grid coordinates.

    Hexagonal close-packing: odd rows are offset by 0.5 units.
    The output is normalised so the max absolute coordinate is 1.
    """
    coords = []
    for row in range(levels):
        x_offset = 0.5 if row % 2 else 0.0
        for col in range(levels):
            x = col + x_offset
            y = row * (3.0 ** 0.5) / 2.0
            coords.append([x, y])
    pts = torch.tensor(coords, dtype=torch.float32)
    pts = pts - pts.mean(dim=0)
    scale = pts.abs().max()
    if scale > 0:
        pts = pts / scale
    return pts  # [levels*levels, 2]


class Q2D2Codebook(nn.Module):
    """Faithful per-pair fixed geometric-lattice codebook for Q²D² quantisation.

    The latent vector is split into dim/2 independent 2-D pairs.  Each pair is
    bounded with tanh, then independently quantised to the nearest point on a
    shared 2-D grid (rhombic, hex, or square).  The grid is never updated during
    training — only the encoder learns to map into it.

    This faithfully implements the algorithm in arXiv:2512.01537v1 (Q²D²):
    - tanh bounding: z̃ = tanh(z) ∈ (-1, 1)²
    - per-pair independent nearest-neighbour lookup
    - straight-through estimator (applied in VectorQuantization.forward)
    - no EMA, no commitment loss

    Args:
        dim (int): Total latent dimension (must be even). n_pairs = dim // 2.
        levels (int): Grid points per lattice dimension.  Codebook has levels²
            entries per pair.  The paper uses 7–11; 11 is a stable default.
        grid (str): ``"rhombic"`` (default, best per paper), ``"hex"``, or
            ``"square"``.
    """
    def __init__(self, dim: int, levels: int = 11, grid: str = "rhombic"):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Q2D2Codebook requires even dim, got {dim}")

        self.dim = dim
        self.n_pairs = dim // 2
        self.levels = levels
        self.codebook_size = levels * levels   # per-pair vocab size
        self.grid_type = grid

        if grid == "rhombic":
            grid_pts = _build_rhombic_lattice(levels)
        elif grid == "hex":
            grid_pts = _build_hex_lattice(levels)
        else:
            grid_pts = _build_square_lattice(levels)

        # [codebook_size, 2] — shared 2-D grid, fixed (not trained)
        self.register_buffer("grid", grid_pts)
        # 'embed' alias for API compatibility with EuclideanCodebook
        self.register_buffer("embed", grid_pts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quantize_pairs(self, x_tanh: torch.Tensor) -> torch.Tensor:
        """Per-pair nearest-neighbour lookup.

        Args:
            x_tanh: ``[N, dim]`` — tanh-bounded encoder output.
        Returns:
            ``[N, n_pairs]`` integer indices into ``self.grid``.
        """
        N = x_tanh.shape[0]
        x_pairs = x_tanh.view(N, self.n_pairs, 2)           # [N, P, 2]
        g = self.grid                                         # [C, 2]

        # Squared Euclidean distance: [N, P, C]
        x2 = x_pairs.pow(2).sum(-1, keepdim=True)            # [N, P, 1]
        g2 = g.pow(2).sum(-1).view(1, 1, -1)                 # [1, 1, C]
        xg = torch.einsum("npi,ci->npc", x_pairs, g)         # [N, P, C]
        dist2 = x2 + g2 - 2.0 * xg                          # [N, P, C]

        return dist2.argmin(dim=-1)                           # [N, P]

    # ------------------------------------------------------------------
    # Public API (mirrors EuclideanCodebook interface)
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return pair-0 index shaped to ``[*spatial]`` (RVQ compat)."""
        shape = x.shape
        x_flat = rearrange(x, "... d -> (...) d")
        pair_ind = self._quantize_pairs(torch.tanh(x_flat))  # [N, n_pairs]
        return pair_ind[:, 0].view(*shape[:-1])

    def encode_pairs(self, x: torch.Tensor) -> torch.Tensor:
        """Return all per-pair indices shaped ``[n_pairs, *spatial]``.

        Use this for analysis — it exposes one integer index per pair per frame,
        analogous to the per-codebook index stream in RVQ.

        Args:
            x: ``[..., dim]`` encoder output (any batch shape, e.g. ``[B, T, dim]``).
        Returns:
            ``[n_pairs, *spatial]`` — e.g. ``[n_pairs, B, T]`` for ``x: [B, T, dim]``.
        """
        shape = x.shape           # e.g. [B, T, dim]
        spatial = shape[:-1]      # e.g. [B, T]
        x_flat = rearrange(x, "... d -> (...) d")             # [N, dim]
        pair_ind = self._quantize_pairs(torch.tanh(x_flat))  # [N, n_pairs]
        return pair_ind.t().view(self.n_pairs, *spatial)      # [n_pairs, *spatial]

    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        """Reconstruct using pair-0 index only.

        Only used in the RVQ ``encode`` path for residual subtraction, which is
        a no-op with ``n_q=1``.  Other pairs are zeroed out.
        """
        pts = self.grid[embed_ind]  # [..., 2]
        out = torch.zeros(*embed_ind.shape, self.dim,
                          device=self.grid.device, dtype=self.grid.dtype)
        out[..., :2] = pts
        return out

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        shape, dtype = x.shape, x.dtype
        x_flat = rearrange(x, "... d -> (...) d")             # [N, dim]

        x_tanh = torch.tanh(x_flat)                          # bound to (-1, 1)
        pair_ind = self._quantize_pairs(x_tanh)              # [N, n_pairs]

        # Dequantize: look up and concatenate all pair grid points
        N = x_flat.shape[0]
        quantized_flat = self.grid[pair_ind].view(N, self.dim)  # [N, dim]
        quantized = quantized_flat.view(shape).type(dtype)

        # Return pair-0 index as embed_ind for STE / RVQ interface compat
        embed_ind = pair_ind[:, 0].view(*shape[:-1])
        return quantized, embed_ind


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size, the number of vectors in the codebook
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
                            the dimension of each vector in the codebook
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.,
        use_q2d2: bool = False,
        q2d2_grid: str = "rhombic",
        q2d2_levels: int = 11,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight
        self.use_q2d2 = use_q2d2

        if use_q2d2:
            self._codebook = Q2D2Codebook(dim=_codebook_dim, levels=q2d2_levels,
                                          grid=q2d2_grid)
        else:
            self._codebook = EuclideanCodebook(dim=_codebook_dim, codebook_size=codebook_size,
                                               kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                                               decay=decay, epsilon=epsilon,
                                               threshold_ema_dead_code=threshold_ema_dead_code)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def encode_pairs(self, x: torch.Tensor) -> torch.Tensor:
        """Q2D2 only: return per-pair indices ``[n_pairs, B, T]`` for analysis."""
        if not self.use_q2d2:
            raise RuntimeError("encode_pairs is only available when use_q2d2=True")
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        return self._codebook.encode_pairs(x)  # [n_pairs, B, T]

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x):
        device = x.device
        x = rearrange(x, "b d n -> b n d") # [2,128,32] -> [2,32,128]
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)
        if self.training and not self.use_q2d2:
            warnings.warn('When using RVQ in training model, first check '
                          'https://github.com/facebookresearch/encodec/issues/25 . '
                          'The bug wasn\'t fixed here for reproducibility.')
            if self.commitment_weight > 0:
                # x = F.normalize(x)  
                # quantize = F.normalize(quantize)  
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x, n_q: tp.Optional[int] = None):
        quantized_out = 0.0
        residual = x # x is encoder output emb

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)

        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized # y^hat

            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
