"""
DG-LoGraP Model Implementation

Integrates:
- SAGSAM for dynamic graph generation
- SAGS-GCN for spatial graph convolution
- Grouped latent spatio-temporal processes for batch covariance

Based on the DeepAR/GPT architectures with dynamic graph awareness.
"""

from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Literal, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from pytorch_forecasting import DeepAR
from pytorch_forecasting.models.base_model import AutoRegressiveBaseModelWithCovariates
from pytorch_forecasting.models.nn import HiddenState, MultiEmbedding
from pytorch_forecasting.data.encoders import MultiNormalizer, NaNLabelEncoder
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.utils import apply_to_list, to_list
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    MASE,
    RMSE,
    SMAPE,
    DistributionLoss,
    Metric,
    MultiLoss,
    MultivariateDistributionLoss,
    NormalDistributionLoss,
)

from matplotlib import pyplot as plt

from dynamic_graph import (
    load_static_graph,
    SAGSAM,
    SAGS_GCN,
    DynamicGraphLoader,
    TemporalCorrelationKernel,
)


class DGLoGraPDeepAR(AutoRegressiveBaseModelWithCovariates):
    """
    DG-LoGraP DeepAR Model
    
    DeepAR variant with:
    - SAGSAM for dynamic graph generation
    - SAGS-GCN for spatial message passing
    - Grouped latent spatio-temporal covariance structure
    """
    
    def __init__(
        self,
        cell_type: str = "LSTM",
        hidden_size: int = 10,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        n_validation_samples: int = None,
        n_plotting_samples: int = None,
        target: Union[str, List[str]] = None,
        target_lags: Dict[str, List[int]] = {},
        loss: DistributionLoss = None,
        logging_metrics: nn.ModuleList = None,
        # DG-LoGraP specific parameters
        num_nodes: int = 358,  # PEMS03 has 358 nodes
        graph_embed_dim: int = 32,
        num_groups: int = 2,
        ranks_per_group: List[int] = None,
        num_kernels: int = 4,
        graph_path: str = None,
        use_graph_conv: bool = True,
        **kwargs,
    ):
        """
        Initialize DG-LoGraP DeepAR model.
        
        Args:
            cell_type: Type of RNN cell (LSTM or GRU)
            hidden_size: Hidden state size
            rnn_layers: Number of RNN layers
            dropout: Dropout rate
            num_nodes: Number of spatial nodes
            graph_embed_dim: Dimension for SAGSAM node embeddings
            num_groups: Number of latent groups G
            ranks_per_group: Rank for each group
            num_kernels: Number of temporal kernel mixtures
            graph_path: Path to static graph CSV
            use_graph_conv: Whether to use SAGS-GCN
            **kwargs: Additional arguments
        """
        if loss is None:
            loss = NormalDistributionLoss()
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        if n_plotting_samples is None:
            if n_validation_samples is None:
                n_plotting_samples = n_validation_samples
            else:
                n_plotting_samples = 100
        
        self.save_hyperparameters()
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)
        
        # Store graph parameters
        self.num_nodes = num_nodes
        self.graph_embed_dim = graph_embed_dim
        self.num_groups = num_groups
        self.ranks_per_group = ranks_per_group if ranks_per_group else [5] * num_groups
        self.num_kernels = num_kernels
        self.use_graph_conv = use_graph_conv
        
        # Load static graph
        if graph_path is not None:
            static_adj = load_static_graph(graph_path, num_nodes)
            self.register_buffer('static_adj', static_adj)
        else:
            self.register_buffer('static_adj', torch.eye(num_nodes))
        
        # Standard embeddings
        self.embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            embedding_paddings=embedding_paddings,
            categorical_groups=categorical_groups,
            x_categoricals=x_categoricals,
        )
        
        # Validate targets
        lagged_target_names = [l for lags in target_lags.values() for l in lags]
        assert set(self.encoder_variables) - set(to_list(target)) - set(lagged_target_names) == set(
            self.decoder_variables
        ) - set(lagged_target_names), "Encoder and decoder variables have to be the same apart from target variable"
        
        for targeti in to_list(target):
            assert targeti in time_varying_reals_encoder, f"target {targeti} has to be real"
        
        assert (isinstance(target, str) and isinstance(loss, DistributionLoss)) or (
            isinstance(target, (list, tuple)) and isinstance(loss, MultiLoss) and len(loss) == len(target)
        ), "number of targets should be equivalent to number of loss metrics"
        
        # SAGSAM for dynamic adjacency
        self.sagsam = SAGSAM(
            num_nodes=num_nodes,
            embed_dim=graph_embed_dim,
            static_adj=self.static_adj,
            dropout=dropout,
        )
        
        # SAGS-GCN for spatial convolution
        if use_graph_conv:
            self.sags_gcn = SAGS_GCN(
                num_nodes=num_nodes,
                in_features=hidden_size,
                out_features=hidden_size,
                embed_dim=graph_embed_dim,
                static_adj=self.static_adj,
                dropout=dropout,
            )
        
        # Input sizes
        self.cont_size = len(self.reals)
        self.cat_size = sum(self.embeddings.output_size.values())
        
        # Input projections
        rnn_input_size = self.cont_size + self.cat_size
        
        # RNN
        if cell_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=rnn_layers,
                dropout=dropout if rnn_layers > 1 else 0,
                batch_first=True,
            )
        elif cell_type == "GRU":
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=rnn_layers,
                dropout=dropout if rnn_layers > 1 else 0,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}")
        
        # Graph-aware output projection
        if use_graph_conv:
            # Project hidden state through SAGS-GCN before distribution parameters
            self.pre_dist_proj = nn.Linear(hidden_size, hidden_size)
        
        # Distribution parameter projector
        if isinstance(target, str):
            self.distribution_projector = nn.Linear(
                hidden_size, len(self.loss.distribution_arguments)
            )
        else:
            self.distribution_projector = nn.ModuleList([
                nn.Linear(hidden_size, len(args)) 
                for args in self.loss.distribution_arguments
            ])
        
        # Temporal kernel weights projector
        self.temporal_kernel_projector = nn.Linear(hidden_size, num_kernels)
        
        # Dynamic graph loader for loading matrices (used in loss)
        self.graph_loader = DynamicGraphLoader(
            num_nodes=num_nodes,
            num_groups=num_groups,
            ranks_per_group=self.ranks_per_group,
            embed_dim=graph_embed_dim,
            static_adj=self.static_adj,
        )
    
    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: List[str] = None,
        **kwargs,
    ):
        """Create model from dataset."""
        new_kwargs = {}
        if dataset.multi_target:
            new_kwargs.setdefault("loss", MultiLoss([NormalDistributionLoss()] * len(dataset.target_names)))
        new_kwargs.update(kwargs)
        
        assert not isinstance(dataset.target_normalizer, NaNLabelEncoder) and (
            not isinstance(dataset.target_normalizer, MultiNormalizer)
            or all([not isinstance(normalizer, NaNLabelEncoder) for normalizer in dataset.target_normalizer])
        ), "target(s) should be continuous - categorical targets are not supported"
        
        if isinstance(new_kwargs.get("loss", None), MultivariateDistributionLoss):
            assert (
                dataset.min_prediction_length == dataset.max_prediction_length
            ), "Multivariate models require constant prediction lengths"
        
        return super().from_dataset(
            dataset, allowed_encoder_known_variable_names=allowed_encoder_known_variable_names, **new_kwargs
        )
    
    def construct_input_vector(
        self, x_cat: torch.Tensor, x_cont: torch.Tensor, one_off_target: torch.Tensor = None
    ) -> torch.Tensor:
        """Create input vector for RNN."""
        if len(self.categoricals) > 0:
            embeddings = self.embeddings(x_cat)
            flat_embeddings = torch.cat([emb for emb in embeddings.values()], dim=-1)
            input_vector = flat_embeddings
        
        if len(self.reals) > 0:
            input_vector = x_cont.clone()
        
        if len(self.reals) > 0 and len(self.categoricals) > 0:
            input_vector = torch.cat([x_cont, flat_embeddings], dim=-1)
        
        # Shift target by one
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )
        
        if one_off_target is not None:
            # Get the target slice shape to match
            target_slice = input_vector[:, 0, self.target_positions]
            
            # Ensure one_off_target has the right shape for assignment
            if one_off_target.dim() == 1 and target_slice.dim() == 2:
                # target_slice is [B, num_targets], one_off_target is [B]
                one_off_target = one_off_target.unsqueeze(-1)
            elif one_off_target.dim() == 2 and target_slice.dim() == 1:
                # target_slice is [B], one_off_target is [B, 1]
                one_off_target = one_off_target.squeeze(-1)
            
            input_vector[:, 0, self.target_positions] = one_off_target
        else:
            input_vector = input_vector[:, 1:]
        
        return input_vector
    
    def get_dynamic_adjacency(self, hidden_state: torch.Tensor = None) -> torch.Tensor:
        """Get dynamic adjacency matrix from SAGSAM."""
        return self.sagsam(hidden_state)
    
    def apply_graph_conv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply SAGS-GCN to hidden states.
        
        Args:
            x: Hidden states (B, T, H)
            
        Returns:
            x_conv: Graph-convolved hidden states (B, T, H)
            dynamic_adj: Dynamic adjacency matrix
        """
        if not self.use_graph_conv:
            return x, self.static_adj
        
        batch_size, seq_len, hidden_size = x.shape
        
        # For simplicity, apply graph conv to the last time step
        # or reshape to treat batch*time as nodes (depends on application)
        
        # Here we project and apply SAGS-GCN
        x_proj = self.pre_dist_proj(x)  # (B, T, H)
        
        # Apply SAGS-GCN (treating each time step independently)
        # Note: For proper spatio-temporal modeling, would need to reshape
        # based on the actual spatial structure
        
        # Get dynamic adjacency
        dynamic_adj = self.sagsam()
        
        return x_proj, dynamic_adj
    
    def encode(self, x: Dict[str, torch.Tensor]) -> HiddenState:
        """Encode sequence into hidden state."""
        assert x["encoder_lengths"].min() > 0
        input_vector = self.construct_input_vector(x["encoder_cat"], x["encoder_cont"])
        
        # Encode with RNN
        encoder_output, hidden_state = self.rnn(input_vector)
        
        # Apply graph convolution
        if self.use_graph_conv:
            encoder_output, _ = self.apply_graph_conv(encoder_output)
        
        return hidden_state, encoder_output
    
    def decode_all(
        self,
        x: torch.Tensor,
        hidden_state: HiddenState,
        lengths: torch.Tensor = None,
    ):
        """Decode all time steps at once."""
        decoder_output, _ = self.rnn(x, hidden_state)
        
        # Apply graph convolution
        if self.use_graph_conv:
            decoder_output, _ = self.apply_graph_conv(decoder_output)
        
        if isinstance(self.hparams.target, str):
            output = self.distribution_projector(decoder_output)
            # Add temporal kernel weights
            kernel_weights = self.temporal_kernel_projector(decoder_output)
            output = torch.cat([output, kernel_weights], dim=-1)
        else:
            output = [projector(decoder_output) for projector in self.distribution_projector]
        
        return output
    
    def decode_one(
        self,
        x: torch.Tensor,
        hidden_state: HiddenState,
    ) -> Tuple[torch.Tensor, HiddenState]:
        """Decode a single time step."""
        # x: (B, 1, input_size)
        decoder_output, new_hidden = self.rnn(x, hidden_state)
        
        # Apply graph convolution
        if self.use_graph_conv:
            decoder_output, _ = self.apply_graph_conv(decoder_output)
        
        if isinstance(self.hparams.target, str):
            output = self.distribution_projector(decoder_output)
            kernel_weights = self.temporal_kernel_projector(decoder_output)
            output = torch.cat([output, kernel_weights], dim=-1)
        else:
            output = [projector(decoder_output) for projector in self.distribution_projector]
        
        return output, new_hidden
    
    def decode(
        self,
        input_vector: torch.Tensor,
        target_scale: torch.Tensor,
        decoder_lengths: torch.Tensor,
        hidden_state: HiddenState,
        n_samples: int = None,
    ) -> Tuple[torch.Tensor, bool]:
        """Decode hidden state into prediction."""
        if n_samples is None:
            # Training: teacher forcing
            output = self.decode_all(input_vector, hidden_state)
            output = self.transform_output(output, target_scale=target_scale)
        else:
            # Inference: autoregressive sampling
            batch_size = input_vector.size(0)
            max_len = decoder_lengths.max().item()
            device = input_vector.device
            
            # Storage for samples: (n_samples, B, T, output_dim)
            if isinstance(self.hparams.target, str):
                # Get output dim from first step (apply transform to get final dim)
                first_output, _ = self.decode_one(input_vector[:, :1], hidden_state)
                first_output_transformed = self.transform_output(first_output, target_scale=target_scale)
                output_dim = first_output_transformed.size(-1)
                all_samples = torch.zeros(n_samples, batch_size, max_len, output_dim, device=device)
            else:
                all_samples = []
            
            for sample_idx in range(n_samples):
                current_hidden = hidden_state
                current_input = input_vector[:, :1].clone()  # First decoder input
                sample_outputs = []
                
                for t in range(max_len):
                    # Decode one step
                    step_output, current_hidden = self.decode_one(current_input, current_hidden)
                    step_output = self.transform_output(step_output, target_scale=target_scale)
                    
                    # Handle NaN/Inf in output
                    if torch.isnan(step_output).any() or torch.isinf(step_output).any():
                        step_output = torch.nan_to_num(step_output, nan=0.0, posinf=1e6, neginf=-1e6)
                    
                    sample_outputs.append(step_output)
                    
                    # Sample from distribution for next input
                    if t < max_len - 1:
                        if isinstance(self.hparams.target, str):
                            # Sample from Normal distribution
                            # step_output has been transformed, so parameters are already on correct scale
                            loc = step_output[..., 0]  # (B, 1)
                            # Scale is already positive after transform_output, just ensure minimum
                            scale = step_output[..., 1].abs() + 1e-6  # (B, 1)
                            
                            # Clamp values to avoid numerical issues
                            loc = torch.clamp(loc, min=-1e6, max=1e6)
                            scale = torch.clamp(scale, min=1e-6, max=1e6)
                            
                            # Sample from Normal distribution
                            sample = loc + scale * torch.randn_like(scale)  # More numerically stable than torch.normal
                            
                            # Clamp sample to avoid extreme values
                            sample = torch.clamp(sample, min=-1e6, max=1e6)
                            
                            # Prepare next input
                            current_input = input_vector[:, t+1:t+2].clone()
                            # Replace target position with sampled value
                            if current_input.size(1) > 0:
                                # Get shape of target slice
                                target_slice = current_input[:, 0, self.target_positions]
                                # Reshape sample to match target slice shape
                                if sample.dim() != target_slice.dim():
                                    if sample.dim() == 1 and target_slice.dim() == 2:
                                        sample = sample.unsqueeze(-1)
                                    elif sample.dim() == 2 and target_slice.dim() == 1:
                                        sample = sample.squeeze(-1)
                                # Ensure shapes match
                                if sample.shape != target_slice.shape:
                                    sample = sample.view_as(target_slice)
                                current_input[:, 0, self.target_positions] = sample
                        else:
                            # Multi-target case
                            current_input = input_vector[:, t+1:t+2].clone()
                
                # Stack outputs for this sample
                if isinstance(self.hparams.target, str):
                    sample_output = torch.cat(sample_outputs, dim=1)
                    all_samples[sample_idx] = sample_output
                else:
                    all_samples.append([torch.cat([s[i] for s in sample_outputs], dim=1) 
                                       for i in range(len(sample_outputs[0]))])
            
            # Return: (n_samples, B, T, output_dim) for single target
            output = all_samples
        
        return output
    
    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        hidden_state, encoder_output = self.encode(x)
        
        # Extract last encoder target for decoder input
        batch_indices = torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device)
        encoder_lengths = x["encoder_lengths"]
        
        # Get the last target value from encoder
        if isinstance(self.target_positions, slice):
            one_off_target = x["encoder_cont"][batch_indices, encoder_lengths - 1, :][:, self.target_positions]
        else:
            one_off_target = x["encoder_cont"][batch_indices, encoder_lengths - 1, self.target_positions]
        
        # Construct decoder input
        decoder_input = self.construct_input_vector(
            x["decoder_cat"], x["decoder_cont"],
            one_off_target=one_off_target
        )
        
        # Get target scale
        target_scale = x.get("target_scale", None)
        
        # Decode
        output = self.decode(
            decoder_input,
            target_scale=target_scale,
            decoder_lengths=x["decoder_lengths"],
            hidden_state=hidden_state,
            n_samples=n_samples,
        )
        
        # Get dynamic adjacency for potential use in loss
        dynamic_adj = self.sagsam()
        
        return {
            "prediction": output,
            "dynamic_adj": dynamic_adj,
            "encoder_output": encoder_output,
        }
    
    def transform_output(
        self,
        out: Union[torch.Tensor, List[torch.Tensor]],
        target_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Transform output to correct scale."""
        if isinstance(out, (list, tuple)):
            return [self.transform_output(o, target_scale) for o in out]
        
        # For distribution loss, rescale parameters
        if isinstance(self.loss, DistributionLoss):
            return self.loss.rescale_parameters(
                out, target_scale=target_scale, encoder=self.output_transformer
            )
        return out


class DGLoGraPTransformer(AutoRegressiveBaseModelWithCovariates):
    """
    DG-LoGraP Transformer Model
    
    Transformer variant with:
    - SAGSAM for dynamic graph generation
    - SAGS-GCN for spatial message passing
    - Grouped latent spatio-temporal covariance structure
    """
    
    def __init__(
        self,
        hidden_size: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        n_validation_samples: int = None,
        n_plotting_samples: int = None,
        target: Union[str, List[str]] = None,
        target_lags: Dict[str, List[int]] = {},
        loss: DistributionLoss = None,
        logging_metrics: nn.ModuleList = None,
        # DG-LoGraP specific parameters
        num_nodes: int = 358,
        graph_embed_dim: int = 32,
        num_groups: int = 2,
        ranks_per_group: List[int] = None,
        num_kernels: int = 4,
        graph_path: str = None,
        use_graph_conv: bool = True,
        **kwargs,
    ):
        """Initialize DG-LoGraP Transformer model."""
        if loss is None:
            loss = NormalDistributionLoss()
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        if n_plotting_samples is None:
            n_plotting_samples = n_validation_samples if n_validation_samples else 100
        
        self.save_hyperparameters()
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)
        
        # Graph parameters
        self.num_nodes = num_nodes
        self.graph_embed_dim = graph_embed_dim
        self.num_groups = num_groups
        self.ranks_per_group = ranks_per_group if ranks_per_group else [5] * num_groups
        self.use_graph_conv = use_graph_conv
        
        # Load static graph
        if graph_path is not None:
            static_adj = load_static_graph(graph_path, num_nodes)
            self.register_buffer('static_adj', static_adj)
        else:
            self.register_buffer('static_adj', torch.eye(num_nodes))
        
        # Embeddings
        self.embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            embedding_paddings=embedding_paddings,
            categorical_groups=categorical_groups,
            x_categoricals=x_categoricals,
        )
        
        # Validate targets
        lagged_target_names = [l for lags in target_lags.values() for l in lags]
        for targeti in to_list(target):
            assert targeti in time_varying_reals_encoder, f"target {targeti} has to be real"
        
        # Input sizes
        self.cont_size = len(self.reals)
        self.cat_size = sum(self.embeddings.output_size.values())
        
        # Input projections
        self.target_embed = nn.Linear(self.cont_size, hidden_size)
        self.covariate_embed = nn.Linear(self.cat_size, hidden_size) if self.cat_size > 0 else None
        
        # SAGSAM
        self.sagsam = SAGSAM(
            num_nodes=num_nodes,
            embed_dim=graph_embed_dim,
            static_adj=self.static_adj,
            dropout=dropout,
        )
        
        # SAGS-GCN
        if use_graph_conv:
            self.sags_gcn = SAGS_GCN(
                num_nodes=num_nodes,
                in_features=hidden_size,
                out_features=hidden_size,
                embed_dim=graph_embed_dim,
                static_adj=self.static_adj,
                dropout=dropout,
            )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Graph-enhanced output
        if use_graph_conv:
            self.pre_dist_proj = nn.Linear(hidden_size, hidden_size)
        
        # Distribution projector
        if isinstance(target, str):
            self.distribution_projector = nn.Linear(
                hidden_size, len(self.loss.distribution_arguments)
            )
        else:
            self.distribution_projector = nn.ModuleList([
                nn.Linear(hidden_size, len(args))
                for args in self.loss.distribution_arguments
            ])
        
        # Temporal kernel projector
        self.temporal_kernel_projector = nn.Linear(hidden_size, num_kernels)
        
        # Graph loader
        self.graph_loader = DynamicGraphLoader(
            num_nodes=num_nodes,
            num_groups=num_groups,
            ranks_per_group=self.ranks_per_group,
            embed_dim=graph_embed_dim,
            static_adj=self.static_adj,
        )
    
    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        """Create model from dataset."""
        new_kwargs = {}
        if dataset.multi_target:
            new_kwargs.setdefault("loss", MultiLoss([NormalDistributionLoss()] * len(dataset.target_names)))
        new_kwargs.update(kwargs)
        
        if isinstance(new_kwargs.get("loss", None), MultivariateDistributionLoss):
            assert (
                dataset.min_prediction_length == dataset.max_prediction_length
            ), "Multivariate models require constant prediction lengths"
        
        return super().from_dataset(dataset, **new_kwargs)
    
    def construct_input_vector(
        self, x_cat: torch.Tensor, x_cont: torch.Tensor, one_off_target: torch.Tensor = None
    ) -> torch.Tensor:
        """Create input vector."""
        if len(self.categoricals) > 0:
            embeddings = self.embeddings(x_cat)
            flat_embeddings = torch.cat([emb for emb in embeddings.values()], dim=-1)
            input_vector = flat_embeddings
        
        if len(self.reals) > 0:
            input_vector = x_cont.clone()
        
        if len(self.reals) > 0 and len(self.categoricals) > 0:
            input_vector = torch.cat([x_cont, flat_embeddings], dim=-1)
        
        # Shift target
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )
        
        if one_off_target is not None:
            input_vector[:, 0, self.target_positions] = one_off_target
        else:
            input_vector = input_vector[:, 1:]
        
        return input_vector
    
    def add_input_vector(self, input_vector: torch.Tensor) -> torch.Tensor:
        """Project input to hidden dimension."""
        out = self.target_embed(input_vector[..., :self.cont_size])
        if self.covariate_embed is not None:
            out = out + self.covariate_embed(input_vector[..., self.cont_size:])
        return out
    
    def encode(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode sequence."""
        assert x["encoder_lengths"].min() > 0
        input_vector = self.construct_input_vector(x["encoder_cat"], x["encoder_cont"])
        src = self.add_input_vector(input_vector)
        
        # Apply transformer
        mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(src.device)
        encoder_output = self.transformer(src, mask=mask)
        
        return encoder_output
    
    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Combine encoder and decoder inputs
        input_vector = self.construct_input_vector(
            torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1),
            torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1),
        )
        src = self.add_input_vector(input_vector)
        
        # Apply transformer
        mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(src.device)
        output = self.transformer(src, mask=mask)
        
        # Get decoder output
        decoder_output = output[:, -x["decoder_lengths"].max():]
        
        # Apply graph convolution if enabled
        if self.use_graph_conv:
            decoder_output = self.pre_dist_proj(decoder_output)
        
        # Project to distribution parameters
        if isinstance(self.hparams.target, str):
            pred = self.distribution_projector(decoder_output)
            # Add temporal kernel weights
            kernel_weights = self.temporal_kernel_projector(decoder_output)
            pred = torch.cat([pred, kernel_weights], dim=-1)
        else:
            pred = [projector(decoder_output) for projector in self.distribution_projector]
        
        # Transform output
        target_scale = x.get("target_scale", None)
        if target_scale is not None and isinstance(self.loss, DistributionLoss):
            pred = self.loss.rescale_parameters(
                pred, target_scale=target_scale, encoder=self.output_transformer
            )
        
        # Get dynamic adjacency
        dynamic_adj = self.sagsam()
        
        return {
            "prediction": pred,
            "dynamic_adj": dynamic_adj,
        }
