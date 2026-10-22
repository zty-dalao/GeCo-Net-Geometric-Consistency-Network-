"""Quaternion-conditioned view weighting around an existing aggregator.

The wrapper does not re-implement any aggregator network.  It keeps the
project's ``adafusor``/``localfusor``/``meanfusor``/``varfusor`` as the feature
path and only adds a per-view bias to the weight logits, so the original
behaviour is recovered exactly when that bias is zero.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pose import (
    pose_vec_to_quaternion,
    relative_quaternion,
    rotation_angle_degrees,
)

_SUPPORTED_EMBEDDINGS = ("absolute_q", "relative_q")


class QuaternionAwareAggregator(nn.Module):
    def __init__(self, base_aggregator, embedding="relative_q", use_translation=False,
                 embedding_dim=16):
        super().__init__()
        if base_aggregator is None:
            raise ValueError("QuaternionAwareAggregator requires a base aggregator")
        for attribute in ("input_fc", "weight_fc", "output_fc"):
            if not hasattr(base_aggregator, attribute):
                raise ValueError(
                    "base_aggregator must be one of the learned view aggregators "
                    f"(missing {attribute})"
                )
        self.embedding = str(embedding)
        if self.embedding not in _SUPPORTED_EMBEDDINGS:
            raise ValueError(f"quaternion embedding must be one of {_SUPPORTED_EMBEDDINGS}")
        self.use_translation = bool(use_translation)
        self.latent_size = int(base_aggregator.latent_size)

        pose_dim = 4 + (3 if self.use_translation else 0)
        self.pose_embed = nn.Sequential(
            nn.Linear(pose_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
        )
        self.pose_to_weight_logits = nn.Linear(
            embedding_dim, self.latent_size, bias=False,
        )
        # Start exactly at the base aggregator's weights.
        nn.init.zeros_(self.pose_to_weight_logits.weight)

        # The base aggregator is already registered by the main model, so keep
        # only a non-registered reference here to avoid duplicate state_dict
        # entries and duplicate optimizer parameters.
        object.__setattr__(self, "_base", base_aggregator)
        self.last_metrics = {}

    @property
    def base(self):
        return self._base

    def pose_embedding(self, poses):
        """Return (pose feature, quaternion) for this view batch."""
        if self.embedding == "absolute_q":
            quaternion = pose_vec_to_quaternion(poses)
        else:
            quaternion = relative_quaternion(poses)
        if not self.use_translation:
            return quaternion, quaternion
        translation = poses[..., :3] - poses[0:1, :3]
        scale = (
            torch.linalg.norm(translation.detach(), dim=-1, keepdim=True)
            .mean()
            .clamp_min(1.0)
        )
        return torch.cat((quaternion, translation / scale), dim=-1), quaternion

    def forward(self, latent, poses):
        if poses is None:
            raise ValueError("QuaternionAwareAggregator requires poses")
        if latent.ndim != 3 or latent.shape[1] != self.latent_size:
            raise ValueError(
                f"Expected latent with shape [views, {self.latent_size}, points], "
                f"got {tuple(latent.shape)}"
            )
        if poses.shape[0] != latent.shape[0]:
            raise ValueError(
                f"Expected one pose per view ({latent.shape[0]}), got {poses.shape[0]}"
            )

        pose_input, quaternion = self.pose_embedding(poses)
        view_bias = self.pose_to_weight_logits(self.pose_embed(pose_input))
        output = self._base(latent, view_bias=view_bias)
        self.last_metrics = self._view_metrics(quaternion)
        return output

    def _view_metrics(self, quaternion):
        metrics = {
            "relative_pose_angle_deg_mean": rotation_angle_degrees(quaternion).mean(),
        }
        weights = getattr(self._base, "last_view_weights", None)
        if weights is not None and weights.shape[0] > 1:
            weights = weights.reshape(weights.shape[0], -1).clamp_min(1e-8)
            entropy = -(weights * weights.log()).sum(dim=0)
            metrics["view_weight_entropy_norm"] = (
                entropy.mean() / math.log(weights.shape[0])
            )
            metrics["view_weight_max_mean"] = weights.max(dim=0).values.mean()
        return {key: value.detach() for key, value in metrics.items()}
