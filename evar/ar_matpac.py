"""MATPAC / MATPAC++ wrapper for EVAR.

MATPAC: Masked Latent Prediction and Classification for Self-Supervised Audio
Representation Learning (https://arxiv.org/abs/2502.12031)
MATPAC++: Enhanced Masked Latent Prediction (https://arxiv.org/abs/2508.12709)

Unlike ar_m2d.py, this wrapper does NOT compute/normalize the log-mel itself: MATPAC's
inference model (`matpac.model.get_matpac`) takes raw audio [B, samples] and internally
produces the log-mel, applies its own fixed standardization (lms_mean/std), runs the encoder,
and time-pools — returning a clip embedding [B, 3840]. So EVAR must feed raw audio and must
NOT re-normalize.
"""

from evar.ar_base import BaseAudioRepr

try:
    from matpac.model import get_matpac
except Exception as e:  # matpac must be importable in the active env (its pixi env)
    get_matpac = None
    _IMPORT_ERROR = e


class AR_MATPAC(BaseAudioRepr):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        if get_matpac is None:
            raise ImportError(
                "Could not import matpac.model.get_matpac — run EVAR inside the matpac "
                f"pixi env (models/matpac/inference_matpac). Original error: {_IMPORT_ERROR}"
            )
        self.model = get_matpac(
            checkpoint_path=cfg.weight_file,
            inference_type='precise',      # paper setting (no padding)
            pull_time_dimension=True,      # mean-pool time -> [B, 3840]
            concat_freq=True,
        )
        self.model.eval()

    def precompute(self, device, data_loader):
        # No-op: MATPAC standardizes internally with its own fixed stats.
        pass

    def forward(self, batch_audio):
        # batch_audio: [B, samples] raw waveform at cfg.sample_rate (16 kHz)
        emb, _ = self.model(batch_audio)   # [B, 3840]
        return emb
