import json
from pathlib import Path
import warnings

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class DentalVolumeDataset(Dataset):
    """Load only dental ground-truth volumes for decoder pretraining.

    The main project reads every projection together with the volume. Decoder
    pretraining does not use projections, so avoiding them saves both host and
    device memory. SimpleITK returns volumes as [Z, Y, X]; the main model's
    decoder internally predicts [X, Y, Z] and transposes X/Z afterwards, so we
    perform the same conversion before pretraining.
    """

    def __init__(
        self,
        data_root: str,
        split_file: str,
        split: str = "train",
        clamp_min: float = 0.0,
        clamp_max: float = 0.09009,
        limit: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        with Path(split_file).open("r", encoding="utf-8") as handle:
            split_data = json.load(handle)
        requested_subjects = list(split_data[split])
        self.subjects = [
            subject
            for subject in requested_subjects
            if (self.data_root / subject / "gt_volume.nii.gz").is_file()
        ]
        missing_subjects = sorted(set(requested_subjects) - set(self.subjects))
        if missing_subjects:
            warnings.warn(
                f"Skipping {len(missing_subjects)} {split} subjects whose gt_volume.nii.gz "
                f"is missing under {self.data_root}: {', '.join(missing_subjects)}",
                stacklevel=2,
            )
        if limit is not None:
            self.subjects = self.subjects[:limit]
        if not self.subjects:
            raise RuntimeError(f"No usable subjects were found for split '{split}'.")

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        subject = self.subjects[index]
        volume_path = self.data_root / subject / "gt_volume.nii.gz"
        volume_zyx = sitk.GetArrayFromImage(sitk.ReadImage(str(volume_path)))
        volume_zyx = np.asarray(volume_zyx, dtype=np.float32)
        volume_zyx = np.clip(volume_zyx, self.clamp_min, self.clamp_max)

        # Align pretraining with model.forward(), whose decoder works in XYZ
        # order before model.py transposes its output for ITK display/storage.
        volume_xyz = np.ascontiguousarray(volume_zyx.transpose(2, 1, 0))
        volume = torch.from_numpy(volume_xyz).unsqueeze(0)  # [1, X, Y, Z]
        return {"volume": volume, "subject": subject}
