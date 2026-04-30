import torch

from open_gen.data.collate import MultimodalCollator
from open_gen.data.datasets import SyntheticDatasetConfig, SyntheticRobotDataset


def test_collator_handles_optional_modalities() -> None:
    dataset = SyntheticRobotDataset(
        SyntheticDatasetConfig(
            num_samples=2,
            sequence_length=3,
            action_horizon=2,
            image_size=32,
            include_head=False,
            include_proprio=False,
            include_force=False,
            include_past_action=False,
            include_future_targets=False,
        )
    )
    batch = MultimodalCollator(image_size=32, encoder_type="resnet18", train=False)([dataset[0], dataset[1]])
    assert batch["observations"]["left_wrist_image"].shape[:2] == (2, 3)
    assert batch["observations"]["head_image"].shape[1] == 3
    assert batch["observations"]["head_mask"].sum().item() == 0
    assert batch["past_action"]["mask"].sum().item() == 0
    assert batch["action_target"]["value"].shape == (2, 2, 20)
    assert batch["future"]["mask"].sum().item() == 0
    assert isinstance(batch["observations"]["left_wrist_image"], torch.Tensor)
