from copy import deepcopy

from open_gen.data.collate import MultimodalCollator
from open_gen.data.datasets import SyntheticDatasetConfig, SyntheticRobotDataset
from open_gen.model.model_config import config_from_dict
from open_gen.model.opengen1 import OpenGen1Model


def _small_model_cfg() -> dict:
    return {
        "action_dim": 20,
        "proprio_dim": 20,
        "force_dim": 6,
        "image_encoder": {
            "type": "resnet18",
            "pretrained": False,
            "trainable": False,
            "image_size": 32,
            "output_tokens": None,
        },
        "context": {
            "vocab_size": 1024,
            "max_text_tokens": 8,
        },
        "trunk": {
            "width": 128,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "rope_base": 10000.0,
            "dropout": 0.0,
        },
        "action_expert": {
            "width": 128,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "flow_steps": 4,
        },
        "world_model": {
            "enabled": True,
            "latent_dim": 128,
            "future_latent_weight": 1.0,
            "proprio_weight": 0.2,
            "force_weight": 0.2,
        },
    }


def test_model_forward_and_sampling() -> None:
    dataset = SyntheticRobotDataset(
        SyntheticDatasetConfig(
            num_samples=2,
            sequence_length=2,
            action_horizon=2,
            image_size=32,
            seed=3,
        )
    )
    batch = MultimodalCollator(image_size=32, encoder_type="resnet18", train=False)([dataset[0], dataset[1]])
    model = OpenGen1Model(config_from_dict(_small_model_cfg()))
    outputs = model(batch)
    assert "loss" in outputs
    assert outputs["loss"].ndim == 0
    sampled = model.sample_actions(batch, action_horizon=2)
    assert sampled.shape == (2, 2, 20)


def test_streaming_append_matches_shapes() -> None:
    dataset = SyntheticRobotDataset(
        SyntheticDatasetConfig(
            num_samples=1,
            sequence_length=2,
            action_horizon=2,
            image_size=32,
            seed=5,
        )
    )
    item = dataset[0]
    first = deepcopy(item)
    second = deepcopy(item)
    first.left_wrist_image = first.left_wrist_image[:1]
    first.right_wrist_image = first.right_wrist_image[:1]
    first.head_image = first.head_image[:1] if first.head_image is not None else None
    first.proprio = first.proprio[:1] if first.proprio is not None else None
    first.force = first.force[:1] if first.force is not None else None
    first.past_action = first.past_action[:1] if first.past_action is not None else None
    second.left_wrist_image = second.left_wrist_image[1:]
    second.right_wrist_image = second.right_wrist_image[1:]
    second.head_image = second.head_image[1:] if second.head_image is not None else None
    second.proprio = second.proprio[1:] if second.proprio is not None else None
    second.force = second.force[1:] if second.force is not None else None
    second.past_action = second.past_action[1:] if second.past_action is not None else None

    collator = MultimodalCollator(image_size=32, encoder_type="resnet18", train=False)
    batch1 = collator([first])
    batch2 = collator([second])
    model = OpenGen1Model(config_from_dict(_small_model_cfg()))
    state = model.append_observation(None, batch1, include_context=True)
    state = model.append_observation(state, batch2, include_context=False)
    sampled = model.decode_from_state(state, action_horizon=2)
    assert sampled.shape == (1, 2, 20)
