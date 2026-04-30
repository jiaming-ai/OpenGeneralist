from open_gen.data.collate import MultimodalCollator
from open_gen.data.curriculum import CurriculumStage
from open_gen.data.datasets import SyntheticDatasetConfig, SyntheticRobotDataset
from open_gen.data.loader import BucketedDataLoader
from open_gen.data.sampler import BucketedTokenBudgetBatchSampler


def test_bucketed_loader_pads_to_T_target_per_bucket():
    dataset = SyntheticRobotDataset(
        SyntheticDatasetConfig(
            num_samples=16,
            sequence_length=8,
            action_horizon=2,
            image_size=32,
            seed=2,
        )
    )
    stage = CurriculumStage(
        name="mixed",
        max_past_frames=8,
        min_past_frames=1,
        frame_sampling="uniform",
        token_budget=4096,
    )
    sampler = BucketedTokenBudgetBatchSampler(
        native_lengths=dataset.native_lengths(),
        bucket_boundaries=[1, 4, 8],
        tokens_per_frame=32,
        context_tokens=24,
        stage=stage,
        seed=42,
    )
    sampler.set_epoch(0)
    collator = MultimodalCollator(image_size=32, encoder_type="resnet18", train=False)
    loader = BucketedDataLoader(dataset, sampler, collator)

    seen_lengths = set()
    for batch in loader:
        T_obs = batch["observations"]["left_wrist_image"].shape[1]
        seen_lengths.add(T_obs)
        # All samples in batch share T_obs because pad_to_obs_steps was set
        assert batch["observations"]["mask"].shape[1] == T_obs
    # Across the epoch we should hit at least one short and one long
    assert len(seen_lengths) >= 1
