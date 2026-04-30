import pytest

from open_gen.data.curriculum import CurriculumScheduler, parse_curriculum_config
from open_gen.data.sampler import BucketedTokenBudgetBatchSampler, make_buckets


def _stage_payload(name, **kwargs):
    base = dict(
        name=name,
        max_past_frames=4,
        min_past_frames=1,
        frame_sampling="max",
        token_budget=4096,
    )
    base.update(kwargs)
    return base


def test_make_buckets_partitions_range():
    buckets = make_buckets([1, 4, 16])
    assert [(b.min_T, b.max_T) for b in buckets] == [(1, 1), (2, 4), (5, 16)]


def test_curriculum_picks_stage_by_step():
    cfg = parse_curriculum_config(
        {
            "enabled": True,
            "unit": "step",
            "stages": [
                _stage_payload("a", until_step=10, max_past_frames=1),
                _stage_payload("b", until_step=20, max_past_frames=8),
                _stage_payload("c", max_past_frames=64),
            ],
        }
    )
    scheduler = CurriculumScheduler(cfg, fallback=_default_fallback())
    assert scheduler.stage_for(step=0, epoch=0).name == "a"
    assert scheduler.stage_for(step=10, epoch=0).name == "b"
    assert scheduler.stage_for(step=20, epoch=0).name == "c"
    assert scheduler.stage_for(step=999, epoch=0).name == "c"


def _default_fallback():
    from open_gen.data.curriculum import CurriculumStage

    return CurriculumStage(name="fallback", max_past_frames=4, min_past_frames=1)


def test_sampler_yields_uniform_T_per_batch():
    cfg = parse_curriculum_config(
        {
            "enabled": True,
            "unit": "epoch",
            "stages": [_stage_payload("only", frame_sampling="uniform", token_budget=2048)],
        }
    )
    stage = cfg.stages[0]
    sampler = BucketedTokenBudgetBatchSampler(
        native_lengths=[4] * 16,
        bucket_boundaries=[1, 4],
        tokens_per_frame=32,
        context_tokens=24,
        stage=stage,
        seed=11,
    )
    sampler.set_epoch(0)
    for batch in sampler:
        targets = {t for _, t in batch}
        assert len(targets) == 1
        target = next(iter(targets))
        assert 1 <= target <= 4


def test_sampler_ddp_shards_match_global_shape():
    from open_gen.data.curriculum import CurriculumStage

    stage = CurriculumStage(name="s", max_past_frames=4, min_past_frames=1, frame_sampling="max", token_budget=2048)
    common = dict(
        native_lengths=[4] * 32,
        bucket_boundaries=[4],
        tokens_per_frame=32,
        context_tokens=24,
        stage=stage,
        seed=7,
    )
    a = BucketedTokenBudgetBatchSampler(world_size=2, rank=0, **common)
    b = BucketedTokenBudgetBatchSampler(world_size=2, rank=1, **common)
    a.set_epoch(0)
    b.set_epoch(0)
    a_batches = list(iter(a))
    b_batches = list(iter(b))
    assert len(a_batches) == len(b_batches)
    for a_batch, b_batch in zip(a_batches, b_batches):
        assert {t for _, t in a_batch} == {t for _, t in b_batch}
        assert len(a_batch) == len(b_batch)
        a_idx = {idx for idx, _ in a_batch}
        b_idx = {idx for idx, _ in b_batch}
        assert a_idx.isdisjoint(b_idx)


def test_sampler_skips_buckets_outside_stage_range():
    from open_gen.data.curriculum import CurriculumStage

    stage = CurriculumStage(name="single", max_past_frames=1, min_past_frames=1, frame_sampling="max", token_budget=1024)
    sampler = BucketedTokenBudgetBatchSampler(
        native_lengths=[8] * 16,
        bucket_boundaries=[1, 4, 16],
        tokens_per_frame=32,
        context_tokens=24,
        stage=stage,
        seed=3,
    )
    sampler.set_epoch(0)
    for batch in sampler:
        assert all(t == 1 for _, t in batch)


def test_sampler_set_stage_changes_T():
    from open_gen.data.curriculum import CurriculumStage

    sampler = BucketedTokenBudgetBatchSampler(
        native_lengths=[4] * 64,
        bucket_boundaries=[1, 4],
        tokens_per_frame=32,
        context_tokens=24,
        stage=CurriculumStage(name="a", max_past_frames=1, min_past_frames=1, frame_sampling="max", token_budget=1024),
        seed=5,
    )
    sampler.set_epoch(0)
    first_T = {t for batch in sampler for _, t in batch}
    assert first_T == {1}
    sampler.set_stage(
        CurriculumStage(name="b", max_past_frames=4, min_past_frames=4, frame_sampling="max", token_budget=2048)
    )
    sampler.set_epoch(0)
    second_T = {t for batch in sampler for _, t in batch}
    assert second_T == {4}


def test_sampler_rejects_invalid_rank():
    from open_gen.data.curriculum import CurriculumStage

    with pytest.raises(ValueError):
        BucketedTokenBudgetBatchSampler(
            native_lengths=[1],
            bucket_boundaries=[1],
            tokens_per_frame=1,
            context_tokens=0,
            stage=CurriculumStage(name="x", max_past_frames=1, min_past_frames=1, frame_sampling="max", token_budget=1),
            world_size=2,
            rank=2,
        )
