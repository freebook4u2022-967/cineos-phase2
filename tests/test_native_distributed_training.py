import pytest

from cineos.native_image.distributed_training import DistributedTrainingCoordinator
from cineos.native_image.gpu_scheduler import GPUWorker


def _workers():
    return (
        GPUWorker("a", "H100", 80, 0.2),
        GPUWorker("b", "H100", 80, 0.1),
        GPUWorker("c", "A100", 40, 0.3),
    )


def test_distributed_plan_selects_requested_gpu_group():
    plan = DistributedTrainingCoordinator().create_plan(
        _workers(), minimum_vram_gb=64, requested_world_size=2
    )
    assert plan.world_size == 2
    assert [worker.worker_id for worker in plan.workers] == ["b", "a"]


def test_distributed_plan_rejects_insufficient_gpu_group():
    with pytest.raises(RuntimeError, match="insufficient eligible"):
        DistributedTrainingCoordinator().create_plan(
            _workers(), minimum_vram_gb=64, requested_world_size=3
        )


def test_dataset_shards_cover_samples_without_overlap():
    shards = DistributedTrainingCoordinator().shard_dataset(10, 3)
    assert [(item.start_index, item.end_index) for item in shards] == [
        (0, 4),
        (4, 7),
        (7, 10),
    ]


def test_gradient_average_matches_data_parallel_reduction():
    averaged = DistributedTrainingCoordinator().average_gradients(
        ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    )
    assert averaged == pytest.approx((3.0, 4.0))
