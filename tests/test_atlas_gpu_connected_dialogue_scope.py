import json
from pathlib import Path

from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.gpu_connected_benchmark import run_connected_gpu_benchmark
from test_atlas_gpu_connected_benchmark import _receipt, _request


def _executor(request, profile, *, output_dir):
    assert profile is WAN22_TI2V_5B_PROFILE
    return _receipt(request, Path(output_dir))


def test_connected_benchmark_captures_dialogue_scope_from_rendered_requests(tmp_path):
    requests = [_request(index) for index in range(5)]
    requests[1].performance["dialogue_timing"] = [
        {
            "speaker_id": "lead",
            "start_seconds": 0.25,
            "end_seconds": 1.25,
        }
    ]
    requests[1].refresh_hash()

    receipt = run_connected_gpu_benchmark(
        "dialogue-scope",
        requests,
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        shot_executor=_executor,
    )

    assert receipt.dialogue_shot_ids == ("shot-1",)
    payload = json.loads(Path(receipt.manifest_path).read_text(encoding="utf-8"))
    assert payload["schema"] == "cineos-gpu-connected-benchmark/0.3"
    assert payload["dialogue_scope_declared"] is True
    assert payload["dialogue_shot_ids"] == ["shot-1"]


def test_connected_benchmark_declares_empty_dialogue_scope_for_silent_run(tmp_path):
    requests = [_request(index) for index in range(5)]

    receipt = run_connected_gpu_benchmark(
        "silent-scope",
        requests,
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        shot_executor=_executor,
    )

    assert receipt.dialogue_shot_ids == ()
    payload = json.loads(Path(receipt.manifest_path).read_text(encoding="utf-8"))
    assert payload["dialogue_scope_declared"] is True
    assert payload["dialogue_shot_ids"] == []
