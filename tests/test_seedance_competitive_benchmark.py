from cineos.benchmarks.seedance_competitive import seedance_competitive_suite


def test_seedance_competitive_suite_covers_required_difficult_cases():
    suite = seedance_competitive_suite()

    assert suite.target_platform == "gpu"
    assert suite.metadata["real_inference"] is True
    assert suite.metadata["minimum_connected_shots"] == 5
    assert suite.metadata["maximum_connected_shots"] == 10
    assert len(suite.cases) == 10
    assert all(case.mandatory and case.slow for case in suite.cases)
    assert all(case.hardware_requirements["gpu"] is True for case in suite.cases)
    assert all(
        case.hardware_requirements["real_inference"] is True for case in suite.cases
    )

    case_ids = {case.case_id for case in suite.cases}
    assert case_ids == {
        "competitive-identity-closeup",
        "competitive-two-character-dialogue",
        "competitive-hands-object",
        "competitive-walk-run",
        "competitive-fast-camera",
        "competitive-lighting-transition",
        "competitive-physics-weather",
        "competitive-scene-boundary",
        "competitive-qc-rerender",
        "competitive-connected-film",
    }


def test_connected_film_gate_requires_complete_film_signals():
    suite = seedance_competitive_suite()
    connected = next(
        case for case in suite.cases if case.case_id.endswith("connected-film")
    )

    assert connected.renderer_requirements == (
        "identity_lock",
        "scene_memory",
        "automatic_qc",
        "audio",
        "film_assembly",
    )
    assert connected.expected_outputs == (
        "report.json",
        "render_receipt.json",
        "output.mp4",
    )
    assert connected.validation_thresholds["execution_success"] == 1.0
    assert connected.validation_thresholds["render_completion_rate"] == 1.0
    assert connected.validation_thresholds["final_assembly_success"] == 1.0
    assert connected.validation_thresholds["identity_score"] >= 0.88
    assert connected.validation_thresholds["temporal_stability"] >= 0.84


def test_qc_case_requires_reject_rerender_recovery_path():
    suite = seedance_competitive_suite()
    qc = next(case for case in suite.cases if case.case_id.endswith("qc-rerender"))

    assert qc.renderer_requirements == ("automatic_qc", "rerender")
    assert qc.validation_thresholds == {
        "validation_pass_rate": 1.0,
        "render_completion_rate": 1.0,
    }


def test_competitive_suite_hash_is_stable_and_foundation_provenance_is_explicit():
    first = seedance_competitive_suite()
    second = seedance_competitive_suite()

    assert first.content_hash == second.content_hash
    assert (
        first.metadata["foundation_origin_required"] == "external_pretrained_foundation"
    )
    assert "conditioning" in first.metadata["cineos_owned_layers"]
    assert "automatic_qc" in first.metadata["cineos_owned_layers"]
