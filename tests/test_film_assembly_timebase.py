from __future__ import annotations

from cineos.film.assembly import _explicit_trim_filter


def test_explicit_hard_cut_normalizes_each_decoded_timebase() -> None:
    graph = _explicit_trim_filter([1.0, 2.5, 3.0])

    assert graph.count("settb=AVTB,setpts=PTS-STARTPTS") == 3
    assert "[v0][v1][v2]concat=n=3:v=1:a=0[filmv]" in graph


def test_crossfade_normalizes_timebase_without_changing_offsets() -> None:
    graph = _explicit_trim_filter([2.0, 3.0, 4.0], crossfade=0.5)

    assert graph.count("settb=AVTB,setpts=PTS-STARTPTS") == 3
    assert "duration=0.500000:offset=1.500000[xf1]" in graph
    assert "duration=0.500000:offset=4.000000[filmv]" in graph
