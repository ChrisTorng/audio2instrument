import pytest

from audio2instrument.drum_poc import custom_key_map, drum_layer_ranges, infer_layer


def test_custom_key_map_is_stable_and_non_conflicting() -> None:
    assert custom_key_map([46, 42, 44], 50) == {42: 50, 44: 51, 46: 52}


def test_velocity_layer_boundaries_cover_midi_range() -> None:
    assert drum_layer_ranges(3) == ((1, 48), (49, 92), (93, 127))
    assert infer_layer(0.1, [0.1, 0.2, 0.3], 3) == 2
    with pytest.raises(ValueError):
        drum_layer_ranges(4)
