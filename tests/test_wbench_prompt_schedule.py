"""Per-turn WBench prompts: the scene plus the current turn's interaction, nothing carried over."""
from alaya.data.wbench import _build_interaction_prompt_schedule, _strip_camera

CASE = {
    "environment_prompt": "A cafe table on a sidewalk.",
    "character_prompt": "First-person viewer.",
    "perspective_prompt": "Seated at the table.",
    "interactions": [
        {"type": "event_edit", "action": "Rain begins to fall.", "turn": 1},
        {"type": "navigation", "action": "W", "turn": 2},
        {"type": "perspective_switch", "action": "fp_to_tp", "turn": 3},
        {"type": "event_edit", "action": "The rain stops.", "turn": 4},
    ],
}
SCENE = "First-person viewer. A cafe table on a sidewalk."


def test_each_turn_has_the_scene_and_only_its_own_interaction():
    turn1, turn2, turn3, turn4 = (_strip_camera(p) for p in _build_interaction_prompt_schedule(CASE))
    assert turn1 == f"{SCENE} Rain begins to fall."
    assert turn2 == SCENE
    assert turn3.startswith(f"{SCENE} The view switches from first-person to third-person")
    assert turn4 == f"{SCENE} The rain stops."
