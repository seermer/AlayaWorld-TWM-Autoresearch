"""Per-turn WBench prompts: the scene plus the current turn's interaction, nothing carried over."""
from alaya.data.wbench import _THIRD_PERSON_VIEW, _build_interaction_prompt_schedule, _strip_camera

CASE = {
    "settings": {"perspective": "first_person"},
    "environment_prompt": "A cafe table on a sidewalk.",
    "character_prompt": "First-person viewer.",
    "perspective_prompt": "Seated at the table.",
    "interactions": [
        {"type": "event_edit", "action": "Rain begins to fall.", "turn": 1},
        {"type": "navigation", "action": "W", "turn": 2},
        {"type": "perspective_switch", "action": "fp_to_tp", "turn": 3},
        {"type": "event_edit", "action": "The rain stops.", "turn": 4},
        {"type": "perspective_switch", "action": "tp_to_fp", "prompt": "The view returns to the viewer's eyes.", "turn": 5},
    ],
}
SCENE = "A cafe table on a sidewalk. First-person viewer."


def test_each_turn_has_the_scene_and_only_its_own_interaction():
    turn1, turn2, turn3, turn4, turn5 = (_strip_camera(p) for p in _build_interaction_prompt_schedule(CASE))
    assert turn1 == f"{SCENE} Rain begins to fall."
    assert turn2 == SCENE
    assert turn3.startswith(f"{SCENE} The view switches from first-person to third-person")
    assert turn3.endswith(f". {_THIRD_PERSON_VIEW}")
    assert turn4 == f"{SCENE} The rain stops. {_THIRD_PERSON_VIEW}"
    assert turn5 == f"{SCENE} The view returns to the viewer's eyes."


def test_third_person_turns_end_with_the_perspective_sentence():
    case = {
        "settings": {"perspective": "third_person"},
        "environment_prompt": "A meadow.",
        "character_prompt": "A fox walks slowly.",
        "perspective_prompt": "Behind the fox.",
        "interactions": [
            {"type": "navigation", "action": "A", "turn": 1},
            {"type": "event_edit", "action": "A deer appears.", "turn": 2},
        ],
    }
    turn1, turn2 = _build_interaction_prompt_schedule(case)
    assert turn1 == f"A meadow. A fox walks slowly. {_THIRD_PERSON_VIEW} <camera>Behind the fox.</camera>"
    assert turn2 == f"A meadow. A fox walks slowly. A deer appears. {_THIRD_PERSON_VIEW} <camera>Behind the fox.</camera>"
