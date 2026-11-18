"""Build optional per-clip reports for other-animal imitations."""
import json
from collections import Counter, defaultdict
from pathlib import Path


def build_other_items_report(
    ids,
    animals,
    unedited_top_classes,
    after_top_classes,
    *,
    chicken_prompt_scores=None,
    gate_open=None,
):
    """Build clip rows and per-animal summaries from top-class observations."""
    ids = [str(value) for value in ids]
    animals = [str(value) for value in animals]
    unedited_top_classes = [str(value) for value in unedited_top_classes]
    after_top_classes = {
        str(arm): [str(value) for value in classes]
        for arm, classes in after_top_classes.items()
    }
    count = len(ids)
    if len(set(ids)) != count:
        raise ValueError("other-imitation clip ids must be unique")
    if not (len(animals) == len(unedited_top_classes) == count):
        raise ValueError("clip ids, animals, and unedited classes must have equal lengths")
    if not after_top_classes:
        raise ValueError("at least one arm of after-top classes is required")
    for arm, classes in after_top_classes.items():
        if len(classes) != count:
            raise ValueError(f"{arm} after-top classes must have {count} rows")

    chicken_prompt_scores = chicken_prompt_scores or {}
    gate_open = gate_open or {}
    for arm, score_pair in chicken_prompt_scores.items():
        if arm not in after_top_classes:
            raise ValueError(f"scores supplied for unknown arm {arm}")
        before_scores, after_scores = score_pair
        if len(before_scores) != count or len(after_scores) != count:
            raise ValueError(f"{arm} chicken-prompt scores must have {count} rows")
    for arm, values in gate_open.items():
        if arm not in after_top_classes:
            raise ValueError(f"gate state supplied for unknown arm {arm}")
        if len(values) != count:
            raise ValueError(f"{arm} gate states must have {count} rows")

    rows = []
    changed_count_by_arm = {arm: 0 for arm in after_top_classes}
    by_animal = defaultdict(lambda: {
        "clips": 0,
        "arms": {
            arm: {"changed_count": 0, "after_labels": Counter()}
            for arm in after_top_classes
        },
    })

    for index, (item_id, animal, before_class) in enumerate(
        zip(ids, animals, unedited_top_classes)
    ):
        arm_rows = {}
        for arm, after_classes in after_top_classes.items():
            after_class = after_classes[index]
            changed = after_class != before_class
            changed_count_by_arm[arm] += int(changed)
            arm_row = {"top_class": after_class, "changed": changed}
            if arm in chicken_prompt_scores:
                before_scores, after_scores = chicken_prompt_scores[arm]
                arm_row["chicken_prompt_score_before"] = float(before_scores[index])
                arm_row["chicken_prompt_score_after"] = float(after_scores[index])
            if arm in gate_open:
                arm_row["gate_open"] = bool(gate_open[arm][index])
            arm_rows[arm] = arm_row

            animal_summary = by_animal[animal]["arms"][arm]
            animal_summary["changed_count"] += int(changed)
            animal_summary["after_labels"][after_class] += 1

        by_animal[animal]["clips"] += 1
        rows.append({
            "id": item_id,
            "animal": animal,
            "unedited_top_class": before_class,
            "arms": arm_rows,
        })

    summary_by_animal = {}
    for animal in sorted(by_animal):
        summary = by_animal[animal]
        summary_by_animal[animal] = {
            "clips": summary["clips"],
            "arms": {
                arm: {
                    "changed_count": details["changed_count"],
                    "after_labels": dict(sorted(details["after_labels"].items())),
                }
                for arm, details in summary["arms"].items()
            },
        }

    return {
        "items": rows,
        "changed_count_by_arm": changed_count_by_arm,
        "summary_by_animal": summary_by_animal,
    }


def write_other_items_report(path, *args, **kwargs):
    """Write a JSON report and return the exact serialized report data."""
    report = build_other_items_report(*args, **kwargs)
    Path(path).write_text(json.dumps(report, indent=2) + "\n")
    return report
