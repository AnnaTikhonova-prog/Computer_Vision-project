#!/usr/bin/env python3
"""Build and maintain a human review queue for perceptual-hash candidates.

This utility never changes train.csv, val.csv, test.csv, or class_mapping.json.
It only records human decisions in near_duplicate_review.csv and produces an
HTML contact sheet with both images of each candidate displayed side by side.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ("duplicate", "near-duplicate", "different images", "uncertain")
REVIEW_COLUMNS = [
    "pair_id", "image_id_a", "class_name_a", "split_a", "relative_path_a",
    "image_id_b", "class_name_b", "split_b", "relative_path_b",
    "hamming_distance", "same_class", "decision", "review_notes",
]


def pair_id(image_id_a: str, image_id_b: str) -> str:
    return f"{image_id_a}__{image_id_b}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def source_rows(split_dir: Path) -> tuple[list[dict[str, object]], dict[str, dict[str, str]]]:
    candidates_path = split_dir / "near_duplicate_candidates.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))["candidates"]
    image_info: dict[str, dict[str, str]] = {}
    for split in ("train", "val", "test"):
        for row in read_csv(split_dir / f"{split}.csv"):
            image_id = row["image_id"]
            if image_id in image_info:
                raise ValueError(f"image_id appears in multiple manifests: {image_id}")
            image_info[image_id] = {**row, "split": split}
    return candidates, image_info


def create_or_load_review(split_dir: Path) -> list[dict[str, str]]:
    candidates, image_info = source_rows(split_dir)
    review_path = split_dir / "near_duplicate_review.csv"
    existing = {row["pair_id"]: row for row in read_csv(review_path)} if review_path.exists() else {}
    expected_ids = {pair_id(candidate["image_id_a"], candidate["image_id_b"]) for candidate in candidates}
    unexpected = sorted(set(existing) - expected_ids)
    if unexpected:
        raise ValueError(
            "Review CSV no longer matches the candidate report; refusing to discard existing decisions. "
            f"Unexpected pair_id: {unexpected[0]}"
        )
    rows: list[dict[str, str]] = []
    for candidate in candidates:
        first = image_info.get(candidate["image_id_a"])
        second = image_info.get(candidate["image_id_b"])
        if first is None or second is None:
            raise ValueError(f"Candidate references an image absent from frozen manifests: {candidate}")
        identifier = pair_id(candidate["image_id_a"], candidate["image_id_b"])
        saved = existing.get(identifier, {})
        decision = saved.get("decision", "uncertain")
        if decision not in DECISIONS:
            raise ValueError(f"Invalid saved decision for {identifier}: {decision!r}")
        rows.append({
            "pair_id": identifier,
            "image_id_a": first["image_id"], "class_name_a": first["class_name"], "split_a": first["split"], "relative_path_a": first["relative_path"],
            "image_id_b": second["image_id"], "class_name_b": second["class_name"], "split_b": second["split"], "relative_path_b": second["relative_path"],
            "hamming_distance": str(candidate["hamming_distance"]), "same_class": str(candidate["same_class"]),
            "decision": decision, "review_notes": saved.get("review_notes", ""),
        })
    return rows


def write_review(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, rows: list[dict[str, str]]) -> None:
    cards: list[str] = []
    for row in rows:
        def sample(side: str) -> str:
            image_id = html.escape(row[f"image_id_{side}"])
            class_name = html.escape(row[f"class_name_{side}"])
            split = html.escape(row[f"split_{side}"])
            relative = html.escape("../data/oxford-iiit-pet/" + row[f"relative_path_{side}"], quote=True)
            return f'''<figure><a href="{relative}" target="_blank"><img src="{relative}" loading="lazy" alt="{image_id}"></a>
<figcaption><b>{image_id}</b><br>{class_name}<br><span class="split">{split}</span></figcaption></figure>'''
        cards.append(f'''<article id="{html.escape(row["pair_id"])}"><header><b>{html.escape(row["pair_id"])}</b>
distance: {html.escape(row["hamming_distance"])} · same class: {html.escape(row["same_class"])} · decision: <mark>{html.escape(row["decision"])}</mark></header>
<div class="pair">{sample("a")}{sample("b")}</div><p>Notes: {html.escape(row["review_notes"]) or "—"}</p></article>''')
    content = "\n".join(cards)
    path.write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Near-duplicate review</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;background:#f5f5f5}}article{{background:#fff;margin:18px 0;padding:14px;border-radius:8px}}header{{margin-bottom:10px}}.pair{{display:flex;gap:18px;flex-wrap:wrap}}figure{{margin:0;width:min(430px,100%)}}img{{width:100%;max-height:330px;object-fit:contain;background:#eee}}figcaption{{padding-top:5px}}.split,mark{{padding:2px 6px;border-radius:4px;background:#e7edf8}}p{{font-size:.9em}}</style>
<h1>Perceptual-hash review queue</h1><p>Integrity review only; do not use it to select model settings. Decisions are persisted in <code>near_duplicate_review.csv</code>. Edit that CSV or use the CLI command in README, then rebuild this page.</p>{content}</html>''', encoding="utf-8")


def status(rows: list[dict[str, str]]) -> dict[str, object]:
    # Treat both explicit duplicate labels conservatively: neither may cross a
    # frozen split without a separately versioned repair of the split.
    confirmed = [row for row in rows if row["decision"] in {"duplicate", "near-duplicate"}]
    crossings = [row for row in confirmed if row["split_a"] != row["split_b"]]
    unresolved_crossings = [row for row in rows if row["decision"] == "uncertain" and row["split_a"] != row["split_b"]]
    return {
        "total_pairs": len(rows),
        "decisions": dict(sorted(Counter(row["decision"] for row in rows).items())),
        "confirmed_pairs": len(confirmed),
        "confirmed_cross_split_pairs": crossings,
        "unresolved_cross_split_candidates": unresolved_crossings,
        "frozen_manifests_modified": False,
        "next_action_for_crossings": (
            "Do not edit frozen CSVs. Create a separately approved split version "
            "with --confirmed-groups after review."
        ) if crossings else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "splits")
    parser.add_argument("--set-decision", nargs=2, metavar=("PAIR_ID", "DECISION"), help="Persist one reviewed decision")
    parser.add_argument("--notes", default="", help="Optional note used with --set-decision")
    args = parser.parse_args()
    rows = create_or_load_review(args.split_dir)
    if args.set_decision:
        target, decision = args.set_decision
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        matching = [row for row in rows if row["pair_id"] == target]
        if len(matching) != 1:
            raise ValueError(f"Unknown pair_id: {target}")
        matching[0]["decision"] = decision
        matching[0]["review_notes"] = args.notes
    write_review(args.split_dir / "near_duplicate_review.csv", rows)
    write_html(args.split_dir / "near_duplicate_review.html", rows)
    report = status(rows)
    (args.split_dir / "near_duplicate_review_status.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "total_pairs": report["total_pairs"],
        "decisions": report["decisions"],
        "confirmed_pairs": report["confirmed_pairs"],
        "confirmed_cross_split_pairs": len(report["confirmed_cross_split_pairs"]),
        "unresolved_cross_split_candidates": len(report["unresolved_cross_split_candidates"]),
        "status_file": str(args.split_dir / "near_duplicate_review_status.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
