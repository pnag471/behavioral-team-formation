"""
Synthetic student cohort generator (offline, deterministic).

Same arguments + same seed => byte-identical JSON output. All randomness goes
through one random.Random(seed); dimension and skill lists are iterated in a
fixed (rubric-file / sorted) order; JSON is written with sort_keys and no
timestamps.

Each student record:
    {"id": "syn-007", "name": "Synthetic 007",
     "traits": {"accountability": "high", "planning_style": "planner", ...},
     "skills": {"backend": 0.62, "ml": 0.31},        # optional
     "low_scorer": false}

`traits` uses the REAL rubric_v1.yaml dimension names and allowed_values
labels. See eval/cohort.py SHAPE NOTE for why these are labels, not [0,1]
scores. `skills` is a {skill_name: proficiency in [0,1]} dict drawn from
app.matching.REFERENCE_SKILLS (the set skill_coverage is measured against).

Usage (from backend/):
    python3 -m eval.generate_cohort --n 24 --seed 7 \
        --low-scorer-fraction 0.15 --out eval/cohort_seed7.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from typing import List, Optional

from app.matching import REFERENCE_SKILLS
from eval.cohort import trait_dimensions
from scoring.dimension_scorer import get_rubric


def _ordinal_levels(dim: dict) -> List[str]:
    return list(dim["allowed_values"])  # rubric lists ordinal levels low -> high


LOW_MODES = ("all", "partial")


def generate_cohort(
    n: int,
    seed: int,
    low_scorer_fraction: float = 0.0,
    with_skills: bool = True,
    min_skills: int = 1,
    max_skills: int = 3,
    low_mode: str = "all",
) -> dict:
    """low_mode "all": a low scorer is at the lowest level on every ordinal trait.
    low_mode "partial": a low scorer is at the lowest level on 1 or 2 randomly
    chosen ordinal traits, and drawn like everyone else on the rest."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= low_scorer_fraction <= 1.0:
        raise ValueError("low_scorer_fraction must be in [0, 1]")
    if low_mode not in LOW_MODES:
        raise ValueError(f"low_mode must be one of {LOW_MODES}")

    rng = random.Random(seed)
    dims = trait_dimensions()
    ordinal_names = [name for name, d in dims.items() if d["type"] == "ordinal"]
    skill_pool = sorted(REFERENCE_SKILLS)  # sorted: set order must not leak into output

    n_low = round(low_scorer_fraction * n)
    low_indices = set(rng.sample(range(n), n_low))

    students = []
    for i in range(n):
        is_low = i in low_indices
        if not is_low:
            low_traits = set()
        elif low_mode == "all":
            low_traits = set(ordinal_names)
        else:
            k = rng.randint(1, min(2, len(ordinal_names)))
            low_traits = set(rng.sample(ordinal_names, k))
        traits = {}
        for name, d in dims.items():  # rubric file order — fixed
            values = list(d["allowed_values"])
            if d["type"] == "ordinal":
                levels = _ordinal_levels(d)
                if name in low_traits:
                    # "Low on every trait": lowest level on every ORDINAL dim.
                    traits[name] = levels[0]
                else:
                    # TODO(team-decision): non-low students draw uniformly from
                    # the levels ABOVE the lowest, so low_scorer_fraction is the
                    # ONLY source of floor-level students and the floor load is
                    # exactly controllable. Alternatives: uniform over all levels
                    # (floor-level students appear at random, cohorts can go
                    # infeasible), or a skewed distribution matched to real pilot
                    # data once we have it.
                    traits[name] = rng.choice(levels[1:])
            else:
                # TODO(team-decision): categorical dims (planning_style,
                # leadership_style, ...) have no "low" end in the rubric, so a
                # low scorer draws them uniformly like everyone else. "Low on
                # every trait" is therefore implemented as "lowest level on every
                # ORDINAL trait" (accountability, cooperativeness,
                # safety_contribution). Confirm that is the intended meaning.
                traits[name] = rng.choice(values)

        rec = {
            "id": f"syn-{i:03d}",
            "name": f"Synthetic {i:03d}",
            "traits": traits,
            "low_scorer": is_low,
        }
        if low_mode == "partial" and is_low:
            rec["low_traits"] = sorted(low_traits)
        if with_skills:
            # TODO(team-decision): how skills feed in. The real matcher only
            # reads skill NAMES (skill_coverage vs REFERENCE_SKILLS); the
            # proficiency numbers only matter if an optional ability pass is
            # turned on (it is OFF in the default "legacy" profile). Default
            # here: 1–3 skills per student, proficiency uniform in [0,1],
            # independent of low_scorer status. Alternatives: correlate skill
            # proficiency with low_scorer, or omit skills (skill_coverage then
            # contributes 0 for every team).
            k = rng.randint(min_skills, max_skills)
            chosen = sorted(rng.sample(skill_pool, k))
            rec["skills"] = {s: round(rng.random(), 4) for s in chosen}
        students.append(rec)

    # TODO(team-decision): availability, interests/learning_goals and
    # ability_score are left empty. The real objective weights
    # availability_overlap (0.2) and shared_interests (0.1); with empty data
    # those terms are constant (0.5 and 0.0) for every team, so they cannot
    # change which teams form. Add generators for them if the evals should
    # exercise those terms.
    params = {
        "n": n,
        "seed": seed,
        "low_scorer_fraction": low_scorer_fraction,
        "with_skills": with_skills,
        "min_skills": min_skills,
        "max_skills": max_skills,
    }
    if low_mode != "all":  # omitted for "all" so the default output stays byte-identical
        params["low_mode"] = low_mode
    return {
        "generator": "eval.generate_cohort",
        "rubric_schema_version": get_rubric().get("schema_version"),
        "params": params,
        "trait_dimensions": list(dims),
        "n_low_scorers": n_low,
        "students": students,
    }


def to_json(cohort: dict) -> str:
    return json.dumps(cohort, indent=2, sort_keys=True) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=24, help="number of students (default 24)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--low-scorer-fraction", type=float, default=0.0,
                   help="fraction of students who are low scorers (default 0.0)")
    p.add_argument("--low-mode", choices=LOW_MODES, default="all",
                   help="all: low on every ordinal trait (default); partial: low on 1 or 2 ordinal traits")
    p.add_argument("--no-skills", action="store_true", help="omit the skills dict")
    p.add_argument("--out", help="output path (default: stdout)")
    args = p.parse_args(argv)

    text = to_json(generate_cohort(
        n=args.n, seed=args.seed,
        low_scorer_fraction=args.low_scorer_fraction,
        with_skills=not args.no_skills,
        low_mode=args.low_mode,
    ))
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.n} students to {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
