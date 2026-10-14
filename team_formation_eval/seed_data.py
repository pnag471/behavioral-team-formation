"""
Minimal student model and deterministic synthetic cohort generator.

This package validates team-partitioning heuristics against a random
floor and a brute-force ceiling. It only needs enough of a student model
to score skill coverage, so Student is deliberately small: id, name, and
a set of skills. It is not the same Student model as backend/app/models.py
and is not meant to be. This package has no dependency on the FastAPI app.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Student:
    id: str
    name: str
    skills: frozenset


def make_synthetic_cohort(seed, n_students, skill_set, skills_per_student=(1, 3)):
    """
    Deterministic synthetic cohort. Same seed, same n_students, same
    skill_set always produces the same list of students.

    skills_per_student: (min, max) number of skills drawn per student,
    sampled without replacement from skill_set. If skill_set has fewer
    skills than the requested max, every student is capped at
    len(skill_set).

    TODO(prisha): replace with real cohorts (real L0-L5 pulls, or
    resampled real data), one per seed. This function exists only so the
    cross-cohort validation has something to run against before real data
    is wired in.
    """
    rng = random.Random(seed)
    skill_pool = sorted(skill_set)
    lo, hi = skills_per_student
    students = []
    for i in range(n_students):
        k = min(rng.randint(lo, hi), len(skill_pool))
        picked = rng.sample(skill_pool, k)
        students.append(Student(
            id=f"c{seed}_s{i}",
            name=f"Student {i}",
            skills=frozenset(picked),
        ))
    return students
