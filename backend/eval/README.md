# Offline eval tools

Everything here is offline and deterministic: no LLM calls, no network, no database writes.
Run all commands from `backend/`.

## Generate a synthetic cohort

```bash
python3 -m eval.generate_cohort --n 24 --seed 7 --low-scorer-fraction 0.15 --out eval/cohort_seed7.json
```

- `--low-scorer-fraction`: share of students who are low scorers.
- `--low-mode all` (default): a low scorer is at the lowest level on every ordinal trait.
- `--low-mode partial`: a low scorer is at the lowest level on 1 or 2 randomly chosen ordinal traits, and drawn like everyone else on the rest.
- Same arguments and seed give byte-identical JSON.

## Check that team formation is deterministic

```bash
python3 -m eval.determinism_harness eval/cohort_seed7.json --runs 5 --processes 3 --shuffle-input 10
```

Runs the real formation function (`app.matching.form_teams`) repeatedly, in fresh processes with different hash seeds, and on shuffled input orders. Every run must give the same partition. A random-shuffle baseline runs alongside and is expected to FAIL.

## Compare formation methods

```bash
python3 -m eval.quality_eval --sizes 20 22 40 43 80 --seeds 30 --out eval/results
```

Writes `quality_results.json` (raw per-cohort numbers), `quality_table.csv` and `quality_table.md` (one block per cohort size, mean with 95% bootstrap CI). Same arguments give byte-identical files for any `--workers` value.

Methods, all on the same cohorts (team size 4, profile `legacy`, floor policy `one_low_per_team`):

| method | what it is |
|---|---|
| `ours` | `form_teams` as is |
| `ours_no_floor` | floor check disabled during formation (maximin still runs) |
| `ours_no_maximin` | `form_teams` sequence without the maximin pass |
| `random` | seeded shuffle, cut into the same balanced team sizes |
| `skills_only` | production `formation_mode: "skill"` |
| `behavior_only` | `form_teams` with the `skill_coverage` weight set to 0 (the effective weights are printed in the output) |
| `sort_and_chunk` | students sorted by behavior score and cut into consecutive teams, which puts the weakest together |

Cohorts that are infeasible under `one_low_per_team` are skipped and counted in the output.

The low-scorer fraction is a generator setting, not the floor-student share. In partial mode a low scorer can be low only on `safety_contribution`, which is not a floor trait, so fewer students end up at the floor. Each cohort records `n_floor_students`, and each table block reports the actual floor-student share.

## What each metric means

Ordinal levels are mapped to 0, 1, 2 (low, developing, high). A floor student is at the lowest level on accountability or cooperativeness.

| metric | meaning |
|---|---|
| `teams_with_floor_student` | number of teams with at least one floor student |
| `max_floor_students_on_one_team` | most floor students on any one team, counting both traits |
| `max_same_trait_floor_on_one_team` | most floor students on one team for a single trait; `one_low_per_team` keeps this at 1 |
| `*_min_team_mean` | lowest team mean for that trait across all teams |
| `*_mean_of_team_means` | average of the team means for that trait |
| `worst_team_score` | lowest team score (`_total_score` with the default legacy weights) |
| `team_score_std` | spread of team scores across teams |
| `mean_coverage` | average fraction of the reference skills each team covers |
| `worst_off_utility` | lowest individual utility from `compute_assignment_utilities` |

## Why these metrics

- The minimum-trait-floor metrics follow Bell (2007), which found that the weakest team member's conscientiousness and agreeableness predict team performance.
- The team-mean metrics follow Peeters et al. (2006), which found that team averages of conscientiousness and agreeableness predict performance.
- The spread and worst-team metrics measure balance across teams, which is what the maximin pass optimizes.

These citations are from our literature review notes and should be re-checked before they go in a paper.

## Caveats

1. These results show balance and constraint satisfaction under our own scoring, not better collaboration.
2. The profile is "legacy" with no familiarity or preference data.
3. Traits are mapped to 0, 1, 2 evenly spaced.
4. The strength weights in the rubric are provisional and unvalidated.
5. zero_tolerance is not tested because it refuses any cohort with a floor-level student.
