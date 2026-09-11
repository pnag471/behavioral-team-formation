"""
Research-alignment evaluation harness.

This package is ADDITIVE and read-only with respect to the production
algorithm: it imports functions from `app.matching` to score and validate
teams, but never modifies matching.py, models.py, or the rubric.  See
/research_alignment_audit.md for how it is used and why it exists.
"""
