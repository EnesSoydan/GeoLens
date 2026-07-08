"""Offline CLI pipeline scripts (dataset prep, index build, evaluation).

Marked as a package so modules can share pure helpers (e.g.
``scripts.calibrate_confidence`` reuses ``scripts.evaluate``) and so the type
checker resolves them under a single ``scripts.*`` module path.
"""
