"""Supervisão de movimento e regularização cinemática 2D, sem dinâmica 3D."""

SCHEMA = "motion-v1"
RECIPES = ("baseline", "motion", "motion_physics")
JOINTS = (5, 6, 11, 12)
SPATIAL_SCALE = .1
