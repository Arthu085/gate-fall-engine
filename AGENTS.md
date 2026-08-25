# AGENTS.md — GateFall

This file documents only what diverges from `~/.codex/AGENTS.md` and the project's experimental
invariants. Every global rule not explicitly replaced here remains applicable.

## Declared overrides

- **Documentation, code comments, and PR descriptions use Brazilian Portuguese.** This replaces
  the global default prose language because the project is an academic deliverable for a
  Portuguese-speaking audience. Code, identifiers, commit messages, and PR titles remain English.
- **Comments are allowed but should be used sparingly.** This replaces the global prohibition on
  production comments. Prefer a better name or smaller function first. Keep comments that anchor a
  formula, sign convention, axis or channel ordering, unit, or the paper from which a descriptor
  originated.
- **No tests are required; disable the `test-writer` step of `code-workflow`.** The `implementer`
  verifies against the repository gate instead.
- **Every project change, addition, or reformulation must update or add to both `README.md` and the
  MkDocs documentation.**
- **Every addition, modification, or removal of Python code must run `uv run pyright`.** This check
  is mandatory before considering the change complete.

## Experimental invariants

1. **A, B, and C are identical except for each line's content in the window.** Window size,
   train/test split, seed, temporal encoder, and epoch count are identical across configurations;
   only the per-timestep feature vector changes. Apply any necessary shared change to all three and
   rerun all three.
2. **Split train and test data by video, preferably by subject, never by window.** Windows from the
   same video on both sides leak information and inflate metrics.
3. **The pipeline is monocular RGB.** No depth-derived descriptor may enter any branch, including
   columns D–K of the URFD feature CSV produced from Kinect depth data.
4. **Backbones are frozen and features are precomputed offline.** Do not train or fine-tune
   YOLO-Pose, DINOv3, or SAM 3. Only the fusion head and TCN are trained.
5. **Group features in HDF5 files by video.** Never create one file per frame.

## Dominant risk

The dominant risk is scope, not technical difficulty. Every proposed addition—a fourth
configuration, extra dataset, another backbone, or another metric—must explicitly state what it
displaces. Without that trade-off, the default answer is no.
