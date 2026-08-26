# CLAUDE.md — GateFall

This file contains the conventions and experimental invariants that must be followed for all work
in this repository. These instructions are self-contained and do not assume that context files,
skills, or custom agents are installed outside the repository.

## Repository conventions

- **Project documentation, code comments, and PR descriptions use Brazilian Portuguese.** Code,
  identifiers, commit messages, and PR titles remain in English.
- **Comments are allowed but should be used sparingly.** Prefer a clearer name or a smaller function
  first. Preserve comments that document a formula, sign convention, axis or channel ordering,
  unit, or the paper from which a descriptor was derived.
- **Automated tests are not required.** Do not assume that a test framework exists or introduce one
  solely for a change that does not require tests.
- **Documentation changes follow the ownership of each surface.** Update only the surface made
  inaccurate by a change. Keep `README.md` limited to the project overview, installation, basic
  usage, and links; place technical details on the corresponding MkDocs page. Do not duplicate the
  same explanation across both surfaces.
- **Every addition, modification, or removal of Python code requires `uv run pyright`.** Run this
  check before considering the change complete.

## Experimental invariants

1. **A, B, and C are identical except for the content of each line in the window.** Window size,
   train/test split, seed, temporal encoder, and epoch count are identical across configurations;
   only the per-timestep feature vector changes. Apply any necessary shared change to all three
   configurations and rerun all three.
2. **Split train and test data by video, preferably by subject, never by window.** Windows from the
   same video on both sides leak information and inflate metrics.
3. **The pipeline is monocular RGB.** No depth-derived descriptor may enter any branch, including
   columns D–K of the URFD feature CSV produced from Kinect depth data.
4. **Keep backbones frozen and precompute features offline.** Do not train or fine-tune YOLO-Pose,
   DINOv3, or SAM 3. When training is implemented, train only the fusion head and TCN.
5. **When HDF5 storage is implemented, group features by video.** Never create one file per frame.

## Dominant risk

The dominant risk is scope, not technical difficulty. Every proposed addition—a fourth
configuration, another dataset, another backbone, or another metric—must explicitly state what it
displaces. Without that trade-off, the default answer is no.
