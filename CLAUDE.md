# CLAUDE.md — GateFall

This file exists **only** to document what diverges from

`~/.claude/CLAUDE.md` and the project's experimental invariants. Everything not

declared here remains fully applicable from the global file, without repetition.

---

## Declared overrides

Each item names the global rule it replaces and provides the justification. The scope of the

override is the named rule — nothing beyond it is affected.

- **Docs, comments, and PR titles/descriptions in Brazilian Portuguese.** Replaces the global language rule

  (§5, "Docs and code comments: English by default"). Code, identifiers,

  commit messages — these are not allowed to be overridden by the global rules, and this repository does not override them.

- \*\*Comments are allowed, without formal restrictions, but should be used

  sparingly.\*\* Completely replaces the global absolute prohibition

  (§6, "No comments in production code, ever"). The guideline is: code saturated with

  comments is poorly factored code — before adding a comment, try a better name or

  a smaller function. The comments worth keeping are those that anchor a formula, sign

  convention, axis/channel ordering, unit, or the reference to the paper from which the

  descriptor originated.

- \*\*No tests are required; the `test-writer` step of `code-workflow` is

  disabled in this repository.\*\* Replaces the mandatory testing step of the

  global pipeline (§3); the `implementer` verifies against the repository gate and

  nothing else changes in the sequence.

- \*\*Every change, addition, or reformulation of the project must update or add to the README.md

  and the MkDocs documentation.\*\*

- \*\*Every change, addition, or removal of Python code must run uv run pyright.

  This check is mandatory whenever Python code is modified, added, or removed,

  and the result must be verified before considering the change complete.\*\*

---

## Experimental invariants

These are not overrides: they are project domain rules. Violating them invalidates the

result, not merely the code style.

1. **A, B, and C are identical except for the content of each line in the window.**

   Window size, train/test split, seed, temporal encoder, and number of

   epochs are the same across all three configurations; only the feature vector per

   timestep changes. Any change that affects only one of them invalidates the

   comparison — if a change is necessary, apply it to all three and re-run all three.

2. \*\*The train/test split is always performed by video, preferably by subject.

   Never by window.\*\* Windows from the same video appearing on both sides of the split leak

   information and inflate the metric.

3. **The pipeline is monocular RGB.** No descriptor derived from a depth map

   may enter any branch — including columns D–K of the URFD feature CSV,

   which are calculated from the Kinect depth sensor.

4. **Frozen backbones, features precomputed offline.** YOLO-Pose, DINOv3,

   and SAM 3 are neither trained nor fine-tuned. Only the fusion head and

   the TCN are trained.

5. **Features are grouped in HDF5 files by video.** Never one file per frame.

---

## Dominant risk

The dominant risk of this project is **scope, not technical difficulty**. Every

proposed addition — a fourth configuration, an extra dataset, another backbone,

another metric — must explicitly state **what it displaces in exchange**. Without that

explicit trade-off, the default answer is no.
