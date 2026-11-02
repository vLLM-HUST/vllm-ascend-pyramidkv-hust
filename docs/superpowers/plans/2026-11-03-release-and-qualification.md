# Standalone candidate release and qualification plan

> **For agentic workers:** Use superpowers:executing-plans inline, with a fresh
> pre-merge review. The user authorized issue/PR handling and publication.

**Goal:** Close superseded upstream work, publish an independently installable
PyramidKV candidate, and make the remaining effect/resource work actionable.
**Architecture:** Keep the reviewed adapter unchanged; change version metadata
and delivery documentation. Publish immutable wheel/sdist assets on GitHub.
**Tech Stack:** Python packaging, GitHub API, pinned Ascend TP2 host.
**Spec:** ../../releases/0.4.0rc1.md

## Global constraints

Preserve current algorithm and host support. Historical shared-adapter results
keep their original source identity. Use temporary storage outside workspace;
keep the canonical runtime on main. No PyPI credential is configured, so the
public channel for this candidate is GitHub Release. The choice of rc1 is the
implementer's: full independent-version quality/capacity qualification is pending.

## Review focus

- Release version agrees across package, runtime and extension manifest.
- The wheel installs/discovers without KVCompress, torch or host packages.
- Downloaded public bytes equal the tested bytes, with precise source identity.
- Runtime installation and disable/rollback use fresh processes.
- Old shared performance and other MOD scores cannot qualify this release.

### Task 1: Reconcile tracking and historical reports

- [x] Update #1/#8 while preserving their historical records.
- [x] Close shared #33 as superseded by PyramidKV #28/#29.
- [x] Independently review #549; verify all raw checksums and offline results.
- [ ] Fix substantive verifier findings without changing measured originals;
  require final tests/lint and merge the exact reviewed head.

### Task 2: Publish the standalone candidate

Files: pyproject.toml, package __init__, extension manifest, README,
install-and-rollback.md, docs/releases/0.4.0rc1.md.

- [x] Set 0.4.0rc1 consistently and document public install and support limits.
- [x] Run full tests, formatting/lint, NPU oracle, build and clean-wheel checks.
- [ ] Obtain fresh release-branch review, resolve findings, pass CI and merge.
- [ ] Build/tag exact merged source; verify wheel/source identity and create a
  GitHub prerelease with wheel, sdist, provenance and SHA256SUMS.
- [ ] Download public assets and verify their bytes; test installed-wheel serving
  and fresh-process disable/rollback before marking release acceptance complete.

### Task 3: Arrange remaining qualification and resource handoff

- [ ] Create linked, assigned tasks for #1 effect verification and #8 execution
  resources/judge calibration, with exact inputs, outputs and stop conditions.
- [ ] Publish the current-version evaluation protocol and explicit preparation
  dependencies; request concrete resource/judge handoff through existing owners.
- [ ] Read back all issue/PR/release state, synchronize canonical source/environment,
  stop owned services and remove temporary worktrees/artifacts.
