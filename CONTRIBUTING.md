# Contributing

This repository uses a lightweight release workflow: short-lived work branches,
an integration branch for the next release, and a protected release branch.

## Branches

### `main`

`main` is the released branch.

- It should always be releasable.
- It is the source for version tags and crates.io publishes.
- Do not do normal feature work directly on `main`.
- Merge into `main` only through release PRs or urgent hotfix PRs.

### `develop`

`develop` is the integration branch for the next release.

- Normal work targets `develop`.
- It should stay green in CI.
- It may contain completed work that has not been released yet.
- Merge `develop` into `main` only when preparing a release.

### Work branches

Create short-lived branches from `develop` for normal work:

```bash
git checkout develop
git pull --ff-only
git checkout -b feat/my-change
```

Use descriptive prefixes:

- `feat/` for new behavior
- `fix/` for bug fixes
- `docs/` for documentation
- `chore/` for maintenance
- `release/` for release preparation

For urgent production fixes, branch from `main` instead:

```bash
git checkout main
git pull --ff-only
git checkout -b fix/my-hotfix
```

After a hotfix is released from `main`, merge or cherry-pick it back into
`develop`.

## Pull Requests

Normal PR flow:

1. Branch from `develop`.
2. Keep the change focused.
3. Open a PR back into `develop`.
4. Wait for CI to pass.
5. Review the diff, tests, docs, and public API impact.
6. Merge once the branch is green and the change is accepted.

Release PR flow:

1. Create `release/vX.Y.Z` from `develop`.
2. Apply the version bump, release notes, and documentation updates.
3. Open a PR from `release/vX.Y.Z` into `main`.
4. Run the full release checklist from `docs/PUBLISHING_GUIDE.md`.
5. Merge only when the release commit is green and approved.
6. Tag and publish from `main`.

Prefer squash merges for short-lived feature, fix, docs, and chore branches
that target `develop`, unless the branch history is intentionally structured
and useful to keep.

Do not squash release PRs or direct `develop` -> `main` promotion PRs. Use a
normal merge commit so `main` and `develop` keep a clean ancestry relationship
and can be merged back together without duplicating equivalent changes.

## Quality Gates

Before opening or merging a normal PR, run the checks that match the change:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace -- --test-threads=1
```

For async changes:

```bash
cargo clippy -p whisper-cpp-plus --all-targets --features async -- -D warnings
cargo test -p whisper-cpp-plus --features async -- --test-threads=1
```

For macOS or Metal-sensitive changes:

```bash
cargo xtask test-setup
MACOSX_DEPLOYMENT_TARGET=14.0 cargo clippy -p whisper-cpp-plus --all-targets --features metal -- -D warnings
MACOSX_DEPLOYMENT_TARGET=14.0 cargo test -p whisper-cpp-plus --features metal -- --test-threads=1
```

For prebuilt-cache changes:

```bash
cargo xtask clean
MACOSX_DEPLOYMENT_TARGET=14.0 cargo xtask prebuild --force
cargo xtask info
WHISPER_PREBUILT_PATH=prebuilt/<apple-target>/release cargo test -p whisper-cpp-plus-sys
WHISPER_PREBUILT_PATH=prebuilt/<apple-target>/release cargo test -p whisper-cpp-plus --test stream_pcm_integration -- --nocapture --test-threads=1
```

The project should stay warning-free. If a warning is intentional, prefer to
make that intent explicit in code rather than allowing warning noise to build up.

The macOS workflow uses path filtering for pull requests. Documentation-only
changes do not run the expensive model-backed macOS and Metal test job. Code,
build, dependency, `xtask`, sys-crate, high-level crate, and workflow changes do
run the full macOS job. Pushes to `main` and `develop` always run the full macOS
job so integration branches stay verified.

Clippy is used as a CI gate. The baseline is:

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p whisper-cpp-plus --all-targets --features async -- -D warnings
```

On macOS, also run:

```bash
MACOSX_DEPLOYMENT_TARGET=14.0 cargo clippy -p whisper-cpp-plus --all-targets --features metal -- -D warnings
```

## Releases

Crates.io publishes are irreversible, so releases only happen from a clean,
verified `main` commit.

The release sequence is:

1. Merge completed work into `develop`.
2. Create `release/vX.Y.Z` from `develop`.
3. Bump versions and update release documentation.
4. Open a release PR into `main`.
5. Run the publishing guide checklist.
6. Merge the release PR into `main` with a normal merge commit, not a squash
   merge.
7. Tag the release commit on `main`.
8. Publish `whisper-cpp-plus-sys` first.
9. Wait for the crates.io index to show the sys crate.
10. Publish `whisper-cpp-plus`.
11. Create the GitHub release.
12. Merge `main` back into `develop` if the branches diverged during release.

See `docs/PUBLISHING_GUIDE.md` for the detailed command checklist.

## External Dependencies

The workspace pins a stream-PCM fork of `whisper.cpp` through the sys crate
submodule. Updates to that pin should be treated as release-sensitive work:

- update the submodule intentionally
- record the upstream base release or commit
- run the model-backed tests
- verify macOS/Metal behavior when relevant
- document the change in the release notes
