# 2026-09-07 — Publishing the backlog: the repo the preprint points at

**Subject:** what a reader arriving from the Zenodo record could and could not see.

## The situation

`paper/MANUSCRIPT.md` §7 says the preprint lives in this repository, and the
Zenodo record for 10.5281/zenodo.22126781 lists this repo as `isSupplementedBy`.
Both were true. What was not true was that the repo's default branch held the
current state of the work.

Local `main` was 9 ahead and 2 behind `origin/main`. A separate branch,
`review-response`, carried 11 commits that had never been pushed at all, and the
DOI stamp plus the regime_b reproducibility paragraph existed only as uncommitted
working-tree edits.

Concretely, a reader following the DOI to the repo saw the manuscript, the
figures, and the regime_b bundle — but not:

- `CHANGELOG.md`, and the hypothesis history in it
- `episode_loop.py`, the rung-5 runner
- the three measured runs of 2026-09-03/04 and their raw rows
- `runs/2026-09-04_episode-loop.md` and
  `runs/2026-09-04_keys-episode-oracle-slots.md`
- the two SCOREBOARD rows recording both as **MIXED**

Every one of those items is a negative or partial result. The paper was public
and the record of what did not work was not.

## What was done

Committed the DOI/regime_b edits, brought `CITATION.cff` and the concept-DOI log
onto the branch (both byte-identical to the copies on `origin/main`, verified by
`git hash-object` against `git rev-parse origin/main:<path>` before touching
them), merged `origin/main`, pushed `review-response`, then fast-forwarded `main`
and pushed that.

The merge produced five conflicts: `AGENTS.md`, `CHANGELOG.md`, `RESEARCH_LOG.md`,
`SCOREBOARD.md`, `paper/MANUSCRIPT.md`. All were add/add or content conflicts where
this branch appends and `main` does not. Before resolving, the remote side of each
conflict region was extracted and counted — zero lines in every case except the
`RESEARCH_LOG.md` "Last updated" stamp, where the branch value (2026-09-04) is the
newer one. Resolution therefore kept the branch side without discarding anything
authored on `main`.

The check that matters: `git diff --diff-filter=D --name-only origin/main..HEAD`
returned empty before the push. No file that existed on the public default branch
was removed by this merge.

Excluded deliberately: `debug_index/` (7.9M of Tantivy artifacts, build output),
and the 29 modified / 21 untracked files still sitting in the tree, which have not
been read and may be unrelated backlog.

## What this is and is not

This is not a result. Nothing was measured and nothing was rerun. It is a
provenance repair: the artifact a public DOI points at now matches the state of
the work, including the parts that failed.

It is worth naming why it mattered. The two Sept 4 cells both read "write landed,
mouth missed" — `write_norm` tracks gain, `slots_written` hits 11, and the model
still does not say where the keys are. Those are exactly the rows that keep the
rank-split claim in §5.1 honest: 128 dimensions steer meaning before they carry
spelling. A repo showing only the preprint and the oracle successes would have
been a weaker and less accurate object than the one that now exists.

## Next

Decide what the remaining untracked tree is. It gets read before it gets pushed,
or it gets left alone — not swept in because it happened to be in the directory.
