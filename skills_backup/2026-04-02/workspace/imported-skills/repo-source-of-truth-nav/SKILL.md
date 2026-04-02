---
name: repo-source-of-truth-nav
description: Use when entering an unfamiliar repository that has multiple guides, entry scripts, historical directories, or overlapping outputs, and you need to quickly identify the true source of truth, active working area, canonical entry points, and trustworthy results without reading everything.
---

# Repo Source Of Truth Navigator

## Overview

This skill helps an agent build the right mental model before touching code or operations.

Use it when a repository has:

- multiple READMEs or handover docs
- several training or evaluation script directories
- legacy outputs mixed with current runs
- paper names that do not exactly match code names

The goal is not to summarize the whole repo.

The goal is to answer:

- where is the real repo boundary
- which documents are authoritative
- which directories are active now
- which entry points are canonical
- which results should be trusted first

## Core Rule

Do not start by reading everything.

Start by locating the repository’s navigation layer, then read downward only as needed.

## Workflow

### 1. Establish the repo boundary first

Verify:

- Git root
- major top-level directories
- whether important subdirectories are separate repos or only subtrees

Typical checks:

```bash
git rev-parse --show-toplevel
find . -maxdepth 2 -type f \( -name 'README.md' -o -name '*GUIDE*.md' -o -name '*HANDOVER*.md' \)
```

### 2. Find the navigation documents

Prefer high-level files that explain:

- repository boundary
- active project area
- entry-point scripts
- output layout
- trust or handoff rules

Examples:

- root `README.md`
- `PROJECT_GUIDE.md`
- `HANDOVER.md`
- newcomer or migration guides

Treat these as the navigation layer, not as yet another copy of implementation details.

### 3. Separate active paths from historical or reference paths

Classify important directories into:

- active code and scripts
- current output roots
- local runtime assets
- historical results
- external reference copies

Do not assume every large directory is part of the active execution path.

### 4. Resolve naming mismatches early

Build a small mapping for:

- paper names versus code names
- method names versus flag combinations
- old stage names versus current stage names

This avoids reading correct code with the wrong interpretation.

### 5. Identify the canonical entry points

For the current task, explicitly locate:

- preparation entry point
- training entry point
- evaluation entry point
- migration or sync entry point if relevant

Prefer maintained script directories over historical wrappers.

### 6. Define result trust order

When outputs conflict, decide what should be trusted first.

Typical signals:

- current stage outputs versus historical outputs
- curated result directories versus scratch outputs
- documented plots or final tables versus ad hoc logs

### 7. Produce a short repo operating summary

Before deeper work, summarize:

- repo root
- active subproject
- authoritative docs
- canonical entry points
- directories to avoid treating as primary

## Heuristics

### Usually trustworthy first

- root-level project guide or handover
- active script directory named around the current experiment
- config or env files used by those scripts
- result directories explicitly called final, useful, curated, or current stage

### Usually lower priority

- legacy scripts with old naming conventions
- copied external repos
- stale outputs that are not referenced by current docs
- notebooks that are not part of the documented workflow

## Common Mistakes

- treating a subdirectory as its own repo when it is only part of the main repo
- diving into model code before finding the current script entry points
- assuming the largest output directory is the current source of truth
- ignoring explicit handover documents and reconstructing the repo from guesswork

## Output Standard

When using this skill, report:

- repo root
- active project area
- authoritative docs to read first
- canonical prepare, train, and eval entry points
- historical or reference directories that should not be mistaken for the active path
- current result trust order

## Resources

Pair this skill with project-specific guides when they exist.

In this repository, the concrete examples are:

- `PROJECT_GUIDE.md`
- `NEWCOMER_HANDOVER.md`
