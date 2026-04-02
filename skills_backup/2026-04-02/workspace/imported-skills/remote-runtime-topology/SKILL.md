---
name: remote-runtime-topology
description: Use when work happens across VPNs, jump hosts, login nodes, development containers, and scheduler queues, and you need to decide the correct runtime layer for each task before debugging the project itself.
---

# Remote Runtime Topology

## Overview

This skill turns a messy remote setup into a simple topology:

1. network path
2. control plane
3. execution plane
4. storage plane

Use it whenever a task involves any combination of:

- VPN or campus network access
- jump hosts or bastions
- login nodes
- interactive dev containers
- scheduler jobs
- shared storage mounted across runtimes

## Core Rule

Verify the access chain before debugging the project.

Many "training failures" are really topology mistakes:

- using the login node as if it were the compute runtime
- treating a waiting container as ready
- checking only the scheduler state without verifying the real process
- running commands in one runtime while reading files from another

## Role Map

### VPN or jump host

Purpose:

- provides reachability
- does not host the actual workload

### Login node

Purpose:

- SSH landing point
- light inspection, Git, path checks, submission, container discovery

Usually not the right place for model loading or training.

### Dev container

Purpose:

- interactive compute runtime
- direct GPU or accelerator work
- fast debugging and manual launches

### Scheduler queue

Purpose:

- batch execution
- long runs that do not fit or should not occupy interactive containers

### Shared storage

Purpose:

- stable project path across login node, containers, and jobs

## Workflow

### 1. Draw the chain from outside in

Write the real path as:

`local -> VPN/jump -> login node or portal -> container or queue -> shared storage`

If one hop is broken, do not skip ahead to application debugging.

### 2. Verify each layer in order

Examples:

- jump host or VPN reachability
- login-node SSH access
- container list or scheduler visibility
- chosen runtime availability
- project mount visibility inside the runtime

### 3. Decide where the task belongs

Typical placement:

- network validation: VPN or jump host
- Git and lightweight inspection: login node
- interactive training and debugging: dev container
- long unattended run: scheduler queue

Only use a runtime for the kind of work it is meant to host.

### 4. Interpret runtime states correctly

Examples:

- `运行` or `RUNNING`: usable runtime
- `等待` or `PENDING`: not yet bound to resources
- `退出` or `EXITED`: stale metadata, not a live target

Do not attach project meaning to a runtime that is not actually ready.

### 5. Bind one task to one execution plane

Once the runtime is chosen:

- keep commands there
- verify the project mount there
- monitor processes there

Do not mix log checks, process checks, and relaunches across different runtimes unless explicitly intended.

### 6. Verify the chosen runtime immediately

Minimum checks:

```bash
hostname
whoami
pwd
```

For accelerator work also check:

```bash
nvidia-smi
```

or the cluster’s equivalent accelerator health command.

## Decision Heuristics

### Prefer dev containers when

- the task is interactive
- you need to inspect GPU health directly
- the user wants quick manual control

### Prefer scheduler jobs when

- the run is long-lived
- the interactive runtime is scarce
- the workflow is already batch-friendly

### Prefer login nodes only when

- the work is lightweight
- you are preparing, inspecting, syncing, or submitting

## Common Mistakes

- launching training on the login node
- debugging code before checking network reachability
- assuming all remote paths point to the same storage
- treating a scheduler `RUNNING` state as proof the intended GPU code is alive
- switching runtimes mid-task without re-verifying mounts and environment

## Output Standard

When using this skill, report:

- the access chain
- the chosen execution plane
- why that runtime was selected
- how readiness was verified
- what layer is broken if the task cannot proceed

## Resources

Use together with environment- or cluster-specific access skills.

In this repository, related concrete skills are:

- `hpc-login-ssh`
- `hpc2-gpu-runtime`
- `hpc-sbatch`
