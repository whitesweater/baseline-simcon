---
name: experiment-contract-guard
description: Use when a training, evaluation, or reproduction task must be fixed or adapted without silently changing the experiment definition, especially when some knobs are part of the scientific contract and other knobs are only runtime or scheduling compromises.
---

# Experiment Contract Guard

## Overview

This skill protects reproducibility when an agent is under pressure to "just make it run."

The core idea is to separate:

- experiment-definition changes
- runtime-only changes

Then fix the problem using the lowest-impact layer first.

Use this skill whenever a model run has to be relaunched, memory-tuned, moved across machines, or adjusted after failure.

## Why This Skill Exists

Many bad recovery attempts come from mixing up two different things:

- "the experiment we are claiming to run"
- "the operational conditions used to make it finish"

Examples of experiment-definition changes:

- enabling or disabling a decoder branch
- changing latent-token count
- changing max token count
- changing loss terms or loss weights
- freezing parameters that were supposed to be trained
- changing dataset construction, filtering, or label handling

Examples of runtime-only changes:

- machine or container choice
- queue versus dev container placement
- logging path
- master port
- batch size
- gradient accumulation
- dataloader worker count
- allocator settings
- enabling a memory optimization that does not change the intended algorithmic contract

## Workflow

### 1. Identify the contract source first

Before changing anything, find the most authoritative sources for the run:

- current project guide or handover
- canonical training scripts
- experiment sheet or paper note
- explicit user constraints in the chat

Do not infer the contract from convenience wrappers if the project already has a more canonical entry point.

### 2. Classify every proposed change

For each candidate change, place it into one of three buckets:

- immutable without explicit approval
- safe runtime knob
- ambiguous, requires confirmation

Default to the stricter bucket when unsure.

### 3. Fix at the lowest-impact layer first

Use this order:

1. runtime placement
2. environment repair
3. runtime memory or scheduling knobs
4. experiment-definition changes

Do not skip directly to architecture or logic edits when the problem may only be:

- wrong machine
- wrong queue node
- broken environment
- mismatched driver or library stack

### 4. Never hide a contract change inside a "temporary workaround"

Do not silently do any of the following:

- freeze trainable modules
- reduce latent count
- shorten max token count
- disable a loss term
- switch datasets or checkpoints
- alter tokenizer behavior

If such a change is truly necessary, surface it explicitly as a contract change.

### 5. Report preserved constraints after recovery

After any relaunch, state two things:

- what was preserved
- what changed operationally

Good report shape:

- preserved: decoder training, latent count, max token count, loss logic
- changed: per-device batch, gradient accumulation, runtime host, logging path

## Decision Heuristics

### Usually immutable

- model branch structure
- trainable versus frozen parameter intent
- latent-token count
- token-budget constraints tied to the experiment
- loss composition
- dataset definition
- evaluation protocol

### Usually runtime-only

- container or node selection
- path rebinding
- batch size
- gradient accumulation
- epoch count when the user already approved it as operationally flexible
- allocator flags
- environment rebuild

### Usually ambiguous

- gradient checkpointing
- mixed precision mode changes
- activation checkpointing mode differences
- save strategy changes
- checkpoint selection changes

These can be safe in some projects and contract-relevant in others. Check the project’s norms.

## Red Flags

Stop and realign if a proposed fix:

- changes what parameters are trained
- changes what tokens or sequence lengths are modeled
- changes what data the run sees
- changes what loss is optimized
- is justified only by "it fits now"

## Output Standard

When using this skill, the final recommendation should clearly separate:

- contract-preserving actions
- contract-changing actions
- blocked options that were rejected to preserve reproducibility

This makes later handoff much easier and prevents accidental paper drift.
