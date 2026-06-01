# Open Source Maintenance

This document summarizes the public maintenance model for `ai4s`.

## Project scope

`ai4s` is a Python research and competition pipeline for AI-for-science energy storage strategy work. It combines power-market price prediction, NWP weather features, historical price statistics, strategy generation, submission-format validation, and experiment diagnostics.

The repository is intended to be useful for:

- researchers studying energy-storage strategy generation under market and weather uncertainty
- developers reproducing LightGBM, NWP, and rolling-validation baselines
- maintainers comparing candidate strategies with explicit legality and provenance checks

## Maintainer responsibilities

The primary maintainer is responsible for:

- reviewing changes to data parsing, NWP feature extraction, strategy generation, and validation logic
- triaging bug reports about submission legality, data-shape mismatches, and diagnostic scripts
- keeping generated artifacts, private competition data, and secrets out of version control
- documenting validated candidates, failed candidates, and known risk boundaries
- cutting releases when the public baseline, documentation, or validation workflow changes materially

## Current maintenance priorities

1. Keep the final submission path reproducible and clearly documented.
2. Strengthen tests for data-shape handling, submission legality, and strategy comparison.
3. Improve documentation for user-provided data paths and non-redistributable inputs.
4. Separate reusable source code from generated outputs and experiment scratch files.
5. Add small, reviewable issues for diagnostics, documentation, and robustness improvements.

## API-credit use boundary

If external AI/API credits are used, they should support repository maintenance: documentation, issue triage, test generation, experiment-log summarization, and code-review assistance. They should not replace the checked-in model pipeline, fabricate leaderboard results, or expose private data.
