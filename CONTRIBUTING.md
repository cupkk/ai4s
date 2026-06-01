# Contributing

Thanks for helping improve this project.

## How to contribute

1. Open an issue before starting a large change so the scope can be discussed.
2. Keep changes focused and include a short explanation of the experiment, data path, or script behavior affected.
3. Do not commit private datasets, secrets, leaderboard credentials, generated model artifacts, or large output folders.
4. Add or update tests when changing data parsing, submission validation, strategy generation, or diagnostics.
5. Run the relevant checks before opening a pull request.

## Development checks

```bash
python -m src.check_submission --submission output.csv
python -m src.compare_strategies --output outputs/strategy_compare.csv
```

Use narrower commands when working on a specific module. If a command requires private or competition-provided data, document the expected input path instead of committing the data.

## Pull request expectations

Pull requests should include:

- What changed and why
- Which files or scripts were affected
- How the change was validated
- Any data assumptions or known limitations

Maintainers may ask for a smaller patch, extra provenance notes, or additional validation before merging.
