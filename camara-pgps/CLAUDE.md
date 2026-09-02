# Project Instructions

## General Rules

- Read existing code before modifying it.
- Do not rewrite working code unnecessarily.
- Make the smallest change required to complete the task.
- Do not create new files unless they are actually necessary.
- Do not modify unrelated files.
- Preserve the existing architecture and naming conventions.
- Never remove functionality without explaining why.
- Do not invent APIs, endpoints, classes, or configuration.
- If something is unclear, inspect the repository before making assumptions.

## Workflow

For every task:

1. Inspect the relevant files.
2. Explain the proposed change briefly.
3. Implement only the required changes.
4. Run relevant tests.
5. Fix errors caused by the implementation.
6. Summarize exactly what changed.

## Before Large Changes

For architectural or multi-file changes:

1. Analyze the existing implementation.
2. Create an implementation plan.
3. Wait for approval before making major changes.

## Code Quality

- Prefer simple, maintainable code.
- Reuse existing utilities and abstractions.
- Avoid unnecessary dependencies.
- Keep functions focused.
- Add tests for important new behavior.
- Do not change formatting across unrelated files.

## Git

- Never delete or overwrite unrelated work.
- Never reset, revert, or force-push unless explicitly instructed.
- Do not commit changes unless explicitly asked.
- create new branch for each feature 
## Repository Efficiency

Do not inspect:
- .venv/
- node_modules/
- __pycache__/
- .git/
- generated files
- model/cache directories
- large datasets

Only inspect files relevant to the current task.