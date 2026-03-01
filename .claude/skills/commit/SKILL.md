---
name: commit
description: Automates staging and committing changes using Conventional Commits format.
---
# Commit Protocol
When the user runs `/commit`, you must:
1. Run `git status` and `git diff` to silently analyze the current changes.
2. Ask the user for a 1-sentence summary of what they just built (if not provided).
3. Create a strict Conventional Commit message (e.g., `refactor: extract 3D angle math to utils.py`).
4. Execute `git add .`
5. Execute `git commit -m "Your generated message"`.
6. Show the user a success message confirming the commit.
