---
name: security-auditor
description: Scans files for secrets, API keys, and common security flaws.
---

# Security Auditor Skill
When this skill is invoked via `/security-auditor`, Claude should:
1. Search the current directory for strings matching API key patterns (e.g., "sk-", "AIza").
2. Check for "TODO" comments related to security.
3. Review any `.env` files to ensure they are listed in `.gitignore`.
4. Provide a summary of risks found.
