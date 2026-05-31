# Frontend Follow-ups

- [ ] `server/config/database.js`: re-introduce env-aware DB selection
      (`NODE_ENV === 'production'` → MySQL via 72a906a's path,
       otherwise → SQLite). Currently both paths use SQLite which
       regresses the prod story landed in 72a906a. Not blocking the
       supervisor demo (no prod deploy yet); fix before any deployment.
