# Chat uses a Manifest; CI uses a fixed checkout

Interactive Chat must review arbitrary pull requests without requiring the user
to clone or switch repositories. It therefore fetches one stable set of changed
file patches from GitHub, verifies that the head SHA did not change during the
fetch, and builds an in-memory Manifest plus virtual diff files. Every Chat Lens
reads the same `ReviewWorkspace`.

CI already owns a complete checkout at the event's fixed head SHA. It continues
to use `GitCheckoutSource`, which verifies the head, base commit, worktree root,
and cleanliness before allowing any Lens to run.

Both adapters implement the same `review(repo, number)` interface and share Lens
prompts, structured Finding schemas, Report rendering, and LangGraph
orchestration. The interactive adapter gains zero-setup Review at the cost of
GitHub patch truncation and missing unchanged files; the CI adapter retains
complete repository context and fixed-SHA guarantees.
