# PR Attribution Workflow

> **Date:** 2026-05-10
> **Purpose:** How to properly credit contributors when implementing features inspired by un-mergeable PRs

## Context

When implementing a feature inspired by an existing PR that is un-mergeable (e.g., built against old version), proper attribution is important for:
- Contributor recognition on GitHub profiles
- Historical record of ideas
- Maintaining good open-source relationships

## Steps

### 1. Create new PR with attribution

In the new PR description:
```markdown
## Attribution

Co-authored-by: Original Author <email@users.noreply.github.com>

Inspired by and crediting [PR #N](https://github.com/.../pull/N) by @username.
The core ideas — [list specific concepts] — were all from their work.
This is a [rebased/fresh] implementation on [version].
```

### 2. Add co-author trailer to commit

```bash
git commit --amend --author="Original Author <email@users.noreply.github.com>" \
    -m "feat: description

Co-authored-by: Original Author <email@users.noreply.github.com>

Inspired by PR #N by @username. The [specific concepts] were from their work."
```

The `Co-authored-by:` trailer shows on GitHub as a co-authored contribution.

### 3. Close original PR with credit comment

```bash
gh pr close N --comment "Thanks @username for the excellent work on this!

The core ideas — [list concepts] — are all from your PR. We've implemented a
[rebased/fresh] version as PR #M. Your work was the inspiration and starting
point for this feature. Co-author credit has been added to the new PR's commit.

Closing this to avoid duplicate efforts. Thanks again for the contribution! 🙏"
```

### 4. Update documentation

Reference the original PR in relevant docs:
- README.md: "Inspired by PR #N by @username"
- Skill docs: "Attribution: PR #N by @username"
- Commit history: Co-author trailer preserves the link

## Why This Matters

- GitHub counts co-authored contributions on contributor profiles
- PR #1 stays visible as "Closed" with all commits intact
- Anyone browsing history can see the original contribution
- Maintains good open-source relationships

## Example

PR #1 (bilingual search) → PR #2 (rebased implementation)
- @kelongyan credited in PR #2 description
- Co-author trailer on PR #2 commit
- PR #1 closed with thank-you comment linking to PR #2
