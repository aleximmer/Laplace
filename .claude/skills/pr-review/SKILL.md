---
name: pr-review
description: Review a GitHub pull request of this repository for bugs and AGENTS.md compliance, and post the findings on the pull request.
disable-model-invocation: true
argument-hint: "[owner/repo/pull/N]"
allowed-tools: Bash(gh pr view:*), Bash(gh pr diff:*), Bash(gh pr comment:*), mcp__github_inline_comment__create_inline_comment
---

Review the pull request $ARGUMENTS and post the result on it.

This skill is adapted from the `code-review` plugin that ships with Claude Code.
It differs in two ways: a review runs even when Claude has commented on the pull
request before, because a maintainer asked for it explicitly, and every run ends
with a comment that names the reviewed commit, so a clean review is
distinguishable from a run that did nothing.

Assumptions for every agent and subagent you launch: all tools work, so do not
make exploratory calls, and only call a tool when the task requires it.

## Steps

1. Run `gh pr view <PR> --json state,isDraft,headRefOid,title,body` and record the
   head commit SHA. If the pull request is closed or a draft, post one short
   comment with `gh pr comment` that says why no review ran, and stop. Do not
   stop because Claude has commented on the pull request before.

2. Launch a haiku agent to return the paths of the guidance files in this
   checkout: `CLAUDE.md` and `AGENTS.md` at the repository root, plus any
   `CLAUDE.md` or `AGENTS.md` in a directory that contains a file the pull
   request modifies. The checkout is the main branch, not the pull request.

3. Launch a sonnet agent to read the pull request with `gh pr view` and
   `gh pr diff` and return a summary of the changes.

4. Launch four agents in parallel. Each returns a list of issues, and each issue
   carries a description and the reason it was flagged, such as "bug" or
   "AGENTS.md adherence". Tell every agent the pull request title and
   description so it knows the author's intent.

   Agents 1 and 2, sonnet: audit the changes for compliance with the guidance
   files. Only apply a guidance file to files under its directory.

   Agent 3, opus: scan the diff itself for obvious bugs without reading further
   context. Flag only significant bugs. Do not flag anything that needs context
   outside the diff to confirm.

   Agent 4, opus: look for problems in the introduced code, including incorrect
   logic and security issues, reading surrounding code where needed. Stay
   within the changed code.

   Only high-signal issues count. Flag an issue when the code will fail to
   import, parse or type-check, when it will produce wrong results regardless
   of input, or when it breaks a guidance rule you can quote. Do not flag
   style, quality concerns, issues that depend on particular inputs or state,
   or subjective suggestions. When in doubt, leave it out. False positives cost
   reviewer trust.

5. For each issue from agents 3 and 4, launch a subagent in parallel to validate
   it with high confidence against the code. Give it the pull request title,
   description and the issue. Use opus for bugs and sonnet for guidance
   violations, and for the latter check that the rule applies to that file and
   is really broken.

6. Drop every issue that step 5 did not confirm. Also drop an issue that an
   earlier Claude comment on this pull request already reports on the same
   code, unless the code is unchanged and the issue still stands.

7. Print the remaining issues to the terminal, then post them. For each issue,
   call `mcp__github_inline_comment__create_inline_comment` with
   `confirmed: true`, one comment per unique issue. Describe the issue briefly
   and link the relevant code. Add a committable suggestion block only when
   committing it fixes the issue entirely and the fix is small; describe larger
   fixes in words.

8. Finish with one summary comment through `gh pr comment`, whatever the
   outcome, in this form:

   ```
   ## Claude review

   Reviewed commit <full head SHA>.

   <Found N issues, posted as inline comments. | No issues found.> Checked for bugs and AGENTS.md compliance.
   ```

## What not to flag

- Pre-existing issues outside the changed code.
- Code that looks wrong but is correct.
- Nitpicks a senior engineer would not raise.
- Anything a linter catches. Do not run the linter to check.
- General concerns such as missing tests or broad security hygiene, unless a
  guidance file requires them.
- Issues a guidance file mentions but the code silences on purpose, for
  example with a lint ignore comment.

## How to write the comments

Write like a careful colleague leaving a note, not like a report. State the
defect, what it causes, and the fix, in one short paragraph of plain sentences.
Talk about the code, not the author. Leave out praise, filler, hedging, emojis,
em dashes, bold labels followed by colons, and openers such as "Note that" or
"It's worth mentioning". Use a code span only for an identifier the reader has
to find. Do not restate the diff or summarize what the pull request does.

## Linking to code

Use the gh CLI for everything on GitHub, never web fetch. Link code with the
full commit SHA so the Markdown preview renders, for example
`https://github.com/aleximmer/Laplace/blob/<full sha>/laplace/baselaplace.py#L10-L15`.
The repository name must match the one under review, the file name is followed
by `#`, and the range is `L<start>-L<end>` with at least one line of context on
each side of the lines you comment on. Shell substitutions such as
`$(git rev-parse HEAD)` do not work inside a comment.
