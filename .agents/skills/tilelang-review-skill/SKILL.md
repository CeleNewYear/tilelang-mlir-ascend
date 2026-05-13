---
name: tilelang-review-skill
description: TileLang npuir 代码审查、格式校验与 GitHub 工作流技能。用户提及 review、代码审查、PR 前检查、lint、format、ruff、clang-format、commit、push、PR、rebase、upstream、issue、GitHub Actions、gh CLI、规范检查、CI 不通过时必须使用本技能。优先识别行为回归、数值风险、同步风险与测试缺口，其次才是风格问题。
---

# TileLang Review Skill

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## Scope

- pre-PR code review for npuir branch
- format and lint checks aligned with CI
- risk-focused review for correctness, performance, and synchronization
- commit, push, rebase, PR, and issue workflow

---

## Part 1 — Code Review

### Review priorities

1. Behavior regressions
2. Precision and dtype risks
3. Synchronization and pipeline hazards
4. Missing tests
5. Style and format consistency

### Review checklist

1) API style
- New vector paths use v-prefix APIs by default.
- Legacy npuir_xxx usage is only for compatibility.

2) Correctness
- Check boundary/tail logic for tiled loops.
- Check dtype and cast paths for numerical stability.
- Check sync_block_set/wait pairing in mixed pipelines.

3) Performance
- Check redundant copies and excessive casts.
- Check tile size and loop ordering reasonableness.

4) Tests
- Verify at least one focused repro or unit test exists.
- Prefer covering both normal and boundary tile shapes.
- Check whether implementation is based on existing patterns in examples/ and testing/npuir/.
- Flag scratch-built operator code when a close existing template is available.

5) Formatting
- Keep CI style requirements satisfied.

---

## Part 2 — Format Validation

Before creating or updating a PR, run format validation from repository root:

```bash
bash format.sh --files changed_files
```

The changed_files placeholder represents the file list modified in the current branch.

---

## Part 3 — GitHub Workflow

### Branch sync and rebase

```bash
git fetch upstream
git checkout <feature-branch>
git rebase upstream/main
```

### Commit and push

- Write clear, focused commit messages with category prefix
- Run pre-PR format validation
- `git push origin <feature-branch>`

### Pull request

- Target branch: `main`
- PR title must start with a category prefix:
  `[DOCS]` `[Example]` `[Codegen]` `[Pass]` `[Op]` `[Runtime]` `[CI]` `[Fix]` `[Refactor]` `[Benchmark]`
- Include repro steps, scope, and risk notes
- Track CI and update quickly on failures

### Issue creation

- Title prefix: same category convention
- Body must include: environment, minimal repro script, expected vs actual behavior, logs or IR snippet, impact scope and urgency

## References

- references/checklist.txt
- references/pr-workflow.txt
- references/issue-template.txt

## Docs to consult first

- docs/Tilelang-Ascend贡献指南.md
- docs/Tilelang算子调试指南.md
- docs/开发指南.md

## Related skills

- tilelang-error-fixer
- tilelang-debug-helper
