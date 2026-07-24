---
name: implement
description: Drive the architect→workhorse flow — turn an approved plan into delegated Sonnet implementation with test and review gates, keeping the main (Opus/Fable) thread as coordinator.
disable-model-invocation: true
argument-hint: [spec, plan-file path, or task description]
---

# /implement — delegated implementation flow

You (the main thread, running the strong model) act as architect and coordinator.
Delegate the typing to the `implementer` agent (Sonnet); keep design judgment and
final arbitration here. Do not make non-trivial edits yourself in this flow.

## Steps

1. **Get a spec.** If the argument is already a spec or points at an approved plan
   (plan-mode output, a `docs/*-plan.md`, an issue), use it. Otherwise spawn the
   `architect` agent to produce one, review it yourself, and fix gaps before
   proceeding. A spec must name files, changes, constraints, and test filters.
2. **Write the spec down.** Save it to the scratchpad (or `docs/` if it should be
   committed) so it survives context compaction and can be handed to agents verbatim.
3. **Delegate.** Spawn one `implementer` agent per independent work package —
   in parallel only when packages touch disjoint files. Paste the full relevant
   spec section into each prompt; agents see nothing else. Ordered packages run
   sequentially, feeding each agent the previous report.
4. **Test gate.** Spawn `test-runner` with the spec's test filter(s). On failures,
   send the digest back to the responsible `implementer` (SendMessage to continue
   it with context intact) rather than fixing in the main thread.
5. **Review gate.** For multi-file or kernel/numeric changes, spawn `code-reviewer`
   on the diff. Triage its findings yourself — you are the arbiter; drop false
   positives, route real ones back to the implementer.
6. **Report.** Summarize what changed, test results, and anything left open.
   Commit only if the user asked for that.

## Rules of thumb

- Trivial one-file fixes don't need this ceremony — just do them directly.
- Never run the full `dotnet test` suite in the main thread; that's what
  `test-runner` is for.
- If an implementer reports a spec gap, resolve it here (you have the design
  context) and send the correction back — don't let Sonnet redesign.
