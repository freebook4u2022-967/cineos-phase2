# NOVA Director Alpha

NOVA converts a structured creative brief into renderer-independent story, scene,
shot, camera, performance, pacing, and continuity plans. The built-in
`rule-based` provider is deterministic and offline. Optional planning providers
implement `PlanningProvider`; NOVA does not require or privilege an external LLM.

NOVA resolves every required character and environment against the approved Asset
Registry before planning. A missing identity stops planning instead of creating a
substitute. The resulting core `MovieProject` remains consumable by the Film
Compiler, while the persisted `nova` section retains the auditable directing plan.

Scene and shot rationales explain production intent. They are concise summaries,
not private model reasoning or chain-of-thought.

## Python API

```python
director = NOVADirector(registry)
plan = director.create_plan(brief, seed=7, planner="rule-based")
package = FilmCompiler().compile(plan.project)
```

Determinism covers the brief, approved registry, provider/version, seed, and
limits. Revision copies the plan, changes only accepted finding targets, retains
stable IDs and approved assets, and appends a revision history entry.
