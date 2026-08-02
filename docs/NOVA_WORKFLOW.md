# NOVA Workflow

1. Register and approve characters, environments, and production assets.
2. Write a JSON creative brief and reference approved asset UUIDs or exact names.
3. Run `cineos nova plan brief.json --output project.json --seed 0`.
4. Compile normally with `cineos compile project.json --output package.json`.
5. Run `cineos nova critique project.json --output critique.json`.
6. Select findings and run `cineos nova revise project.json --critique
   critique.json --output revised-project.json`.
7. Inspect with `cineos nova show revised-project.json`.

Use `--max-scenes`, `--max-shots`, and `--target-duration` to constrain planning.
`--dry-run` validates a plan without writing it. `--json` makes CLI status output
machine-readable. Studio exposes the same generation, critique, and accepted-only
revision operations through `StudioController` and preserves rejected/manual work.
