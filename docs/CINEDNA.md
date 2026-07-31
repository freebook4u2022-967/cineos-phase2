# CineDNA v1

CineDNA is CINEOS's renderer-independent, persistent character identity format.
It records only explicitly approved references and human-authored descriptions;
it does not contain face embeddings, recognition results, or renderer conditioning.

## Authoring

A canonical character asset must have at least one reference whose
`approval_status` is `approved`. Its metadata must contain a `cinedna` object
with explicit `face` and `body` objects. Optional keys are `wardrobe`, `voice`,
`motion`, `expressions`, `constraints`, and `metadata`. Missing visual attributes
remain empty: the builder never guesses them.

The stable profile identity is the character asset UUID. `profile_version`
identifies revisions, while a SHA-256 content hash covers canonical JSON for all
profile content other than the hash itself. Registry updates retain older
versions.

## CLI

The commands use `assets.json` and `cinedna.json` by default. Alternate files can
be selected with `--registry` and `--profiles` before the action.

```console
cineos cinedna build CHARACTER_UUID
cineos cinedna list
cineos cinedna show CHARACTER_UUID
cineos cinedna validate CHARACTER_UUID
cineos cinedna export CHARACTER_UUID --output profile.json
```

Every action supports `--json`. Exported profiles are canonical, deterministic
JSON and verify their content hash when loaded.
