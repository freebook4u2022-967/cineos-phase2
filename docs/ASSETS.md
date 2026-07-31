# Canonical assets and references

CINEOS keeps production identity in an external, deterministic asset registry. An asset has a stable UUID, type, name, description, integer version, tags, JSON metadata, creation/update timestamps, and a SHA-256 content hash. Supported types are character, environment, wardrobe, prop, vehicle, storyboard, and generic reference.

## Reference workflow

Reference records contain a UUID, media path, MIME type, view, SHA-256 checksum, optional pixel dimensions, approval status (`pending`, `approved`, or `rejected`), notes, priority, and source/provenance. Views include front, three-quarter, both profiles, rear, full-body, close-up, expression, wardrobe, prop, and environment. Multiple approved records may point at one character.

The registry stores paths and checksums only: it never copies, uploads, or embeds the referenced file. `AssetValidator.validate(registry, check_files=True)` opts in to local existence and SHA-256 checks. Ordinary schema validation does not require mounted production storage.

## Relationships and packages

Typed directed links cover character-to-wardrobe, character-to-prop, character-to-vehicle, scene-to-environment, and storyboard-to-scene. Both endpoints must resolve before adding a link. Film Packages contain stable asset identity metadata, never image bytes or reference media paths.

Project JSON connects to the catalog with an `asset_registry` file path and an
`asset_ids` array of UUIDs. Registry paths are relative to the project file.
Project validation rejects UUIDs absent from that registry; compilation filters
the package asset manifest to the selected identities and does not serialize
their reference records.

## JSON and CLI

Assets, relationships, keys, and tags are sorted before encoding, so an unchanged load/save round trip produces identical bytes.

```console
cineos assets add-character character.json
cineos assets add-environment environment.json
cineos assets list
cineos assets show 6fa459ea-ee8a-3ca4-894e-db77e160355e
cineos assets validate
cineos assets export --output assets-export.json
```

Commands use `assets.json` by default. Pass `cineos assets --registry PATH ...` to select another registry, and `--json` for machine-readable output.
