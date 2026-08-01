# Reference Conditioning Contract

Conditioning schema `1.0` represents one compiled shot. It contains character
identity and CineDNA revision constraints, environment, wardrobe, prop/vehicle,
camera, cross-shot continuity, approved reference IDs, a deterministic seed,
and explicit renderer capability requirements.

## Determinism and authority

The builder accepts a `FilmPackage`, `AssetRegistry`, and `CineDNARegistry`.
Canonical JSON uses sorted keys and compact separators; its SHA-256 digest is
stored as `content_hash`. Missing assets, profiles, relationships, approvals,
and contradictory locks are errors: the builder never supplies creative data.
The deterministic seed derives solely from the Film Package hash and shot ID.

## Renderer negotiation

Plugins advertise `RendererCapabilities`. Before execution,
`renderer.accepts_conditioning(package)` checks resolution, FPS, maximum
duration and character count, plus image, multiple-reference, face-identity,
control-image, and motion-reference features. Unsupported requirements produce
a `UnsupportedRendererCapabilities` error describing every mismatch.

## CLI

```console
cineos condition build SHOT-ID
cineos condition validate conditioning.json
cineos condition show conditioning.json
cineos condition export SHOT-ID --output conditioning.json
```

Source paths may be selected with `condition --package`, `--registry`, and
`--profiles`. Build writes `<shot-id>.conditioning.json`; export requires an
explicit destination.
