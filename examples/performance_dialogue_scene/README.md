# Dialogue performance example

Build `performance.json` from an approved AudioProject:

```sh
cineos performance build shot-001 --audio-project audio-project.json --output performance.json
cineos performance validate performance.json
cineos performance inspect performance.json --json
```
