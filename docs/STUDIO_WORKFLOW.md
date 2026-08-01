# Studio Alpha workflow

1. Create a project or open a `.cineos.json` project and edit its metadata.
2. Attach canonical assets and explicitly approve identity references.
3. Add scenes and shots; reorder through the controller so the Core timeline stays synchronized.
4. Run **Validate** and resolve every reported Core/asset/reference issue.
5. Compile a deterministic FilmPackage and inspect renderer environment and hardware status.
6. Build conditioning, then dry-run the film through Atlas Runtime.
7. Render a selected shot and inspect identity, wardrobe, props, environment, and temporal validation.
8. Approve, reject, rerender, or require manual review. Recovery attempts remain recorded.
9. Build or resume the complete film. Cancellation is cooperative and preserves build state.
10. Export the MP4, package, reports, recovery history, and checksums; open the output directory to inspect them.

Studio coordinates this sequence but does not reimplement compiler, renderer,
validation, recovery, assembly, or export rules.
