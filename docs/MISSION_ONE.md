# Mission One

Mission One connects directed three-shot planning to a portable CogVideoX-2B Colab package. Compile the example with `cineos mission-one compile examples/mission_one/creative-brief.json --output-dir build`, upload an exported package to the notebook, render sequentially, and import `render-results.json` for honest verification.

The production target is exactly one character, one environment, three 6–8 second shots, and 18–24 seconds total. Dialogue voice is generated separately and attached during assembly; the video prompt requests only visible speaking behavior. Real rendering is a manual, hardware-gated validation and is not represented as an automated test.
