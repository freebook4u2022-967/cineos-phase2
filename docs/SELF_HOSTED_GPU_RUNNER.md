# Optional self-hosted GPU runner

Use a dedicated, access-controlled GitHub runner labelled `self-hosted`, `linux`,
`x64`, and `nvidia-gpu`. Pin the NVIDIA driver/CUDA stack, isolate credentials,
clean workspaces, record `cineos release diagnose` and hardware reports, and
apply job timeouts. A separate manually dispatched workflow may install an
approved renderer, run real-renderer integration and hardware benchmarks, and
validate a release candidate. Never run expensive inference in standard CI;
never treat a missing GPU result as a measured pass.
