# Prime Intellect GPU setup notes

## What was attempted
- Installed Prime CLI successfully via:
  - `uv tool install prime`
- Configured API key with `prime config set-api-key`
- Configured SSH private key path with `prime config set-ssh-key-path ~/.ssh/id_ed25519`
- Verified config with `prime config view`
- Queried availability with `prime availability list`

## Practical observation
The cheapest interesting single-GPU options visible at setup time included:
- `A6000_48GB` on massedcompute at about `$0.54/hr`
- `L40S_48GB` on massedcompute at about `$0.82/hr`
- `H100_80GB` at about `$2.29/hr`

## Pod lifecycle tested
- Created pod successfully:
  - name: `nc-mini-a6000`
  - GPU: `A6000_48GB x1`
  - image: `ubuntu_22_cuda_12`
- Pod reached `ACTIVE` state.
- SSH access failed because the offered local key was rejected by the remote host.
- Attempted to inspect Prime SSH key APIs; docs show a separate SSH-key upload endpoint.
- GET `/api/v1/ssh_keys/` worked and returned zero keys on the account.
- POST upload attempt for the SSH public key returned `401 Unauthorized` with the current token/permissions.
- Pod was terminated immediately after the SSH failure to avoid wasting spend.

## Honest conclusion
Prime compute is usable in principle, and pod creation works with the current token.
However, remote training is currently blocked by **SSH key enrollment / permission issues**, not by GPU availability.

## Next unblock options
1. Use a token with SSH-key write permission.
2. Upload a public SSH key manually in the Prime dashboard, then reprovision.
3. Use an alternate remote environment if immediate GPU execution is more important than Prime specifically.

## Relevance to project
A GPU VM is still the right place for heavier GRU / Transformer / Mamba-style training.
The final marimo notebook should remain CPU-friendly, but the heavy experiments can be run offline on GPU and then summarized in the notebook.
