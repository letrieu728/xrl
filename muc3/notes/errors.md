# Errors

- `dbarts`: `object '.Random.seed' not found` during the first CartPole BART fit; fixed by initializing R RNG.
- Full CartPole `infer_DBN()` exits during BART structure learning without a Python traceback after successful 200-episode rollout. This occurred with both HTE and expectation paths.
- Isolated CartPole HTE target `cart_position` failed with `AssertionError` in `HTE.is_significant`: 125 posterior draws were produced, but the repository hard-codes an assertion requiring 500. Cause: the earlier Windows stability setting changed dbarts from four chains to one.
- The command runner can return before a native R child has flushed its output. Checkpoint inspection subsequently confirmed that isolated target `cart_velocity` completed and wrote its result. Treat tool output alone as insufficient completion evidence; use result checkpoints.
