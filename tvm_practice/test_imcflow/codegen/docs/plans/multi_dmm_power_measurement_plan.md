# Multi-DMM Power Measurement Plan

## Goal

Allow one TVM power-measurement region to acquire current traces from several
DMMs concurrently.  The first target configuration is GPIB addresses 1, 2,
and 4, configured by the TVM-owned JSON file
`codegen/power_config/dmm_gpib124.json`.

The measurement server remains the only host that opens PyVISA devices.  The
board never receives a JSON path and never chooses VISA addresses through the
TCP protocol.

## Non-goals

- Do not change the legacy direct bridge protocol or add the old RPC layer.
- Do not introduce tags, clock synchronization, or a new sampling mode.
- Do not make the board copy files to meas-2.
- Do not perform hardware validation as part of this change; the user will
  run the actual DMM/board experiment.

## Current state and constraint

- `IMCFLOW_POWER_DMM_NAME` selects one logical DMM at TVM compile time.
- `ext_codegen.py` emits one `dmm_config_t` and invokes
  `dmm_start_current*(1, ...)`.
- The direct C client and bridge already support `n_dmms > 1`; they arm all
  supplied DMMs, issue the bridge GO transaction, and return one `RESULT` per
  config in its input order.
- The bridge is configured locally when it starts, using `--config`; it cannot
  safely accept a config path from the board.

## Public interface

### DMM list

Add `IMCFLOW_POWER_DMM_NAMES`, a comma-separated, ordered list of logical
names from the bridge JSON.

```bash
export IMCFLOW_POWER_DMM_NAMES=VDD,DDA,DDC
```

Compatibility rules:

1. If `IMCFLOW_POWER_DMM_NAMES` is unset, preserve the existing
   `IMCFLOW_POWER_DMM_NAME` behaviour (default `DMM_GPIB3`).
2. If the plural variable is set, it is authoritative and the singular
   variable is ignored with a compile-time warning.
3. Reject an empty element, duplicate name, or more than a fixed safe maximum
   (for example 16) before code generation.
4. All DMMs in a run initially share existing global acquisition settings:
   NPLC, interval, sample count, range, reset, start timeout, and result
   timeout.  Per-DMM acquisition settings are deliberately deferred.

### Measurement-server setup

Keep the JSON in TVM:

```
codegen/power_config/dmm_gpib124.json
```

Use `scripts/start_power_bridge_meas2.sh` to:

1. copy that JSON with `scp` to `meas-2:/tmp/imcflow_power_config/`;
2. activate the `imcflow` conda environment on meas-2;
3. launch `measure-bridge-daemon --config <copied-json>` over SSH.

The script's default port remains 9911 so it does not replace the existing
GPIB3 bridge on 9910.  The evaluation command uses the chosen port through
`DMM_BRIDGE_HOST` and `DMM_BRIDGE_PORT`.

## Implementation steps

### 1. Parse and record the DMM list in TVM

In `python/tvm/relay/backend/contrib/imcflow/ext_codegen.py`:

- replace the internal scalar `POWER_DMM_NAME` with immutable
  `POWER_DMM_NAMES`;
- parse/validate the environment once at module configuration time;
- retain the legacy singular fallback exactly as described above;
- include both the ordered names and the source environment variable in
  generated `power/build_metadata.json` so a trace is reproducible.

Update `codegen/test.py` metadata collection to report `dmm_names` as a list,
while optionally retaining `dmm_name` when there is exactly one name for
backward-compatible readers.

### 2. Generate an array of DMM configs

Refactor `generatePowerMeasureStart()` to emit:

```c
dmm_config_t _power_dmm_cfgs[] = { ... };
dmm_start_current(n_dmms, _power_dmm_cfgs);
```

(`dmm_start_current_now` is selected for now mode as today.)

Every config receives its own `server_ofname`.  Derive it deterministically
from the existing region/model/tile suffix plus a filesystem-safe logical DMM
name, for example:

```
<prefix>_model_VDD.txt
<prefix>_model_DDA.txt
<prefix>_model_DDC.txt
```

Sanitize any character outside `[A-Za-z0-9_.-]` and reject a collision after
sanitization.  This avoids relying on bridge-side filename interpretation.

### 3. Consume all results before closing

Refactor `generatePowerMeasureEnd()` to call `dmm_wait_result()` or
`dmm_get_result_now()` exactly once per configured DMM, then call `dmm_close`
once.

- Print the logical config name and the returned bridge name for diagnostics.
- Treat a failed result from any DMM as a measurement failure, cleanly close
  the bridge session, and follow the existing kernel failure path.
- Do not assume response order alone: compare the returned name with the
  expected config name and report a mismatch.
- Preserve the current timing boundary: measurement start/end placement for
  MODEL, REGION, and TILE scope does not change.

### 4. Artifact collection and plotting

Audit `run_dataset_eval.sh` and the legacy artifact/plot helper so their file
collection uses the per-run `power/` directory rather than one expected
`*.txt` filename.

For a multi-DMM run:

- retain every raw trace as-is;
- generate one PNG per DMM, named after the DMM logical name;
- write a small `power_metadata.json` mapping DMM name, VISA address as
  declared in the source config (if available locally), raw filename, sample
  count, interval, and plot filename;
- do not sum or align rails automatically.  The DMMs share a bridge GO
  transaction, but their device clocks/sample phase are still distinct.

Keep the existing single-DMM `power_trace.png` name for compatibility.  For
multi-DMM, also create `power_trace_DMM_GPIB*.png`; an optional combined plot
may be added later only when its time-axis semantics are explicitly defined.

### 5. Documentation and command examples

Update `docs/power_v2_quickstart.md` with:

```bash
./scripts/start_power_bridge_meas2.sh \
  --config power_config/dmm_gpib124.json --port 9911

export DMM_BRIDGE_HOST=<meas-2 board-facing IP>
export DMM_BRIDGE_PORT=9911
export IMCFLOW_POWER_DMM_NAMES=VDD,DDA,DDC
```

Document that the DMM list is compile-time input: changing it requires
re-running TVM compilation and rebuilding the host binary.  Starting the
bridge with a different JSON only requires restarting the bridge.

## Verification (no physical DMM required)

1. **Parser unit tests**: singular fallback, valid plural list, whitespace,
   duplicate, blank element, excessive count, and plural-overrides-singular.
2. **Generated-C inspection test**: assert the config array contains three
   names, start receives `3`, end contains three result calls, and each
   server output name is unique/sanitized.
3. **C API mock/loopback test**: use the existing bridge test mechanism to
   verify START encodes three configs and RESULT parsing occurs three times.
4. **Artifact test**: feed three synthetic raw traces to the collection/plot
   helper and assert all raw files, metadata entries, and three PNGs exist.
5. **Regression**: run existing single-DMM unit tests unchanged with only
   `IMCFLOW_POWER_DMM_NAME=DMM_GPIB3` set.

Physical validation is intentionally left to the user: start the meas-2
bridge with the supplied script, compile with the plural variable, and run a
short MODEL or REGION power capture.

## Completion criteria

- A three-DMM compile emits one concurrent bridge session containing GPIB1,
  GPIB2, and GPIB4 logical configs.
- A run writes and retrieves three distinct raw traces and three identifiable
  plots under the normal `eval_dir/<model>/power/<run_id>/` result directory.
- Existing GPIB3-only commands and their artifact layout continue to work.
