# Review of 27ba187

The project is a capable, device-specific prototype. Its strongest parts are
explicit UNKNOWN cards, foundation contradiction checks, temporal foundation
tracking, failed-move exclusions, session logging, and bounded search. These
are useful defenses against noisy screenshots. Live correctness still depends
heavily on the calibrated layout and on unverified card identities.

## Fixes in this working tree

- Validate ranks against the actual deck, rather than accepting every string
  except `?`. Invalid tableau and waste identities remain UNKNOWN slots.
- Require a valid identity and positive score even when a foundation read has
  `reliable=True`. Nonempty malformed foundation reads are reported as
  unreliable rather than silently treated as empty.
- Remove the ADB-only `shell` prefix when executing commands through Android
  `su -c`, and quote arguments so spaces and shell metacharacters survive.
- Correct the README's game model, execution defaults, and repository links;
  add runtime and development dependency manifests.
- Add nine regression tests for the trust rules and command construction.

## Remaining priorities

1. **Detect duplicate visible identities.** A dry run of
   `Gameplay/frame_0100.png` reads the Queen of Spades in both columns 1 and 2.
   The state converter checks visible cards against foundations, but does not
   reject duplicates across tableau/waste. At least one of those reads must
   be wrong; the solver still plans moves. Add annotated screenshot fixtures
   and a policy for retrying or masking conflicting reads before executing
   a batch. Choosing which duplicate to trust needs evidence from the images
   or temporal history.
2. **Validate stock/waste accounting over an entire draw cycle.** The reader
   detects visible waste corners, while the state converter treats that list
   as the entire waste pile. If earlier waste cards are fully obscured, they
   are counted as undrawn stock. Separately, simulated redeals reset stock to
   the initial `stock_total`, including cards already played out of waste.
   Verify the UI's rendering and track the actual remaining stock/waste pool;
   changing a formula alone would not resolve both problems.
3. **Measure move correctness.** The failing-deal test currently checks that
   execution returns expected types. It does not compare extracted cards or
   selected moves with ground truth. Add labeled boards and tests for legal
   moves and card conservation, then record actual device move success rates.
4. **Consolidate active entry points.** The root modules, `pipeline/`,
   `*_WORKING.py`, and archived scripts make it easy to run a stale version.
   Keep one documented implementation and clearly identify experimental files.
5. **Calibrate gesture timing on the device.** This commit lowers swipe
   duration to 560–780 ms, while nearby comments describe failures below about
   700 ms. Static tests cannot establish whether that change improves speed
   without increasing rejected drags. Use session logs and one move per read
   to measure both before tuning further.

## Validation

`python -m pytest tests test_logging.py -q`: 21 passed, 4 subtests passed.

`python solitaire_auto_bot.py --sim Gameplay/frame_0100.png --solver race`:
completed and emitted a two-move simulated batch. This verifies pipeline
execution, not the accuracy of that screenshot's extracted identities.

Device commands were mocked in regression tests. No live gestures or device
timing validation were performed. Changes are uncommitted for inspection.
