Title: Meridian release gate policy
Provenance: invented local research fixture, not a real company policy.
Version: 3; effective 2026-10-01; supersedes version 2.
Only staged percentage is affected by this gate; it does not establish safety for all traffic.
A normal production expansion requires replay_success_pct >= 98.0.
The same expansion requires p95_turn_latency_ms <= 2400.
Both metrics must use the signed 100-session evaluation in the candidate summary.
The latency condition cannot be waived by a replay-only exception.
A replay exception must name a candidate and be signed by Release Owner.
The current decision cutoff is 2026-10-20T12:00:00Z.
Unapproved drafts and documents after the cutoff cannot change this decision.
