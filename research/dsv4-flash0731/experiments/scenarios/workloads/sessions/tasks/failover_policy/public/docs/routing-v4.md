Title: Harbor routing policy
Provenance: invented local research fixture, not a real incident.
Version: 4; effective 2026-10-10.
Unless an approved scoped override applies, interactive traffic uses east as primary and west as fallback.
Batch traffic uses west as primary and east as fallback.
Traffic must remain in its named data-residency region group.
A signed incident override takes precedence only for the tenant and time window it names.
