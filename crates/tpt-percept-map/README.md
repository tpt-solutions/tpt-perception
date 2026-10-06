# tpt-percept-map

Spatial mapping for `tpt-perception`.

- **Occupancy grids** — 2-D log-odds (Bresenham) and 3-D log-odds
  (Amanatides–Woo) with clamped, bounded updates; probability queries.
- **TSDF** — truncated signed distance fusion with inverse-range weighting,
  weighted merging, trilinear sampling and zero-crossing surface extraction
  (13-neighbour, robust to diagonal ray steps).
- **SDF** — dense trilinear signed distance grids over bounded boxes,
  numeric gradients for trajectory optimization, margin collision checks.
- **Semantic maps** — labeled point clouds with per-class counts, centroids
  and bounding boxes; class/confidence filters.

`no_std + alloc`. Part of [tpt-perception](../../). Dual-licensed
MIT OR Apache-2.0.
