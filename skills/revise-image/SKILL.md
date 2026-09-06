---
name: revise-image
description: Refine an accepted image, recreate a reference, restyle a scene, or swap a character while retaining the composition and details the user already likes. Useful for several rounds of visual corrections.
---

Keep track of the target, the accepted result, and the requested change as different things. A new frame becomes the latest attempt; it is not automatically the best one.

Read the current state and inspect relevant images by ID. Historical placeholders do not contain pixels. Use a crop if a detail such as a face, landmark, or object placement cannot be read at preview scale. Pin references with their purposes in the creative brief when several images describe different parts of the request.

Retain a short `must_keep` list and the user's accepted compromises. Record approval only when the user expresses it. For a small correction, edit the relevant phrase; for accumulated or contradictory clauses, consolidate the prompt against the brief. Preserve effective user wording. Describe geometry directly: “a broad base tapering to a flat top” does not introduce a literal rocket.

For a restyle, retain subject, action, placement, and mood while replacing the medium or marks. Avoid copying content from the style reference. For a character swap, retain the accepted staging while replacing character-specific features. For a substitute detail, preserve its role in the scene: a connection between islands, quiet motion, a dark glitter material, or empty space may matter more than the object's name.

Generation is text-to-image. A same-seed prompt change can rearrange the entire frame. Use `restore_image_settings` when returning to an earlier recipe; it copies the exact stored seed instead of transcribing a rounded number. Use `fresh_seed=true` when deliberately testing a new sample.

On each important attempt, name the correction in `change_note`, inspect the actual returned frame, then use `review_attempt` to record the observed change and any regression. Compare against the best accepted image. A missing detail remains missing even when the prompt describes it. Distinguish uncertainty from success, and predictions from rendered results.

Respect requests to discuss or edit without generating. The turn's allowance is a ceiling. After repeated regressions, stop and compare the strongest candidates with the unresolved requirement; do not keep adding increasingly specific clauses or claim a seed will guarantee the missing element. Never trade away an accepted requirement silently to call a result finished.
