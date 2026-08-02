# Latent face objectives (background preservation, face focus)

These features reweight or anchor training using an offline face / subject mask
(`face_mask_source`: `manifest`, `subject_mask`, or `face_parse`). Soft
rectangle masks from `faces.json` use `face_detector.mask_edge_sharpness`.

```yaml
train:
  face_mask_source: manifest
  face_detector:
    mask_edge_sharpness: 80.0
  background_preservation:
    enabled: true
    weight: 0.01
  face_focus:
    enabled: false
    power: 1.0
```

If `face_detector` is omitted, soft-mask settings are read from
`background_preservation` for backward compatibility. Face-mask machinery loads
when **either** `face_focus` or `background_preservation` is enabled.

## Background preservation

Keeps a face LoRA from learning the scenery in its reference images. Compares the
trainable LoRA UNet with the frozen base UNet **outside** the trained region
(outside a soft face box for `manifest`, outside the subject silhouette for
`subject_mask`, or outside the parse-defined background for `face_parse`).

`weight` is the strength of this second loss. Start at `0.01` and use the
smallest value that prevents scene memorization without weakening the learned
face.

The base-UNet branch is frozen. Soft masks are detached before the loss is
calculated. Rows without a usable mask are skipped rather than anchoring the
whole frame to the base model.

## Face-focused identity loss

Instead of adding an opposing background term, weights the **main** diffusion
(identity) loss by the soft face / subject mask so training pressure lands inside
the trained region while the rest of the frame receives little or no gradient.

`power` is in `[0, 1]` and shapes the mask via `mask ** power`. At `1.0` the mask
is used as-is; at `0.0` every weight is pulled toward `1.0`, disabling the effect.
Rows without a usable region fall back to an all-ones weight.

Face focus and background preservation act on largely disjoint regions when
combined: identity loss inside the subject, base-model matching outside.

## Restricting the trained timestep band

```yaml
  timestep_range:
    min_fraction: 0.0
    max_fraction: 1.0
```

Limits which diffusion timesteps the loss trains on, as fractions of
`num_train_timesteps`. Smaller timesteps are lower noise. Setting `max_fraction`
below `1.0` skips high-noise steps, preserving the base model's global
composition while injecting identity at low noise.

This implementation uses the normal DDPM training scheduler. EDM style training
remains unsupported for face-mask objectives.
