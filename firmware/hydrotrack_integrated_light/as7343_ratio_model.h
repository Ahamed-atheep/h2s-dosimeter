/*
  as7343_ratio_model.c / .h
  ------------------------------------------------------------------
  DEMO-ONLY calibration model, trained on 5ppm_2_.csv.

  WHAT THIS MODEL ACTUALLY IS:
  A RandomForestRegressor (50 trees, depth 3) trained on 57 AS7343
  spectral readings of PHOTOCOPIED PRINTOUTS of the 13 simulated badge
  images in real_color_badges.zip (5-19 ppm.h, several missing: no 16
  or 18 ppm reading exists in the CSV even though those badges exist
  in the zip). It predicts dose_ppmh from the 12 AS7343 channels,
  normalized by dividing each channel by Clear (see WHY RATIOS below).

  HONEST PERFORMANCE (leave-one-concentration-out cross-validation,
  i.e. "how well does it predict a dose level it never saw during
  training" -- the realistic test for this use case):
      MAE  ~ 2.3 ppm.h
      R^2  ~ 0.50
  On a 5-19 ppm.h span, that is a rough model, not an accurate one.
  Treat its output as "which quarter of the range am I in", not as a
  trustworthy point estimate. See LIMITATIONS below before deploying
  this on a real device for anything safety-relevant.

  WHY RATIOS, NOT RAW COUNTS:
  The raw channel counts in the CSV vary by up to 10x between repeat
  reads of the SAME nominal ppm (e.g. the 5 ppm.h rows range from
  Clear=1527 to Clear=17999/saturated). The AS7343 gain/ATIME/ASTEP
  are fixed in hydrotrack_integrated_light.ino (GAIN_256X, ATIME=29,
  ASTEP=599), so that swing is NOT an auto-gain artifact -- it's
  almost certainly strip positioning/distance/tilt under the sensor,
  and/or copy-to-copy exposure drift from the photocopier. Dividing
  each channel by Clear removes most of the shared illumination-
  intensity component and noticeably improves the signal (e.g. the
  NIR_855nm/Clear ratio correlates with ppm at r=-0.70 vs r=-0.60 for
  raw NIR_855nm counts). It does not fully fix the noise -- see below.

  INPUT ORDER (12 features, each = raw_channel / Clear from the same
  reading):
      input[0]  = F1_405nm  / Clear
      input[1]  = F2_425nm  / Clear
      input[2]  = FZ_450nm  / Clear
      input[3]  = F3_475nm  / Clear
      input[4]  = F4_515nm  / Clear
      input[5]  = F5_550nm  / Clear
      input[6]  = FY_555nm  / Clear
      input[7]  = FXL_600nm / Clear
      input[8]  = F6_640nm  / Clear
      input[9]  = F7_690nm  / Clear
      input[10] = F8_745nm  / Clear
      input[11] = NIR_855nm / Clear
  This is the same column order as 5ppm_2_.csv. Do NOT feed raw
  counts in directly -- compute the /Clear ratio first or the model
  will silently give garbage output (it was never trained on that
  scale).

  LIMITATIONS -- READ BEFORE TRUSTING A NUMBER FROM THIS MODEL:
    1. n=57 samples across 13 concentration levels (~4-5 reads each)
       is a small dataset for a 12-input model. The leave-one-out
       numbers above are the honest estimate of how it generalizes;
       don't expect better in the field.
    2. The underlying simulated badge images (checked directly, not
       assumed) have a clean, smooth, perfectly monotonic sensing-zone
       color vs. ppm relationship -- the EXPIRED and VALID variants at
       the same ppm are pixel-identical, so that's not a confound.
       The huge noise you see in the CSV is therefore coming from the
       print + physical-scan step (photocopier and/or sensor
       placement), not from the source data. Fixing that acquisition
       step (see suggestions in chat) will help far more than any
       change to the ML model.
    3. Bigger, harder-to-fix issue: this model calibrates AS7343
       spectral response to PHOTOCOPIER TONER reproducing an sRGB
       target, not to actual lead-acetate/H2S staining chemistry.
       Two materials can match under one summary color (metamerism)
       and still have different full reflectance spectra across
       405-855nm -- which is exactly what a 12-channel spectral sensor
       is sensitive to. This model may not transfer to a real exposed
       PbAc strip even if it fits the printed proxy well. It should be
       validated against real chemically-exposed strips before being
       trusted for real dose readings, same caveat as
       demo_print_calibration.h already flags for the single-channel
       print model.
    4. Range: trained on 5.0-19.0 ppm.h only, and is missing 16 and 18
       ppm.h entirely (present in the image set, absent from the CSV).
       Don't extrapolate outside 5-19 ppm.h.

  See dose_model_light.c/.h for the original photo-pixel-trained model
  and demo_print_calibration.c/.h for the single-channel printed-badge
  reflectance model this complements.
*/

#ifndef AS7343_RATIO_MODEL_H
#define AS7343_RATIO_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

// channels[12] must be raw_channel / Clear, in the exact order:
// F1, F2, FZ, F3, F4, F5, FY, FXL, F6, F7, F8, NIR (matches CSV header).
// Returns predicted dose in ppm.h, valid only within ~5-19 ppm.h.
double score_as7343_ratio(const double channels_over_clear[12]);

#ifdef __cplusplus
}
#endif

#endif
