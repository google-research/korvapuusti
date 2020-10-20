#ifndef KORVAPUUSTI_TOOLS_CARFAC_CARFAC_H_
#define KORVAPUUSTI_TOOLS_CARFAC_CARFAC_H_

#ifdef __cplusplus
#include "carfac/cpp/carfac.h"

extern "C" {
#endif

typedef struct float_ary {
  int len;
  float *data;
} float_ary;

typedef struct const_float_ary {
  int len;
  const float *data;
} const_float_ary;

typedef struct carfac {
  void *cf;
  void *latest_output;
  int num_samples;
  int sample_rate;
  int num_channels;
  const_float_ary poles;
} carfac;

carfac create_carfac(
		int sample_rate,
		
		float *velocity_scale,
		float *v_offset,
		float *min_zeta,
		float *max_zeta,
		float *zero_ratio,
		float *high_f_damping_compression,
		float *erb_per_step,
		float *erb_break_freq,
		float *erb_q,
		float *dh_dg_ratio,
		
		float *stage_gain,
		float *agc1_scale0,
		float *agc1_scale_mul,
		float *agc2_scale0,
		float *agc2_scale_mul,
		float *time_constant0,
		float *time_constant_mul);

void delete_carfac(carfac *cf);

void carfac_run(carfac *cf, float_ary buffer, int open_loop);

int carfac_bm(carfac *cf, float_ary result);

int carfac_nap(carfac *cf, float_ary result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // KORVAPUUSTI_TOOLS_CARFAC_CARFAC_H_
