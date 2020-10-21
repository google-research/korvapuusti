
#include "../carfac.h"

#include <stdio.h>

#include "../carfac/cpp/carfac.h"

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
		float *time_constant_mul) {
  CARParams car_params;
  IHCParams ihc_params;
  AGCParams agc_params;

  if (velocity_scale != NULL) car_params.velocity_scale = *velocity_scale;
  if (v_offset != NULL) car_params.v_offset = *v_offset;
  if (min_zeta != NULL) car_params.min_zeta = *min_zeta;
  if (max_zeta != NULL) car_params.max_zeta = *max_zeta;
  if (zero_ratio != NULL) car_params.zero_ratio = *zero_ratio;
  if (high_f_damping_compression != NULL) car_params.high_f_damping_compression = *high_f_damping_compression;
  if (erb_per_step != NULL) car_params.erb_per_step = *erb_per_step;
  if (erb_break_freq != NULL) car_params.erb_break_freq = *erb_break_freq;
  if (erb_q != NULL) car_params.erb_q = *erb_q;
  if (dh_dg_ratio != NULL) car_params.dh_dg_ratio = *dh_dg_ratio;

  if (stage_gain != NULL) agc_params.agc_stage_gain = *stage_gain;
  if (agc1_scale0 != NULL) agc_params.agc1_scale0 = *agc1_scale0;
  if (agc1_scale_mul != NULL) agc_params.agc1_scale_mul = *agc1_scale_mul;
  if (agc2_scale0 != NULL) agc_params.agc2_scale0 = *agc2_scale0;
  if (agc2_scale_mul != NULL) agc_params.agc2_scale_mul = *agc2_scale_mul;
  if (time_constant0 != NULL) agc_params.time_constant0 = *time_constant0;
  if (time_constant_mul != NULL) agc_params.time_constant_mul = *time_constant_mul;

  auto c = new CARFAC(1, static_cast<float>(sample_rate), car_params,
                      ihc_params, agc_params);
  return carfac{
    c,
    new CARFACOutput(true, true, false, false),
    // We aren't interested in frequencies below 20Hz.
    static_cast<int>(sample_rate / 10),
    sample_rate,
    c->num_channels(),
    {
      c->num_channels(),
      c->pole_frequencies().data(),
    },
  };
}

void delete_carfac(carfac *cf) {
	delete static_cast<CARFACOutput *>(cf->latest_output);
	delete static_cast<CARFAC *>(cf->cf);
}

void carfac_reset(carfac *cf) {
  auto real_cf = static_cast<CARFAC *>(cf->cf);
  real_cf->Reset();
}

void carfac_run(carfac *cf, float_ary buffer, int open_loop) {
  auto real_cf = static_cast<CARFAC *>(cf->cf);
  auto input_map = ArrayXX::Map(reinterpret_cast<const float *>(buffer.data), 1,
                                cf->num_samples);
  auto real_output = static_cast<CARFACOutput *>(cf->latest_output);
  real_cf->RunSegment(input_map, open_loop, real_output);
}

int carfac_bm(carfac *cf, float_ary result) {
  if (result.len != cf->num_samples * cf->num_channels) {
    return -1;
  }

  auto real_output = static_cast<CARFACOutput *>(cf->latest_output);
  memcpy(result.data, real_output->bm()[0].data(), sizeof(float) * result.len);

  return 0;
}

int carfac_nap(carfac *cf, float_ary result) {
  if (result.len != cf->num_samples * cf->num_channels) {
    return -1;
  }

  auto real_output = static_cast<CARFACOutput *>(cf->latest_output);
  memcpy(result.data, real_output->nap()[0].data(), sizeof(float) * result.len);

  return 0;
}
