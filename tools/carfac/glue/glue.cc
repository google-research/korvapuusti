
#include "../carfac.h"

#include <stdio.h>
#include <vector>

#include "../carfac/cpp/carfac.h"

void set_default_params(
		float *velocity_scale,
		float *v_offset,
		float *min_zeta,
		float *max_zeta,
		float *zero_ratio,
		float *high_f_damping_compression,
		float *erb_per_step,
		float *erb_break_freq,
		float *erb_q,
		
		float *tau_lpf,
		float *tau1_out,
		float *tau1_in,
		float *ac_corner_hz,

		float *stage_gain,
		float *agc1_scale0,
		float *agc1_scale_mul,
		float *agc2_scale0,
		float *agc2_scale_mul,
		float *time_constant0,
		float *time_constant_mul,
		float *agc_mix_coeff) {
  CARParams car_params;
  IHCParams ihc_params;
  AGCParams agc_params;

  *velocity_scale = car_params.velocity_scale;
  *v_offset = car_params.v_offset;
  *min_zeta = car_params.min_zeta;
  *max_zeta = car_params.max_zeta;
  *zero_ratio = car_params.zero_ratio;
  *high_f_damping_compression = car_params.high_f_damping_compression;
  *erb_per_step = car_params.erb_per_step;
  *erb_break_freq = car_params.erb_break_freq;
  *erb_q = car_params.erb_q;

  *tau_lpf = ihc_params.tau_lpf;
  *tau1_out = ihc_params.tau1_out;
  *tau1_in = ihc_params.tau1_in;
  *ac_corner_hz = ihc_params.ac_corner_hz;

  *stage_gain = agc_params.agc_stage_gain;
  *agc1_scale0 = agc_params.agc1_scales[0];
  *agc1_scale_mul = agc_params.agc1_scales[1] / agc_params.agc1_scales[0];
  *agc2_scale0 = agc_params.agc2_scales[0];
  *agc2_scale_mul = agc_params.agc2_scales[1] / agc_params.agc2_scales[0];
  *time_constant0 = agc_params.time_constants[0];
  *time_constant_mul = agc_params.time_constants[1] / agc_params.time_constants[0];
  *agc_mix_coeff = agc_params.agc_mix_coeff;
}

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
		
		float *tau_lpf,
		float *tau1_out,
		float *tau1_in,
		float *ac_corner_hz,

		float *stage_gain,
		float *agc1_scale0,
		float *agc1_scale_mul,
		float *agc2_scale0,
		float *agc2_scale_mul,
		float *time_constant0,
		float *time_constant_mul,
		float *agc_mix_coeff) {
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

  if (tau_lpf != NULL) ihc_params.tau_lpf = *tau_lpf;
  if (tau1_out != NULL) ihc_params.tau1_out = *tau1_out;
  if (tau1_in != NULL) ihc_params.tau1_in = *tau1_in;
  if (ac_corner_hz != NULL) ihc_params.ac_corner_hz = *ac_corner_hz;

  if (stage_gain != NULL) agc_params.agc_stage_gain = *stage_gain;
  std::vector<FPType> agc1_scales(agc_params.num_stages);
  std::vector<FPType> agc2_scales(agc_params.num_stages);
  std::vector<FPType> time_constants(agc_params.num_stages);
  agc1_scales[0] = agc1_scale0 == NULL ? 1.0 : *agc1_scale0;
  for (int i = 1; i < agc_params.num_stages; ++i) {
	  agc1_scales[i] = agc1_scales[i - 1] * (agc1_scale_mul == NULL ? sqrt(2.0) : *agc1_scale_mul);
  }
  agc_params.agc1_scales = agc1_scales;
  agc2_scales[0] = agc2_scale0 == NULL ? 1.65 : *agc2_scale0;
  for (int i = 1; i < agc_params.num_stages; ++i) {
	  agc2_scales[i] = agc2_scales[i - 1] * (agc2_scale_mul == NULL ? sqrt(2.0) : *agc2_scale_mul);
  }
  agc_params.agc2_scales = agc2_scales;
  time_constants[0] = time_constant0 == NULL ? 0.002 : *time_constant0;
  for (int i = 1; i < agc_params.num_stages; ++i) {
	  time_constants[i] = time_constants[i - 1] * (time_constant_mul == NULL ? 4.0 : *time_constant_mul);
  }
  agc_params.time_constants = time_constants;
  if (agc_mix_coeff != NULL) agc_params.agc_mix_coeff = *agc_mix_coeff;

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
