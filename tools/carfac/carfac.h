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

carfac create_carfac(int sample_rate);

void delete_carfac(carfac *cf);

void carfac_run(carfac *cf, float_ary buffer);

int carfac_bm(carfac *cf, float_ary result);

int carfac_nap(carfac *cf, float_ary result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // KORVAPUUSTI_TOOLS_CARFAC_CARFAC_H_
