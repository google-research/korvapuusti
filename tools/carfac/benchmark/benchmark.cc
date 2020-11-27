
#include <stdio.h>
#include <vector>

#include "carfac/cpp/carfac.h"

int main(int argc, char** argv) {
  auto cf = new CARFAC(1, 48000.0, CARParams(), IHCParams(), AGCParams());
  std::vector<float> buf(4800);
  auto input_map = ArrayXX::Map(buf.data(), 1, buf.size());
  auto output = new CARFACOutput(true, true, false, false);
  for (int i = 0; i < 10; i++) {
	  cf->RunSegment(input_map, false, output);
  }
  delete output;
  delete cf;
}
