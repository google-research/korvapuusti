#!/bin/bash

export BEST=undefined
export VAR0=1
export VAR1=1
export VAR2=1
export VAR3=1
declare -a MASK_FREQUENCIES=(60.0 568.0 1370.0 2908.0 7000.0)
# declare -a MASK_FREQUENCIES=(7000.0)
declare -a PROBE_LEVELS=(30 60)
# declare -a PROBE_LEVELS=(30)
declare -a MASKING_LEVELS=(40 60 80)
# declare -a MASKING_LEVELS=(80)
echo "Mask_freq, Probe_level, Mask_level, Loss, Var0, Var1, Var2, Var3" &> all_vars_bark_cauchy.txt
for MASK_FREQ in "${MASK_FREQUENCIES[@]}"; do
  export MASK_FREQ=${MASK_FREQ}
  for PROBE_LEVEL in "${PROBE_LEVELS[@]}"; do
    export PROBE_LEVEL=${PROBE_LEVEL}
    for l in "${MASKING_LEVELS[@]}"; do
      export VAR0=1
      export VAR1=1
      export VAR2=1
      export VAR3=1
      export MASKING_LEVEL=$l
      mkdir logs/${MASK_FREQ}_${PROBE_LEVEL}_${l}
      ./simplex_fork.py ./loss.py 4 0.01 500 &> logs/${MASK_FREQ}_${PROBE_LEVEL}_${l}/${MASK_FREQ}_${PROBE_LEVEL}_${l}.txt
      ./get_logs.sh logs/${MASK_FREQ}_${PROBE_LEVEL}_${l}/${MASK_FREQ}_${PROBE_LEVEL}_${l}.txt logs/${MASK_FREQ}_${PROBE_LEVEL}_${l}/logs.txt
      bestVars=`cut -f1,2,3,4,5 -d"," logs/${MASK_FREQ}_${PROBE_LEVEL}_${l}/logs.txt | sort -n | head -1`
      echo "${MASK_FREQ}, ${PROBE_LEVEL}, ${l}, ${bestVars}" >> all_vars_bark_cauchy.txt
      python3 plot_learning.py --logs_dir=logs/${MASK_FREQ}_${PROBE_LEVEL}_${l}
    done
  done
done
