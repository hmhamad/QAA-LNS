
#!/bin/bash

# Example Run Command: bash run.sh --config cifar100.conf --cc 75
# To save logs: bash run.sh --config cifar100.conf --cc 75 --save_logs
# To continue from a checkpoint: bash run.sh --config cifar100.conf --cc 75 --continue_from [checkpoint_directory]
# To continue from a checkpoint and change device: bash run.sh --config cifar100.conf --cc 75 --continue_from [checkpoint_directory] --new_device [device_number]

# Parsing the input arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
      --config)
      configfile="$2"
      shift # past argument
      shift # past value
      ;;
      --cc)
      arch="$2"
      shift # past argument
      shift # past value
      ;;
      --new_device)
      new_device="$2"
      shift # past argument
      shift # past value
      ;;
      --continue_from)
      continue_from="$2"
      shift # past argument
      shift # past value
      ;;
      --save_logs)
      save_logs=true
      shift # past argument
      ;;
      *)    # unknown option
      shift # past argument
      ;;
  esac
done

# Load the config file
continue_from="${continue_from:-n}"
configfile="${configfile:-$continue_from}"
if [[ $continue_from == "n" ]]
then
    source configs/$configfile
else
    source logs/$continue_from/config.conf
fi

# Overwrite the device if a new device is specified, else use the device from the config file
device="${new_device:-$device}"


# Name of the header file
HEADER_FILE="src/c++/constants.h"

# Create or overwrite the header file
echo "#ifndef CONSTANTS_H" > $HEADER_FILE
echo "#define CONSTANTS_H" >> $HEADER_FILE
echo "// Generated constants header file" >> $HEADER_FILE
echo "#define TOTAL_BITS $TOTAL_BITS" >> $HEADER_FILE
echo "#define FRAC_BITS $FRAC_BITS" >> $HEADER_FILE
echo "#define DPLUS_LUT_SIZE $DPLUS_LUT_SIZE" >> $HEADER_FILE
echo "#define DMINUS_LUT_SIZE $DMINUS_LUT_SIZE" >> $HEADER_FILE
echo "#define DPLUS_DMAX $DPLUS_DMAX" >> $HEADER_FILE
echo "#define DMINUS_DMAX $DMINUS_DMAX" >> $HEADER_FILE
echo "#define LIN_SIZE $LIN_SIZE" >> $HEADER_FILE
echo "#define EXP_LUT_SIZE $EXP_LUT_SIZE" >> $HEADER_FILE
echo "#define QUANTIZAION_AWARE_APPROX $QUANTIZAION_AWARE_APPROX" >> $HEADER_FILE
echo "#define X_UB $X_UB" >> $HEADER_FILE
echo "#define SAMPLE_MAGNITUDE $SAMPLE_MAGNITUDE" >> $HEADER_FILE
echo "#endif" >> $HEADER_FILE

# Compile the CUDA code
nvcc -Isrc/c++2a --cubin -arch sm_${arch} src/c++/lns_gpu.cu -o src/c++/lns_lib.cubin

if [[ $save_logs == "true" ]]
then
    CUDA_VISIBLE_DEVICS="" python3 main.py --config ${configfile} --sim_name ${sim_name}  --lns_bits ${TOTAL_BITS} --frac_bits ${FRAC_BITS} --lin_size ${LIN_SIZE} --x_ub ${X_UB} --sample_magnitude $SAMPLE_MAGNITUDE --save_logs --arithmetic $arithmetic --device ${device} --seed ${seed} --dataset ${dataset} --network ${network} --epochs ${epochs} --batchsize=${batchsize} --optim ${optim} --lr ${lr} --mom ${mom} --T_0 ${T_0} --T_mult ${T_mult} --eta_min ${eta_min} --sum_approx ${sum_approx} --sum_technique ${sum_technique} --loss_scale ${loss_scale}  --weight_decay ${weight_decay} --leaky_flag ${leaky_flag} --leaky_coeff ${leaky_coeff} --continue_from ${continue_from}
else
    CUDA_VISIBLE_DEVICS="" python3 main.py --config ${configfile} --sim_name ${sim_name}  --lns_bits ${TOTAL_BITS} --frac_bits ${FRAC_BITS} --lin_size ${LIN_SIZE} --x_ub ${X_UB} --sample_magnitude $SAMPLE_MAGNITUDE --arithmetic $arithmetic --device ${device} --seed ${seed} --dataset ${dataset} --network ${network} --epochs ${epochs} --batchsize=${batchsize} --optim ${optim} --lr ${lr} --mom ${mom} --T_0 ${T_0} --T_mult ${T_mult} --eta_min ${eta_min}  --sum_approx ${sum_approx} --sum_technique ${sum_technique} --loss_scale ${loss_scale}  --weight_decay ${weight_decay} --leaky_flag ${leaky_flag} --leaky_coeff ${leaky_coeff} --continue_from ${continue_from}
fi

