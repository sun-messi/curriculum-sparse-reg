#!/bin/bash
# CR Mode Training Script - Background execution with logging

# Configuration
CONFIG_PATH="configs/celeba32_cr.json"
NUM_GPUS=6
LOG_DIR="logs/cr_training"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_cr_${TIMESTAMP}.log"

# Create log directory
mkdir -p ${LOG_DIR}

# Print start info
echo "========================================" | tee -a ${LOG_FILE}
echo "CR Mode Training Started" | tee -a ${LOG_FILE}
echo "Time: ${TIMESTAMP}" | tee -a ${LOG_FILE}
echo "Config: ${CONFIG_PATH}" | tee -a ${LOG_FILE}
echo "GPUs: ${NUM_GPUS}" | tee -a ${LOG_FILE}
echo "Log: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}

# Run training in background
nohup python train_curriculum.py \
    --config-path ${CONFIG_PATH} \
    --num-gpus ${NUM_GPUS} \
    --distributed \
    --rigid-launch \
    >> ${LOG_FILE} 2>&1 &

# Save PID
PID=$!
echo ${PID} > ${LOG_DIR}/cr_training.pid
echo "Training started with PID: ${PID}" | tee -a ${LOG_FILE}
echo "To monitor: tail -f ${LOG_FILE}"
echo "To stop: kill ${PID}"
