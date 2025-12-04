#!/home/williamfbx/15888-project/handover-sim/venv38/bin/python

set -x
set -e

DIR="$( cd "$( dirname -- "$0" )" && pwd )"
GADDPG_DIR="$( cd "$DIR/../.." && pwd )"
export PYTHONUNBUFFERED="True"
LOG_NAME="agent"

MODEL_NAME=${1-"dummy"}
RUN_NUM=${2-1}
EPI_NUM=${3-25}
EPOCH=${4-latest}

# Change to GA-DDPG directory
cd "$GADDPG_DIR"

LOG="output/${MODEL_NAME}/test_cracker_box_log.txt.`date +'%Y-%m-%d_%H-%M-%S'`"

# Create output directory first
mkdir -p "output/${MODEL_NAME}"

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# Build command with optional perturbation parameters
CMD="/home/williamfbx/15888-project/handover-sim/venv38/bin/python -m core.train_test_offline --pretrained output/${MODEL_NAME} --test --log --record --test_episode_num ${EPI_NUM} --num_runs ${RUN_NUM} --model_surfix ${EPOCH}"

# Add perturbation parameters
if [ ! -z "${PERTURB_TRANSLATION_STD}" ]; then
    CMD="${CMD} --perturb_translation_std ${PERTURB_TRANSLATION_STD}"
fi
if [ ! -z "${PERTURB_ROTATION_STD}" ]; then
    CMD="${CMD} --perturb_rotation_std ${PERTURB_ROTATION_STD}"
fi
if [ ! -z "${PERTURB_DURATION}" ]; then
    CMD="${CMD} --perturb_duration ${PERTURB_DURATION}"
fi

CUDA_VISIBLE_DEVICES=0 ${CMD}
