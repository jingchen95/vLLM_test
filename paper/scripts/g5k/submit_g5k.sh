#!/bin/bash
# Build and submit one XPYD Grid'5000 job (run from the Lille frontend).
#
#   submit_g5k.sh MODE RUN_ID [WALLTIME] [extra oarsub args...]
#   e.g. submit_g5k.sh e0  e0_g5k_night1 3:30 -t night
#        submit_g5k.sh e0c e0c_g5k_1     1:30 -t day
#
# Parameters are baked into a per-run job file under ~/g5k/xpyd_pd/jobs so the
# exact submission is archived; OAR does not forward the submitter's
# environment to the job.
set -euo pipefail
MODE="${1:?mode e0|e0c}"; RUN_ID="${2:?run id}"; WALLTIME="${3:-3:30}"; shift 3 2>/dev/null || shift $#
REPO="${XPYD_G5K_REPO:-${HOME}/vLLM_test}"
P_CLUSTER="${XPYD_G5K_P_CLUSTER:-chicoree}"; D_CLUSTER="${XPYD_G5K_D_CLUSTER:-chuc}"
jobs_dir="${HOME}/g5k/xpyd_pd/jobs"; mkdir -p "${jobs_dir}" "${HOME}/g5k/xpyd_pd/results"
test ! -e "${REPO}/results/g5k/${RUN_ID}" || { echo "run id already used: ${RUN_ID}" >&2; exit 1; }
# Usage policy: weekday 09:00-19:00 usage per cluster is capped at
# 2 h x (cores of the cluster) per day, summed over ALL of the user's jobs
# (chicoree: 64 cores -> 128 core-h/day, i.e. one full node for 2 h). A
# 3.5 h chicoree+chuc daytime reservation alone was 175% (warning, 2026-10-20).
# XPYD jobs therefore go to night/weekend windows unless explicitly allowed.
case " $* " in
  *" -t night "*) ;;
  *) if [ "${XPYD_G5K_ALLOW_DAYTIME:-0}" != "1" ]; then
       echo "refusing a non-night submission: add '-t night', or set XPYD_G5K_ALLOW_DAYTIME=1 after checking" >&2
       echo "usagepolicycheck -t (daily allowance is per cluster and counts all your jobs)" >&2
       exit 1
     fi ;;
esac
job="${jobs_dir}/${RUN_ID}.oar"
exotic=""
case "${P_CLUSTER}${D_CLUSTER}" in *chicoree*) exotic="#OAR -t exotic";; esac
{
  echo "#!/bin/bash"
  echo "#OAR -n xpyd-${MODE}-${RUN_ID}"
  echo "#OAR -q default"
  [ -n "${exotic}" ] && echo "${exotic}"
  if [ "${P_CLUSTER}" = "${D_CLUSTER}" ]; then
    echo "#OAR -l {cluster='${P_CLUSTER}'}/host=2,walltime=${WALLTIME}:00"
  else
    echo "#OAR -l {cluster='${P_CLUSTER}'}/host=1+{cluster='${D_CLUSTER}'}/host=1,walltime=${WALLTIME}:00"
  fi
  echo "#OAR -O ${HOME}/g5k/xpyd_pd/results/OAR_%jobid%.out"
  echo "#OAR -E ${HOME}/g5k/xpyd_pd/results/OAR_%jobid%.err"
  echo "export XPYD_G5K_MODE=${MODE} XPYD_G5K_RUN_ID=${RUN_ID} XPYD_G5K_REPO=${REPO}"
  echo "export XPYD_G5K_P_CLUSTER=${P_CLUSTER} XPYD_G5K_D_CLUSTER=${D_CLUSTER}"
  [ -n "${XPYD_G5K_P_HIGH_MHZ:-}" ] && echo "export XPYD_G5K_P_HIGH_MHZ=${XPYD_G5K_P_HIGH_MHZ}"
  [ -n "${XPYD_G5K_D_HIGH_MHZ:-}" ] && echo "export XPYD_G5K_D_HIGH_MHZ=${XPYD_G5K_D_HIGH_MHZ}"
  echo "exec /bin/bash ${REPO}/paper/scripts/g5k/run_xpyd_g5k.sh"
} > "${job}"
chmod +x "${job}"
echo "job file: ${job}"; cat "${job}"
oarsub "$@" -S "${job}"
