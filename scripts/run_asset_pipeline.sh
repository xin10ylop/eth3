#!/bin/bash
# End-to-end pipeline for one asset, after the raw download (btcmodel.data) has finished.
#   bash scripts/run_asset_pipeline.sh [first_stage]          (ETH by default; BTC_ASSET=BTC for BTC)
# Stages: timeline dataset extras research frozen frozen_test wf wf_eval analysis package prod
# Every head trains in its own process (memory); stages log "STAGE <name> done" to $LOG.
set -euo pipefail
cd "$(dirname "$0")/.."
export BTC_ASSET=${BTC_ASSET:-ETH}
export BTC_ASSET
D=$(python -c "from btcmodel.config import DATA_DIR; print(DATA_DIR)")
A=$(python -c "from btcmodel.config import ART_DIR; print(ART_DIR)")
LOG=$D/pipeline.log
TAG=$(echo "$BTC_ASSET" | tr 'A-Z' 'a-z')
START=${1:-timeline}
STAGES="timeline dataset extras research frozen frozen_test wf wf_eval analysis package prod"
run=0
stage() { echo "STAGE $1 start $(date +%H:%M)" >> "$LOG"; }
done_() { echo "STAGE $1 done $(date +%H:%M)" >> "$LOG"; }
for S in $STAGES; do
  [ "$S" = "$START" ] && run=1
  [ $run = 1 ] || continue
  stage $S
  case $S in
    timeline) python -m btcmodel.timeline >> "$LOG" 2>&1 ;;
    dataset)  python -m btcmodel.dataset > "$D/dataset.log" 2>&1 ;;
    extras)   python scripts/build_targets_extra.py >> "$LOG" 2>&1
              python scripts/build_regime.py > "$D/regime_build.log" 2>&1 ;;
    research) for r in eda3_structure eda4_reference eda5_close_pressure eda6_btc_leads; do
                [ -f research/$r.py ] && PYTHONPATH=. python research/$r.py > "$A/research_$r.txt" 2>&1 || true
              done
              python scripts/plot_regime.py >> "$LOG" 2>&1 ;;
    frozen)   for h in h30 h60 close; do
                BTC_HEADS=h30,h60,close python scripts/train_eval.py valid $h > "$D/tv_$h.log" 2>&1
              done ;;
    frozen_test) for h in h30 h60 close; do
                BTC_HEADS=h30,h60,close python scripts/evaluate_saved.py test $h > "$D/te_frozen_$h.log" 2>&1
              done ;;
    wf)       for h in h10 h15 h30 h60 close; do
                BTC_REGIME=1 python scripts/train_production.py $h --end 2026-04-30 --tag _wf5 --resume >> "$D/wf5_train.log" 2>&1
              done ;;
    wf_eval)  for s in test valid; do for h in h10 h15 h30 h60 close; do
                BTC_REGIME=1 BTC_TAG=_wf5 python scripts/evaluate_saved.py $s $h > "$D/${s}_wf5_$h.log" 2>&1
              done; done ;;
    analysis) python scripts/compare_versions.py test "" _wf5 > /dev/null 2>> "$LOG"
              python scripts/robustness.py test _wf5 > /dev/null 2>> "$LOG"
              python scripts/robustness.py test "" > /dev/null 2>> "$LOG"
              python scripts/regime_eval.py test _wf5 > /dev/null 2>> "$LOG"
              python scripts/regime_eval.py test "" > /dev/null 2>> "$LOG"
              python scripts/summarize.py test _wf5 > /dev/null; python scripts/summarize.py test "" > /dev/null
              python scripts/summarize.py valid "" > /dev/null
              python scripts/make_figures.py test _wf5 >> "$LOG" 2>&1
              python scripts/plot_windows.py test _wf5 >> "$LOG" 2>&1
              python -m tests.test_predictor_consistency _wf5 test h10 h15 h30 h60 close > "$A/predictor_consistency.txt" 2>&1 ;;
    package)  python scripts/package_models.py valid_wf5 models/wf5_20260430 >> "$LOG" 2>&1 ;;
    prod)     for h in h10 h15 h30 h60 close; do
                BTC_REGIME=1 python scripts/train_production.py $h --old-thin 2 --resume >> "$D/prod_train.log" 2>&1
              done
              python scripts/package_models.py prod models/prod_20261102 >> "$LOG" 2>&1 ;;
  esac
  done_ $S
done
echo "PIPELINE COMPLETE $(date +%H:%M)" >> "$LOG"
