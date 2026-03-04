#!/bin/bash
# Download dovekie triplet chains + sn.desdovekie singles from CSD3

REMOTE_NEW="login-q-2:/rds/project/rds-63QXlf5HuFo/dlo26/new_grid/ns"
REMOTE_OLD="login-q-2:/rds/project/rds-63QXlf5HuFo/dlo26/grid/ns"
LOCAL="/data/will/new_grid/ns"

MODELS="lcdm wlcdm walcdm mlcdm klcdm rlcdm Alcdm nrunlcdm"
TRIPLETS="bao.desi_2024_bao_all+planck_2018_plik+sn.desdovekie bao.desi_dr2+planck_2018_CamSpec+sn.desdovekie"

# Only sync _polychord_raw, skip .resume and clusters (large, not needed)
RSYNC_OPTS="-az --progress --exclude=*.resume --exclude=clusters/ --exclude=*.out --exclude=*.err --exclude=machine.file.* --exclude=*.yaml --exclude=*.logZ --exclude=*.1.txt --exclude=*_dead.txt --exclude=*_equal_weights.txt --exclude=*_phys_live.txt --exclude=*_phys_live-birth.txt --exclude=*_prior.txt"

# Triplets (new_grid only)
for model in $MODELS; do
    for ds in $TRIPLETS; do
        echo "--- $model / $ds ---"
        mkdir -p "$LOCAL/$model/$ds"
        rsync $RSYNC_OPTS "$REMOTE_NEW/$model/$ds/" "$LOCAL/$model/$ds/"
    done
done

# Constituent singles: try both grids (old grid has runs missing from new_grid)
SINGLES="bao.desi_2024_bao_all bao.desi_dr2 planck_2018_plik planck_2018_CamSpec sn.desdovekie"
for model in $MODELS; do
    for ds in $SINGLES; do
        echo "--- $model / $ds ---"
        mkdir -p "$LOCAL/$model/$ds"
        rsync $RSYNC_OPTS "$REMOTE_NEW/$model/$ds/" "$LOCAL/$model/$ds/" 2>/dev/null
        rsync $RSYNC_OPTS "$REMOTE_OLD/$model/$ds/" "$LOCAL/$model/$ds/" 2>/dev/null
    done
done

echo "ALL COMPLETE"
