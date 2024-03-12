#!/bin/bash
echo Producing scalding dimension data with TRG.
python3 GiltTNR2D_Ising_benchmarks.py\
    -c "confs/scaldims_trg.yaml"
echo Producing scalding dimension data with Gilt-TNR.
python3 GiltTNR2D_Ising_benchmarks.py\
    -c "confs/scaldims_gilt.yaml"
