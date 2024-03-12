#!/bin/bash
for T_relative in 0.999 0.99999 1 1.00001 1.001
do
    beta=`bc<<<"scale=16; 1.0/(${T_relative}*2.269185314213022)"`
    echo Producing spectra data with TRG for T_relative=${T_relative}
    python3 GiltTNR2D_Ising_benchmarks.py\
        -c "confs/spectra_trg.yaml"\
        -y "beta: !!float ${beta}"
    echo Producing spectra data with Gilt-TNR for T_relative=${T_relative}
    python3 GiltTNR2D_Ising_benchmarks.py\
        -c "confs/spectra_gilt.yaml"\
        -y "beta: !!float ${beta}"
done
