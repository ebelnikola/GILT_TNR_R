#!/bin/bash
for chi in 11 16 21 26 31 36 41 46 51 56 61 66
do
    echo Producing free energy data with TRG for chi=${chi}
    python3 GiltTNR2D_Ising_benchmarks.py\
        -c "confs/free_energy_trg.yaml"\
        -y "cg_chis: !!python/object/apply:builtins.range [0, ${chi}, 1]"

    echo Producing free energy data with Gilt-TNR for chi=${chi}
    python3 GiltTNR2D_Ising_benchmarks.py\
        -c "confs/free_energy_gilt.yaml"\
        -y "cg_chis: !!python/object/apply:builtins.range [0, ${chi}, 1]"
done

