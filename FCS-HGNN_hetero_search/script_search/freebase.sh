#!/bin/bash

export CUDA_VISIBLE_DEVICES="2"
EXEC="../src/main.py"
output_dir="../../../output_dir/V3/run"

data_name='Freebase'
epoch=200
search_homo=False

for batch_size in 16 32
do
    for patience in 5 10 15 20
    do
        for edge_feats in 8 16 32
        do
            for hidden_dim in 64 128 256 512
            do
                for residual in 'False'
                do
                    for use_batchnorm in 'True'
                    do
                        python $EXEC \
                        --epoch $epoch \
                        --data_name $data_name \
                        --patience $patience \
                        --batch_size $batch_size \
                        --output_dir $output_dir \
                        --search_homo $search_homo \
                        --edge_feats $edge_feats \
                        --hidden_dim $hidden_dim \
                        --residual $residual \
                        --use_batchnorm $use_batchnorm
                    done
                done
            done
        done
    done
done
