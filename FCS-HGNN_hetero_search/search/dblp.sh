#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
EXEC="../src/main.py"
output_dir="../../../output_dir/V3/run"

data_name='DBLP'
epoch=200
search_homo=False

batch_size=32
patience=10
edge_feats=8
hidden_dim=64
residual='False'
use_batchnorm='True'

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
