# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Own idea: a bigger mixer (S4: d 128, 6 layers, 8 heads) on the frozen table, cheap now that the first layer is cached.
train $CODE2 s4_rot $CACHE_FLAGS $ROT --d_model 128 --e_layers 6 --n_heads 8 && score $CODE2 s4_rot
echo "exit $? at $(stamp)"
