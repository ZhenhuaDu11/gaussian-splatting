# Dataset
# ../../Data/
# deepblending/drjohnson, playroom
# tanksandtemples/train, truck
# Mip-NeRF360/bicycle, bonsai, counter, flowers, garden, kitchen, room, stump, treehill

# Output
# ../exps/

# Train
python train.py -s <Dataset> -m <Output> 

# Eval
python train.py -s <Dataset> -m <Output> --eval
python render.py -s <Dataset> -m <Output>
python metrics.py -s <Dataset> -m <Output>

# SIBR Viewer
./SIBR_viewers/install/bin/SIBR_remoteGaussian_app -m <Output>
./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m <Output>