python render.py -m ../Output/Mip-NeRF360/bicycle -s ../Dataset/Mip-NeRF360/bicycle
python render.py -m ../Output/Mip-NeRF360/bonsai -s ../Dataset/Mip-NeRF360/bonsai
python render.py -m ../Output/Mip-NeRF360/counter -s ../Dataset/Mip-NeRF360/counter
python render.py -m ../Output/Mip-NeRF360/flowers -s ../Dataset/Mip-NeRF360/flowers
python render.py -m ../Output/Mip-NeRF360/garden -s ../Dataset/Mip-NeRF360/garden
python render.py -m ../Output/Mip-NeRF360/kitchen -s ../Dataset/Mip-NeRF360/kitchen
python render.py -m ../Output/Mip-NeRF360/room -s ../Dataset/Mip-NeRF360/room
python render.py -m ../Output/Mip-NeRF360/stump -s ../Dataset/Mip-NeRF360/stump
python render.py -m ../Output/Mip-NeRF360/treehill -s ../Dataset/Mip-NeRF360/treehill

python render.py -m ../Output/tanksandtemples/train -s ../Dataset/tanksandtemples/train
python render.py -m ../Output/tanksandtemples/truck -s ../Dataset/tanksandtemples/truck

./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m ../Output/Mip-NeRF360/bicycle

python train.py -m /raid/duzhenhua/3Dv_NVS/3DGS/Output/Mip-NeRF360/bicycle -s /raid/duzhenhua/3Dv_NVS/3DGS/Dataset/Mip-NeRF360/bicycle