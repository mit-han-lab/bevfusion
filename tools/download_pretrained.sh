mkdir pretrained &&
cd pretrained &&
wget https://bevfusion.mit.edu/files/pretrained_updated/bevfusion-det.pth &&
wget https://bevfusion.mit.edu/files/pretrained_updated/bevfusion-seg.pth &&
wget https://bevfusion.mit.edu/files/pretrained/lidar-only-det.pth &&
wget https://bevfusion.mit.edu/files/pretrained/lidar-only-seg.pth &&
wget https://bevfusion.mit.edu/files/pretrained_updated/camera-only-det.pth &&
wget https://bevfusion.mit.edu/files/pretrained_updated/camera-only-seg.pth &&
wget https://bevfusion.mit.edu/files/pretrained_updated/swint-nuimages-pretrained.pth &&
cd .. &&
python tools/convert_checkpoints_to_torchsparse.py pretrained/bevfusion-det.pth pretrained/bevfusion-det.pth && 
python tools/convert_checkpoints_to_torchsparse.py pretrained/bevfusion-seg.pth pretrained/bevfusion-seg.pth && 
python tools/convert_checkpoints_to_torchsparse.py pretrained/lidar-only-det.pth pretrained/lidar-only-det.pth && 
python tools/convert_checkpoints_to_torchsparse.py pretrained/lidar-only-seg.pth pretrained/lidar-only-seg.pth

