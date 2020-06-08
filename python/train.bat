call conda activate torch37

python train_rcnn.py                        --num_classes 3     --batch_size 8

python train_retina.py --backbone resnet18  --num_classes 3     --batch_size 4
python train_retina.py --backbone resnet34  --num_classes 3     --batch_size 4
python train_retina.py --backbone resnet50  --num_classes 3     --batch_size 4
python train_retina.py --backbone resnet101 --num_classes 3     --batch_size 4
python train_retina.py --backbone resnext50 --num_classes 3     --batch_size 4
python train_retina.py --backbone resnext101 --num_classes 3    --batch_size 2

python train_yolo.py --model_name yolov3      --num_anchors 9 --num_classes 3 --batch_size 6
python train_yolo.py --model_name yolov3      --num_anchors 6 --num_classes 3 --batch_size 6
python train_yolo.py --model_name yolov3-tiny --num_anchors 6 --num_classes 3 --batch_size 8
python train_yolo.py --model_name yolov3-tiny --num_anchors 4 --num_classes 3 --batch_size 8
python train_yolo.py --model_name yolov3-spp     --num_anchors 9 --num_classes 3 --batch_size 6
python train_yolo.py --model_name yolov3-spp     --num_anchors 6 --num_classes 3 --batch_size 6

pause

