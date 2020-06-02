call conda activate torch37

python train_rcnn.py --batch_size 8
python train_retina.py --backbone resnet18      --batch_size 4
python train_retina.py --backbone resnet34      --batch_size 4
python train_retina.py --backbone resnet50      --batch_size 4
python train_retina.py --backbone resnet101     --batch_size 4
python train_retina.py --backbone resnext50     --batch_size 4
python train_retina.py --backbone resnext101    --batch_size 2
python train_yolo.py --model_name yolov3        --batch_size 6
python train_yolo.py --model_name yolov3_fol    --batch_size 6
python train_yolo.py --model_name yolov3-tiny   --batch_size 8
python train_yolo.py --model_name yolov3-tiny_fol --batch_size 8

pause

