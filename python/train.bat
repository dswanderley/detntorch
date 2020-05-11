call conda activate torch37

python train_yolo.py --model_name yolov3 --batch_size 6
python train_yolo.py --model_name yolov3_fol --batch_size 6
python train_yolo.py --model_name yolov3-tiny --batch_size 8
python train_yolo.py --model_name yolov3-tiny_fol --batch_size 8
python train_rcnn.py --batch_size 8
python train_retina.py --batch_size 8

pause
