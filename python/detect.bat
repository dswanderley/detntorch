call conda activate torch37

python detect_yolo.py --model_name yolov3 --weights_path ../weights/20200419_1419_yolov3_weights.pth.tar
python detect_yolo.py --model_name yolov3-tiny --weights_path ../weights/20200419_1509_yolov3-tiny_weights.pth.tar
python detect_yolo.py --model_name yolov3-tiny_fol --weights_path ../weights/20200419_1522_yolov3-tiny_fol_weights.pth.tar
python detect_yolo.py --model_name yolov3-tiny --weights_path ../weights/20200419_1636_yolov3-tiny_weights.pth.tar
python detect_yolo.py --model_name yolov3-tiny_fol --weights_path ../weights/20200419_1648_yolov3-tiny_fol_weights.pth.tar
python detect_yolo.py --model_name yolov3_fol --weights_path ../weights/20200419_1701_yolov3_fol_weights.pth.tar
python detect_rcnn.py --weights_path ../weights/20200419_1604_faster_rcnn_weights.pth.tar
pause
