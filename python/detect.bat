call conda activate torch37

python detect_rcnn.py --weights_path ../weights/20200512_1958_faster_rcnn_weights.pth.tar
python detect_retina.py --weights_path ../weights/20200512_2035_retinanet_weights.pth.tar
python detect_yolo.py --model_name yolov3 --weights_path ../weights/20200512_2122_yolov3_weights.pth.tar
python detect_yolo.py --model_name yolov3_fol --weights_path ../weights/20200512_2152_yolov3_fol_weights.pth.tar
python detect_yolo.py --model_name yolov3-tiny --weights_path ../weights/20200512_2223_yolov3-tiny_weights.pth.tar
python detect_yolo.py --model_name yolov3-tiny_fol --weights_path ../weights/20200512_2236_yolov3-tiny_fol_weights.pth.tar

pause
