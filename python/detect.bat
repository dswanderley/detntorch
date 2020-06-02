call conda activate torch37

python detect_rcnn.py   --weights_path ../weights/20200512_1958_faster_rcnn_weights.pth.tar
python detect_yolo.py   --model_name yolov3 --weights_path ../weights/20200512_2122_yolov3_weights.pth.tar
python detect_yolo.py   --model_name yolov3_fol --weights_path ../weights/20200512_2152_yolov3_fol_weights.pth.tar
python detect_yolo.py   --model_name yolov3-tiny --weights_path ../weights/20200512_2223_yolov3-tiny_weights.pth.tar
python detect_yolo.py   --model_name yolov3-tiny_fol --weights_path ../weights/20200512_2236_yolov3-tiny_fol_weights.pth.tar
python detect_retina.py --backbone resnet18 --weights_path ../weights/20200531_1631_retinanet_resnet18_weights.pth.tar      --score_thres 0.4 --apply_nms True
python detect_retina.py --backbone resnet18 --weights_path ../weights/20200531_1631_retinanet_resnet18_weights.pth.tar      --score_thres 0.4
python detect_retina.py --backbone resnet34 --weights_path ../weights/20200531_1659_retinanet_resnet34_weights.pth.tar      --score_thres 0.4 --apply_nms True
python detect_retina.py --backbone resnet34 --weights_path ../weights/20200531_1659_retinanet_resnet34_weights.pth.tar      --score_thres 0.4
python detect_retina.py --backbone resnet50 --weights_path ../weights/20200531_1728_retinanet_resnet50_weights.pth.tar      --score_thres 0.4 --apply_nms True
python detect_retina.py --backbone resnet50 --weights_path ../weights/20200531_1728_retinanet_resnet50_weights.pth.tar      --score_thres 0.4
python detect_retina.py --backbone resnext50 --weights_path ../weights/20200531_1852_retinanet_resnext50_weights.pth.tar --score_thres 0.4 --apply_nms True
python detect_retina.py --backbone resnext50 --weights_path ../weights/20200531_1852_retinanet_resnext50_weights.pth.tar --score_thres 0.4 
python detect_retina.py --backbone resnext101 --weights_path ../weights/20200531_1932_retinanet_resnext101_weights.pth.tar --score_thres 0.4 --apply_nms True
python detect_retina.py --backbone resnext101 --weights_path ../weights/20200531_1932_retinanet_resnext101_weights.pth.tar --score_thres 0.4

pause
