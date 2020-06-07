call conda activate torch37

python detect_rcnn.py --num_classes 3   --weights_path ../weights/20200606_1715_faster_rcnn_weights.pth.tar

python detect_yolo.py --model_name yolov3      --num_anchors 9 --num_classes 3 --weights_path ../weights/20200607_0918_yolov3_a9_c3_weights.pth.tar
python detect_yolo.py --model_name yolov3      --num_anchors 6 --num_classes 3 --weights_path ../weights/20200606_2252_yolov3_a6_c3_weights.pth.tar
python detect_yolo.py --model_name yolov3-tiny --num_anchors 6 --num_classes 3 --weights_path ../weights/20200606_2326_yolov3-tiny_a6_c3_weights.pth.tar
python detect_yolo.py --model_name yolov3-tiny --num_anchors 4 --num_classes 3 --weights_path ../weights/20200607_0953_yolov3-tiny_a4_c3_weights.pth.tar

python detect_retina.py --backbone resnet18   --num_classes 3 --weights_path ../weights/20200606_1759_retinanet_resnet18_weights.pth.tar   --score_thres 0.4 --apply_nms True
python detect_retina.py --backbone resnet18   --num_classes 3 --weights_path ../weights/20200606_1759_retinanet_resnet18_weights.pth.tar   --score_thres 0.4
python detect_retina.py --backbone resnet34   --num_classes 3 --weights_path ../weights/20200606_1827_retinanet_resnet34_weights.pth.tar   --score_thres 0.4 --apply_nms True
python detect_retina.py --backbone resnet34   --num_classes 3 --weights_path ../weights/20200606_1827_retinanet_resnet34_weights.pth.tar   --score_thres 0.4
python detect_retina.py --backbone resnet50   --num_classes 3 --weights_path ../weights/20200606_1901_retinanet_resnet50_weights.pth.tar   --score_thres 0.4 --apply_nms True
python detect_retina.py --backbone resnet50   --num_classes 3 --weights_path ../weights/20200606_1901_retinanet_resnet50_weights.pth.tar   --score_thres 0.4
python detect_retina.py --backbone resnet101  --num_classes 3 --weights_path ../weights/20200606_1939_retinanet_resnet101_weights.pth.tar  --score_thres 0.4 --apply_nms True
python detect_retina.py --backbone resnet101  --num_classes 3 --weights_path ../weights/20200606_1939_retinanet_resnet101_weights.pth.tar  --score_thres 0.4
python detect_retina.py --backbone resnext50  --num_classes 3 --weights_path ../weights/20200606_2044_retinanet_resnext50_weights.pth.tar  --score_thres 0.4 --apply_nms True
python detect_retina.py --backbone resnext50  --num_classes 3 --weights_path ../weights/20200606_2044_retinanet_resnext50_weights.pth.tar  --score_thres 0.4 
python detect_retina.py --backbone resnext101 --num_classes 3 --weights_path ../weights/20200606_2125_retinanet_resnext101_weights.pth.tar --score_thres 0.4 --apply_nms True
python detect_retina.py --backbone resnext101 --num_classes 3 --weights_path ../weights/20200606_2125_retinanet_resnext101_weights.pth.tar --score_thres 0.4

pause
