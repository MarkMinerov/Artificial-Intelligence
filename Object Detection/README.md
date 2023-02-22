# Object Detection using Deep Learning Neural Networks

- `Region Proposal Network`
- `ROI Pooling` (Region of Interest Pooling)

- "One Stage" category: `SSD` (Single Shot multibox detector), `YOLO` (You only look once)
- "Two Stage" category: `RCNN`, `Faster RCNN`

## Faster RNN model

![Faster_rnn](faster-RCNN.png)

## SSD. Single Shot multibox detector model

![SSD](./SSD.png)

SSD uses `Non-maximum suppression` operation. This operation uses `Intersection of a Union`, what is `Intersection` / `Union`
If `IoU` gives us number `> 0.5` than we delete one `RoI` with less confidence.

## YOLO. You only look once

![yolo](YOLO.png)

YOLOv4 consist of:

- Backbone: CSPDDarknet53
- Neck: SPP, PAN
- Head: YOLOv3

## Choosing the right neural network for your object detection task

Things to consider:

1. Which is more important for you? Speed or accuracy?
2. How will you deploy your model? On the cloud? On the edge?

## Accuracy VS time

![accuracy_vs_time](accuracy_vs_time.png)

`mAP` - mean average precision
