import pixellib
from pixellib.instance import instance_segmentation

segment_video = instance_segmentation()
segment_video.load_model("mask_rcnn_coco.h5")
segment_video.process_video("sample.mp4", show_bboxes = True, frames_per_second= 15, output_video_name="output_video.mp4")
