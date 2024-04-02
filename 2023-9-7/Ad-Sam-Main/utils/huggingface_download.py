from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
import cv2

pipe = pipeline(task=Tasks.text_to_image_synthesis, 
                model='camenduru/control_v11p_sd15_canny',
                use_safetensors=True)

