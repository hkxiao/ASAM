from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
import cv2

pipe = pipeline(task=Tasks.text_to_image_synthesis, 
                model='AI-ModelScope/stable-diffusion-xl-base-1.0',
                use_safetensors=True,
                model_revision='v1.0.0')

prompt = 'a dog'
output = pipe({'text': prompt})
cv2.imwrite('result.png', output['output_imgs'][0])