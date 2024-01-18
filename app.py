import json
import numpy as np
import PIL
import requests as req
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)
import torch


class InferlessPythonModel:
    def initialize(self):
        model_id = "timbrooks/instruct-pix2pix"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.pipe.to("cuda:0")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

    def infer(self, prompt, image_url):
        url = image_url
        image = PIL.Image.open(req.get(url, stream=True).raw)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        images = self.pipe(
            prompt,
            image=image,
            num_inference_steps=10,
            image_guidance_scale=1,
        ).images

        images_array = np.uint8(images[0])
        output_array = np.array(images_array, dtype=np.float32)
        return {"generated_image": output_array}

    def finalize(self):
        self.pipe = None
