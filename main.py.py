import PIL
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Define constants
MODEL_ID = "timbrooks/instruct-pix2pix"
IMAGE_PATH = "path/to/your/image.jpg"  # Replace with the actual path to your image
PROMPT = "turn him into an animated cartoon for an avatar"

# Load the image from a directory
def load_image(image_path):
    image = PIL.Image.open(image_path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

# Create the pipeline
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,  # Use torch.float16 for reduced memory usage
    safety_checker=None  # Set the safety checker to None if not needed
)
pipe.to("cuda")  # Move the pipeline to the GPU for faster processing
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Load the input image
image = load_image(IMAGE_PATH)

# Generate the cartoonized image
num_inference_steps = 10
image_guidance_scale = 1
images = pipe(PROMPT, image=image, num_inference_steps=num_inference_steps, image_guidance_scale=image_guidance_scale).images

# Display the cartoonized image
plt.imshow(images[0])
plt.axis('off')  # Turn off axis labels
plt.show()
