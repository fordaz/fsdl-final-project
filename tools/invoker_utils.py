import base64
import os
import time
from io import BytesIO

from PIL import Image

from utilities.file_utils import save_json


def save_syn_image(body, image_fname):
    body_bytes = body.encode("ascii")
    im_bytes = base64.b64decode(body_bytes)
    im_file = BytesIO(im_bytes)
    img = Image.open(im_file)
    img.save(f"{image_fname}.png", "PNG")


def save_syn_annotations_kits(base_dir, syn_kits):
    ts = int(time.time())
    for syn_kit in syn_kits:
        output_prefix = f"{syn_kit['index']}_{ts}"

        annotations_fname = os.path.join(base_dir, f"{output_prefix}_annotations.json")
        save_json(syn_kit["annotations"], annotations_fname)
        print(f"Successfully saved file {annotations_fname}")

        image_fname = os.path.join(base_dir, f"{output_prefix}_image.json")
        save_syn_image(syn_kit["image"], image_fname)
        print(f"Successfully saved file {image_fname}")

        image_blocks_fname = os.path.join(base_dir, f"{output_prefix}_image_blocks.json")
        save_syn_image(syn_kit["image_blocks"], image_blocks_fname)
        print(f"Successfully saved file {image_blocks_fname}")
