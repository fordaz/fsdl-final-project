import base64

from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

WIDTH, HIGHT = 800, 1000

box_colors = {
    "answer": (222, 118, 49, 250),
    "header": (159, 227, 157, 250),
    "other": (109, 193, 242, 250),
    "question": (49, 222, 222, 250)
}

def show_block_text(draw, block, font, boxed=False, color=(0,0,0)):
    text = block["text"].encode('utf-8')
    bbox = block['box']
    left, top = bbox[0], bbox[1]
    right, bottom = bbox[2], bbox[3]
    if boxed:
        box_color = box_colors.get(block["label"], (238, 240, 218, 250))
        draw.polygon([(left, top), (right, top), (right, bottom), (left, bottom)], box_color)

    draw.multiline_text((left, top), text, color, font=font, stroke_width=0)


def generate_img(syn_annotation_page, blocks_lookup, boxed_text=False):
    image = Image.new('RGBA', (WIDTH, HIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(image, mode='RGBA')  
    font = ImageFont.load_default()

    for block in syn_annotation_page["form"]:
        show_block_text(draw, block, font, boxed=boxed_text)

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('ascii')