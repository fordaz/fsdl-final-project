import base64

from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

WIDTH, HIGHT = 800, 1000


def show_block_text(draw, block, font, boxed=False, color=(0,0,0)):
    text = block["text"].encode('utf-8')
    bbox = block['box']
    left, top = bbox[0], bbox[1]
    right, bottom = bbox[2], bbox[3]
    if boxed:
        x_len, y_len = right - left, bottom - top
        draw_rectange(draw, left, top, x_len, y_len)
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


def draw_line(draw, x, y, len, horizontal=True, regular=True, fill=255):
    if horizontal:
        if regular:
            draw.line([(x, y), (x+len,y)], fill)
        else:
            for x in range(x, x+len, 4):
                draw.line([(x, y), (x+2,y)], fill)
    else:
        if regular:
            draw.line([(x, y), (x,y+len)], fill)
        else:
            for y in range(y, y+len, 4):
                draw.line([(x, y), (x,y+2)], fill)

def draw_rectange(draw, x, y, x_len, y_len, regular=True, fill=255):
    draw_line(draw, x, y, x_len, horizontal=True, regular=regular, fill=fill)
    draw_line(draw, x, y+y_len, x_len, horizontal=True, regular=regular, fill=fill)
    draw_line(draw, x, y, y_len, horizontal=False, regular=regular, fill=fill)
    draw_line(draw, x+x_len, y, y_len, horizontal=False, regular=regular, fill=fill)
