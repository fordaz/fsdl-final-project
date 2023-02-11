import random
import json

from utilities.img_utils import *
from models.gru_model_sampling import (sample_from_gru_model, decode_samples)

def append_valid_annotations(sampled_annotations, valid_annotations):
    num_valid_annot = 0
    for sampled_annotation in sampled_annotations:
        try:
            sampled_annotation = json.loads(sampled_annotation)
            has_key_properties = all(map(lambda property: property in sampled_annotation, ['box', 'text', 'id', 'label']))
            full_bbox = len(sampled_annotation['box']) == 4
            if has_key_properties and full_bbox:
                valid_annotations.append(sampled_annotation)
                num_valid_annot += 1
        except Exception:
            pass
    return num_valid_annot


def generate_syn_page(model, vectorizer, device, min_num_annot, max_num_annot, max_annot_length, temperature):
    valid_annotations = []
    batch_size = 32
    num_annot = random.randint(min_num_annot, max_num_annot+1)
    max_attemps, attempts = 100, 0
    total_annot, num_valid_annot = 0, 0
    while len(valid_annotations) < num_annot and attempts < max_attemps:
        sampled_token_idxs = sample_from_gru_model(model, vectorizer, device, batch_size, max_annot_length, temperature)
        sampled_annotations = decode_samples(sampled_token_idxs, vectorizer)
        total_annot += len(sampled_annotations)
        num_valid_annot += append_valid_annotations(sampled_annotations, valid_annotations)
        attempts += 1
    valid_annotations = valid_annotations[:min(len(valid_annotations), num_annot)]
    return valid_annotations, total_annot, num_valid_annot


def generate_syn_kit(idx, syn_annotation_page, blocks_lookup):
    syn_kit = {'index': idx, 'annotations': None, 'image': None, 'image_blocks': None}

    syn_kit['annotations'] = syn_annotation_page

    syn_kit['image_blocks'] = generate_img(syn_annotation_page, blocks_lookup, boxed_text=True)

    syn_kit['image'] = generate_img(syn_annotation_page, blocks_lookup, boxed_text=False)

    return syn_kit


def create_boxes_lookup(blocks):
    block_dict = {}
    for block in blocks["form"]:
        block_dict[block["id"]] = block
    return block_dict


def generate_synthetic_images(syn_annotation_pages):
    syn_kits = []
    for idx, syn_annotation_page in enumerate(syn_annotation_pages):
        blocks_lookup = create_boxes_lookup(syn_annotation_page)

        syn_kit = generate_syn_kit(idx, syn_annotation_page, blocks_lookup)

        syn_kits.append(syn_kit)
    return syn_kits
