import torch
import mlflow

import traceback

from models.annotations_dataset import AnnotationsDataset
from serving.wrapper_utils import *

class GRUAnnotationsWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        if torch.cuda.is_available():
            self.gru_model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"])
        else:
            kwargs = {"map_location":torch.device('cpu')}
            self.gru_model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"], **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gru_model.eval()
        self.dataset = AnnotationsDataset.load(context.artifacts["dataset_input_file"])
        self.vectorizer = self.dataset.get_vectorizer()
        self.vocab = self.vectorizer.get_vocabulary()
        self.total_annots = 0
        self.total_valid_annots = 0

    def predict(self, context, model_input):
        print(f"Getting model inputs {model_input} type {type(model_input)}")
        model, vectorizer, vocab = self.gru_model, self.vectorizer, self.vocab

        row = model_input.iloc[0]

        num_pages = int(row.num_pages)
        min_num_annot, max_num_annot = int(row.min_num_annot), int(row.max_num_annot)
        max_annot_length = int(row.max_annot_length)
        temperature = row.temperature

        try:
            syn_annotation_pages = []
            for page_number in range(num_pages):
                raw_syn_annot, total_annot, num_valid_annot = generate_syn_page(
                                                                model, vectorizer, self.device, 
                                                                min_num_annot, max_num_annot, 
                                                                max_annot_length, temperature)
                self.total_annots += total_annot
                self.total_valid_annots += num_valid_annot
                syn_annotation_page = {"form": []}
                syn_annotation_page["form"].extend(raw_syn_annot)

                syn_annotation_pages.append(syn_annotation_page)

            syn_kits = generate_synthetic_images(syn_annotation_pages)

            print(f"Valid annotations ratio {float(self.total_valid_annots)/float(self.total_annots)}")
            return syn_kits
        except Exception as e:
            traceback.print_exc()
            return {"error": "Unable to generate synthetics annotations", "reason": str(e)}