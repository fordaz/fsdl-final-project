import pandas as pd

annotations_v1 = "./datasets/generated/full/annotations/full_annotations.csv"
annotations_v1_df = pd.read_csv(annotations_v1)
print(annotations_v1_df.head())

annotations_v2_df = pd.DataFrame()

annotations_v2_df['surname'] = annotations_v1_df['annotation']
annotations_v2_df['split'] = annotations_v1_df['split']
annotations_v2_df['nationality'] = "Spanish"

print(annotations_v2_df.head())

annotations_v2 = "/Users/fordaz/workspace/machine-learning/PyTorchNLPBook/chapters/chapter_7/7_3_surname_generation/data/surnames/surnames_with_splits_annon.csv"
annotations_v2_df.to_csv(annotations_v2, index=False, columns=["split", "nationality", "surname"])
