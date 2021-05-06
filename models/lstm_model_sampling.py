import torch

"""
This is an adaptation of the source code of the book (Chapter 7): 
Natural Language Processing with PyTorch, by Delip Rao and Brian McMahan
"""

def sample_from_model(model, vectorizer, device, num_samples=1, sample_size=20, temperature=1.0):
    vocab = vectorizer.get_vocabulary()
    begin_seq_index = [vocab.begin_seq_index 
                       for _ in range(num_samples)]
    begin_seq_index = torch.tensor(begin_seq_index, 
                                   dtype=torch.int64).unsqueeze(dim=1)
    indices = [begin_seq_index.to(device)]
    h_t = None
    for time_step in range(sample_size):
        x_t = indices[time_step]
        probability_vector = model.sample(x_t.to(device), h_t, temperature)
        picked_indices = torch.multinomial(probability_vector, num_samples=1)
        indices.append(picked_indices.to(device))
    indices = torch.stack(indices).squeeze().permute(1, 0)
    return indices


def decode_samples(sampled_indices, vectorizer):
    decoded_annotations = []
    vocab = vectorizer.get_vocabulary()
    
    for sample_index in range(sampled_indices.shape[0]):
        generated_annotation = ""
        for time_step in range(sampled_indices.shape[1]):
            sample_item = sampled_indices[sample_index, time_step].item()
            if sample_item == vocab.begin_seq_index or sample_item == vocab.unk_index:
                continue
            elif sample_item == vocab.end_seq_index:
                break
            else:
                generated_annotation += vocab.lookup_index(sample_item)
        decoded_annotations.append(generated_annotation)
    return decoded_annotations


def sample_model(model, vectorizer, device, num_samples=2):
    samples = sample_from_model(model, vectorizer, device, num_samples=num_samples)
    return decode_samples(samples, vectorizer)
