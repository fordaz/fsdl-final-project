import torch
import torch.nn.functional as F

import string
import random

"""
This is an adaptation of the source code of the book (Chapter 7): 
Natural Language Processing with PyTorch, by Delip Rao and Brian McMahan
"""
"""
    begin_seq_index = [vectorizer.char_vocab.begin_seq_index 
                       for _ in range(num_samples)]
    begin_seq_index = torch.tensor(begin_seq_index, 
                                   dtype=torch.int64).unsqueeze(dim=1)
    indices = [begin_seq_index]
    h_t = None
    
    for time_step in range(sample_size):
        x_t = indices[time_step]
        x_emb_t = model.char_emb(x_t)
        rnn_out_t, h_t = model.rnn(x_emb_t, h_t)
        prediction_vector = model.fc(rnn_out_t.squeeze(dim=1))
        probability_vector = F.softmax(prediction_vector / temperature, dim=1)
        indices.append(torch.multinomial(probability_vector, num_samples=1))
    indices = torch.stack(indices).squeeze().permute(1, 0)
    return indices
"""

def sample_from_model(model, vectorizer, device, num_samples=1, sample_size=20, temperature=1.0):
    vocab = vectorizer.get_vocabulary()
    begin_seq_index = [vocab.begin_seq_index for _ in range(num_samples)]
    # print(f"1. begin_seq_index {begin_seq_index}")
    begin_seq_index = torch.tensor(begin_seq_index, 
                                   dtype=torch.int64).unsqueeze(dim=1)
    # print(f"2. begin_seq_index {begin_seq_index}")
    indices = [begin_seq_index.to(device)]
    # print(f"3. indices {indices}")
    h_t = None
    for time_step in range(sample_size):
        x_t = indices[time_step]
        print(f"4. indicesx_t {x_t}")
        probability_vector, h_t = model.sample(x_t.to(device), h_t, temperature)
        picked_indices = torch.multinomial(probability_vector, num_samples=1)
        indices.append(picked_indices.to(device))
    indices = torch.stack(indices).squeeze().permute(1, 0)
    return indices


def sample_from_gru_model(model, vectorizer, device, num_samples=3, sample_size=200, temperature=1.0):
    vocab = vectorizer.get_vocabulary()
    begin_seq_index = [vocab.begin_seq_index for _ in range(num_samples)]
    begin_seq_index = torch.tensor(begin_seq_index, 
                                   dtype=torch.int64).unsqueeze(dim=1)
    indices = [begin_seq_index.to(device)]
    h_t = None
    for time_step in range(sample_size):
        x_t = indices[time_step]
        x_emb_t = model.char_emb(x_t)
        rnn_out_t, h_t = model.rnn(x_emb_t, h_t)
        prediction_vector = model.fc(rnn_out_t.squeeze(dim=1))
        probability_vector = F.softmax(prediction_vector / temperature, dim=1)
        indices.append(torch.multinomial(probability_vector, num_samples=1))
    indices = torch.stack(indices)
    print(f"1. Indices shape {indices.shape}")
    indices = indices.squeeze()
    print(f"2. Indices shape {indices.shape}")
    indices = indices.permute(1, 0)
    print(f"3. Indices shape {indices.shape}")
    # indices = torch.stack(indices).squeeze().permute(1, 0)
    return indices


def assume_prompt(model, vectorizer, device, prompt_str, num_samples):
    hidden = model.init_zero_state(device, num_samples)
    cell = model.init_zero_state(device, num_samples)
    vocab = vectorizer.get_vocabulary()

    for prompt_token in range(len(prompt_str)):
        prompt_idx = vocab.lookup_token(prompt_token)
        prompt_idxs = [prompt_idx for _ in range(num_samples)]
        prompt_input = torch.tensor(prompt_idxs, dtype=torch.int64).unsqueeze(dim=1)
        print(f"Input Hidden and cell {prompt_input.shape} {hidden.shape} {cell.shape} -> {prompt_token}")
        _, (hidden, cell) = model(prompt_input.to(device), (hidden.to(device), cell.to(device)))

    return hidden, cell


def sample_from_model_with_prompt(model, vectorizer, device, prompt_str, num_samples, sample_size, temperature=1.0):
    h_t = assume_prompt(model, vectorizer, device, prompt_str, num_samples)
    vocab = vectorizer.get_vocabulary()
    begin_seq_index = [vocab.lookup_token(string.printable[random.randint(10, 61)]) for _ in range(num_samples)]
    print(f"1. begin_seq_index {begin_seq_index}")
    begin_seq_index = torch.tensor(begin_seq_index, 
                                   dtype=torch.int64).unsqueeze(dim=1)
    print(f"2. begin_seq_index {begin_seq_index}")
    indices = [begin_seq_index.to(device)]
    print(f"3. indices {indices}")
    h_t = None
    for time_step in range(sample_size):
        x_t = indices[time_step]
        print(f"4. indices x_t {x_t} {h_t[0].shape} {h_t[1].shape}")
        probability_vector, _ = model.sample(x_t.to(device), h_t, temperature)
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
    samples = sample_from_model(model, vectorizer, device, num_samples=num_samples, sample_size=500, temperature=0.8)
    return decode_samples(samples, vectorizer)


def sample_with_prompt(model, vectorizer, device, prompt_str, predict_len, temperature=0.8):
    ## based on https://github.com/spro/practical-pytorch/
    ## blob/master/char-rnn-generation/char-rnn-generation.ipynb

    hidden = model.init_zero_state(device, 32)
    cell = model.init_zero_state(device, 32)
    prompt_input = vectorizer.vectorize_input(prompt_str).unsqueeze(dim=1)
    predicted = prompt_str

    # Use priming string to "build up" hidden state
    for p in range(len(prompt_str) - 1):
        _, (hidden, cell) = model(prompt_input[p].to(device), (hidden.to(device), cell.to(device)))
    inp = prompt_input[-1]

    vocab = vectorizer.get_vocabulary()

    for p in range(predict_len):
        output, (hidden, cell) = model([inp.to(device)], (hidden.to(device), cell.to(device)))

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        predicted_idx = torch.multinomial(output_dist, 1)[0]

        if predicted_idx == vocab.begin_seq_index or predicted_idx == vocab.unk_index:
            continue
        elif predicted_idx == vocab.end_seq_index:
            break
        else:
            predicted_char += vocab.lookup_index(predicted_idx)

        predicted += predicted_char
        inp = torch.tensor([predicted_idx]).long()

    return predicted
