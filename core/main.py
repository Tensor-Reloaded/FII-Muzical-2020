import torch
import numpy as np
from torch.nn import functional as F
from transformer import Transformer
from preprocessing import PreprocessMidi
from postprocessing import PostprocessMidi

data = PreprocessMidi()
convert_midi = PostprocessMidi()
vocab = data.get_vocabulary('data/vocabulary.json')
vocab_size = len(vocab)

batch_size = 4
maxlen = 300
num_layers = 2
dimension_model = 128
num_heads = 2
feed_forward_dim = 256
output_size = 64

X, Y = data.get_dataset(maxlen, 1)

print("Vocab size:", vocab_size)
print("Number of files", len(X))

input_notes = "Piano-right_C4_1.25 nextOffset Piano-right_C4_1.25"
input_notes_tokens = data.vectorize(notes = input_notes)

def loss_function(real, pred):
    return F.cross_entropy(pred.view(-1, pred.size(-1)), real.view(-1), reduction="mean")

def sample(logits):
    value, indices = torch.topk(logits, 10)
    indices = np.asarray(indices).astype("int32")
    preds = torch.softmax(value, dim = 0)
    preds = preds.detach().numpy()
    return np.random.choice(indices, p=preds)

def model_train(epochs):
    model = Transformer(num_layers, dimension_model, num_heads, feed_forward_dim,
                        vocab_size, vocab_size, maxlen)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0003, betas=(0.9, 0.98), eps=1e-9)
    output_midi_count = 0
    for epoch in range(epochs):
        model.train()
        for batch in range(0, len(X), batch_size):
            batch_X = torch.from_numpy(X[batch:batch + batch_size])
            batch_Y = torch.from_numpy(Y[batch:batch + batch_size])

            pred = model.compute(batch_X)

            optimizer.zero_grad()
            loss = loss_function(batch_Y, pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if epoch % 10 == 0:
                print("Epoch: ", epoch, "Loss: ",loss.item())

        if epoch % 25 == 0:
            model.eval()
            start_tokens = [_ for _ in input_notes_tokens]
            count = 0
            generated = []
            while count <= output_size:
                pad_len = maxlen - len(start_tokens)
                sample_index = len(start_tokens) - 1

                if pad_len < 0:
                    x = start_tokens[:maxlen]
                    sample_index = maxlen - 1
                elif pad_len > 0:
                    x = start_tokens + [0] * pad_len
                else:
                    x = start_tokens

                x = np.array([x])
                input_gen = torch.from_numpy(x).contiguous().to(non_blocking=True)

                pred = model.compute(input_gen)

                s = sample(pred[0][sample_index])
                generated.append(s)
                start_tokens.append(s)
                count = len(generated)
            print(input_notes_tokens, generated)
            notes = ""
            for el in input_notes_tokens:
                notes += str(vocab[el]) + " "
            for el in generated:
                notes += str(vocab[el]) + " "
            # print(notes)
            convert_midi.compute_song(notes, "midi_ver_"+str(output_midi_count))
            output_midi_count +=1
            


if __name__ == "__main__":
    model_train(400)
