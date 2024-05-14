import torch
from transformers import CLIPTokenizer
import os
import json
import logging

from comfy import sd1_clip

class SDClipModel(sd1_clip.SDClipModel):
    def __init__(self, *args, **kwargs):
        # Call the initializer of the parent class
        super().__init__(*args, **kwargs)
    
    def forward(self, tokens):
        # Capture hidden states at each layer
        backup_embeds = self.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
        tokens = torch.LongTensor(tokens).to(device)

        attention_mask = None
        if self.enable_attention_masks:
            attention_mask = torch.zeros_like(tokens)
            max_token = self.transformer.get_input_embeddings().weight.shape[0] - 1
            for x in range(attention_mask.shape[0]):
                for y in range(attention_mask.shape[1]):
                    attention_mask[x, y] = 1
                    if tokens[x, y] == max_token:
                        break

        # Capture hidden states at each layer
        all_hidden_states = []
        for layer_idx in range(self.num_layers):
            outputs = self.transformer(tokens, attention_mask, intermediate_output=layer_idx, final_layer_norm_intermediate=self.layer_norm_hidden_state)
            all_hidden_states.append(outputs[1])  # Assuming outputs[1] is the hidden state

        self.transformer.set_input_embeddings(backup_embeds)

        if self.layer == "last":
            z = outputs[0]
        else:
            z = outputs[1]

        pooled_output = None
        if len(outputs) >= 3:
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()

        return all_hidden_states, z.float(), pooled_output
    
    def set_up_textual_embeddings(self, tokens, current_embeds):
        out_tokens = []
        next_new_token = token_dict_size = current_embeds.weight.shape[0] - 1
        embedding_weights = []

        for x in tokens:
            tokens_temp = []
            for y in x:
                if isinstance(y, int):
                    if y == token_dict_size:  # EOS token
                        y = -1
                    tokens_temp.append(y)
                else:
                    if isinstance(y, torch.Tensor) and len(y.shape) > 0 and y.shape[0] == current_embeds.weight.shape[1]:
                        embedding_weights.append(y)
                        tokens_temp.append(next_new_token)
                        next_new_token += 1
                    else:
                        logging.warning("WARNING: shape mismatch when trying to apply embedding, embedding will be ignored {} != {}".format(
                            y.shape[0] if len(y.shape) > 0 else 'None', current_embeds.weight.shape[1]))
            while len(tokens_temp) < len(x):
                tokens_temp.append(self.special_tokens["pad"])
            out_tokens.append(tokens_temp)

        n = token_dict_size
        if len(embedding_weights) > 0:
            new_embedding = torch.nn.Embedding(next_new_token + 1, current_embeds.weight.shape[1], device=current_embeds.weight.device, dtype=current_embeds.weight.dtype)
            new_embedding.weight[:token_dict_size] = current_embeds.weight[:-1]
            for x in embedding_weights:
                new_embedding.weight[n] = x
                n += 1
            new_embedding.weight[n] = current_embeds.weight[-1]  # EOS embedding
            self.transformer.set_input_embeddings(new_embedding)

        processed_tokens = []
        for x in out_tokens:
            processed_tokens.append(list(map(lambda a: n if a == -1 else a, x)))  # The EOS token should always be the largest one

        return processed_tokens
    
    def decode_hidden_states(self, hidden_states):
        # Assuming hidden_states is a tensor of shape [num_layers, seq_length, hidden_dim]
        # We will decode only the last layer's hidden states for simplicity
        last_layer_hidden_states = hidden_states[-1]
        
        # Get the logits from the hidden states
        logits = self.transformer.get_output_embeddings()(last_layer_hidden_states)

        # Get the most likely tokens from the logits
        decoded_token_ids = torch.argmax(logits, dim=-1)

        # Convert token ids to tokens
        decoded_tokens = [self.tokenizer.convert_ids_to_tokens(token_id) for token_id in decoded_token_ids[0]]

        return decoded_tokens

    def get_decoded_text(self, text):
        # Tokenize the input text
        tokenized_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        tokens = tokenized_input["input_ids"].tolist()[0]

        # Get hidden states
        tokens_with_weights = self.set_up_textual_embeddings([tokens], self.transformer.get_input_embeddings())
        tokens_with_weights = torch.LongTensor(tokens_with_weights).to(self.transformer.get_input_embeddings().weight.device)

        # Pad or truncate the tokens to match the expected length
        max_length = self.max_length
        if len(tokens_with_weights[0]) > max_length:
            tokens_with_weights = tokens_with_weights[:, :max_length]
        elif len(tokens_with_weights[0]) < max_length:
            padding = torch.full((1, max_length - len(tokens_with_weights[0])), self.special_tokens["pad"]).to(tokens_with_weights.device)
            tokens_with_weights = torch.cat((tokens_with_weights, padding), dim=1)

        all_hidden_states, _, _ = self.forward(tokens_with_weights)
        
        # Decode the hidden states to text
        decoded_tokens = self.decode_hidden_states(all_hidden_states)
        
        # Join tokens to form the decoded text
        decoded_text = self.tokenizer.convert_tokens_to_string(decoded_tokens)

        return decoded_text

    def get_hidden_states(self, text):
        # Tokenize the input text
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        tokenized_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        tokens = tokenized_input["input_ids"].tolist()
        print(tokens)
        tokens = tokens[0]

        tokens_with_weights = self.set_up_textual_embeddings([tokens], self.transformer.get_input_embeddings())
        tokens_with_weights = torch.LongTensor(tokens_with_weights).to(self.transformer.get_input_embeddings().weight.device)
        
        # Pad or truncate the tokens to match the expected length
        max_length = self.max_length
        if len(tokens_with_weights[0]) > max_length:
            tokens_with_weights = tokens_with_weights[:, :max_length]
        elif len(tokens_with_weights[0]) < max_length:
            padding = torch.full((1, max_length - len(tokens_with_weights[0])), self.special_tokens["pad"]).to(tokens_with_weights.device)
            tokens_with_weights = torch.cat((tokens_with_weights, padding), dim=1)

        # Get hidden states for each layer
        all_hidden_states, _, _ = self.forward(tokens_with_weights)

        # Assuming you have the tokenizer's vocabulary mapping
        word_mappings = [tokenizer.convert_ids_to_tokens([token])[0] for token in tokens]
        
        # Print hidden states for each token at each layer
        hidden_states_output = []
        for layer_idx, hidden_states in enumerate(all_hidden_states):
            hidden_states_output.append(f"\nLayer {layer_idx} hidden states:")
            for token_idx, token in enumerate(tokens):
                word = word_mappings[token_idx]
                hidden_state = hidden_states[0][token_idx].cpu().detach().numpy()  # Assuming batch size 1
                hidden_states_output.append(f"Word: {word}, Token idx: {token_idx}, {token}")

        return "\n".join(hidden_states_output)
