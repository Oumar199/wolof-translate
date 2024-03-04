# %%writefile wolof-translate/wolof_translate/models/transformers/main.py
from wolof_translate.models.transformers.position import PositionalEncoding
# from wolof_translate.models.transformers.size import SizePredict
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torch import nn
from typing import *
import torch
import copy
# new Exception for that transformer
class TargetException(Exception):
    
    def __init__(self, error):
        
        print(error)

class GenerationException(Exception):

    def __init__(self, error):

        print(error)

class Transformer(nn.Module):
    
    def __init__(self, 
                 vocab_size: int,
                 encoder,
                 decoder,
                 class_criterion = nn.CrossEntropyLoss(label_smoothing=0.1),
                #  size_criterion = nn.MSELoss(),
                #  n_features: int = 100,
                #  n_layers: int = 2,
                 n_poses_max: int = 2000,
                 projection_type: str = "embedding",
                 max_len: int = 20, 
                 share_weight: bool = False):
        
        super(Transformer, self).__init__()
        
        assert len(encoder.layers) > 0 and len(decoder.layers) > 0
    
        self.dropout = encoder.layers._modules['0'].dropout.p
        
        self.enc_embed_dim = encoder.layers._modules['0'].linear1.in_features
        
        self.dec_embed_dim = decoder.layers._modules['0'].linear1.in_features
        
        # we can initiate the positional encoding model
        self.pe = PositionalEncoding(n_poses_max, self.enc_embed_dim)
        
        if projection_type == "embedding":
            
            self.embedding_layer = nn.Embedding(vocab_size, self.enc_embed_dim)
        
        elif projection_type == "linear":
            
            self.embedding_layer = nn.Linear(vocab_size, self.enc_embed_dim)
        
        # initialize the first encoder and decoder
        self.encoder = encoder
        
        self.decoder = decoder
        
        self.class_criterion = class_criterion
        
        # add dropout to the inputs and outputs of the encoder and decoder
        self.in_dropout = nn.Dropout(p = self.dropout)

        self.out_dropout = nn.Dropout(p = self.dropout)

        # self.size_criterion = size_criterion
        
        # let's initiate the mlp for predicting the target size
        # self.size_prediction = SizePredict(
        #     self.enc_embed_dim,
        #     n_features=n_features,
        #     n_layers=n_layers,
        #     normalization=True, # we always use normalization
        #     drop_out=self.dropout
        #     )

        self.classifier = nn.Linear(self.dec_embed_dim, vocab_size)

        # let us share the weights between the embedding layer and classification
        # linear layer
        if share_weight:
          
          self.embedding_layer.register_forward_hook(self._copy_embedding_weights)


        self.max_len = max_len
        
        
    def forward(self, input_, input_mask = None, target = None, target_mask = None, 
                pad_token_id:int = 3):

        # ---> Encoder prediction
        input_embed = self.embedding_layer(input_)
        
        # recuperate the last input (before position)
        last_input = input_embed[:, -1:]
       
        # add position to input_embedding
        input_embed = self.pe(input_embed)
        
        # recuperate the input mask for pytorch encoder
        pad_mask1 = (input_mask == 0).to(next(self.parameters()).device, dtype = torch.bool) if not input_mask is None else None
        
        # let us compute the states
        input_embed = input_embed.type_as(next(self.encoder.parameters()))

        input_embed = self.in_dropout(input_embed) # apply dropout to the input embed
        
        states = self.encoder(input_embed, src_key_padding_mask = pad_mask1)

        # apply dropout to the states
        states = self.out_dropout(states)
   
        # ---> Decoder prediction
        # let's predict the size of the target 
        # target_size = self.size_prediction(states).mean(axis = 1)
        
        target_embed = self.embedding_layer(target)
        
        # recuperate target mask for pytorch decoder            
        pad_mask2 = (target_mask == 0).to(next(self.parameters()).device, dtype = torch.bool) if not target_mask is None else None
        
        # define the attention mask
        targ_mask = self.get_target_mask(target_embed.size(1))

        # let's concatenate the last input and the target shifted from one position to the right (new seq dim = target seq dim)
        target_embed = torch.cat((last_input, target_embed[:, :-1]), dim = 1)
        
        # add position to target embed
        target_embed = self.pe(target_embed)
        
        # we pass all of the shifted target sequence to the decoder if training mode
        if self.training:
            
            target_embed = target_embed.type_as(next(self.encoder.parameters()))

            # add dropout to the target
            target_embed = self.in_dropout(target_embed)
            
            outputs = self.decoder(target_embed, states, tgt_mask = targ_mask, tgt_key_padding_mask = pad_mask2)

            # add dropout to the outputs
            outputs = self.out_dropout(outputs)
            
        else: ## This part was understand with the help of the professor Bousso.
            
            # if we are in evaluation mode we will not use the target but the outputs to make prediction and it is
            # sequentially done (see comments)
            
            # let us recuperate the last input as the current outputs
            outputs = last_input.type_as(next(self.encoder.parameters()))
            
            # for each target that we want to predict
            for t in range(target.size(1)):
                
                # recuperate the target mask of the current decoder input
                current_targ_mask = targ_mask[:t+1, :t+1] # all attentions between the elements before the last target
                
                # we do the same for the padding mask
                current_pad_mask = None
                
                if not pad_mask2 is None:
                    
                    current_pad_mask = pad_mask2[:, :t+1]
                
                # make new predictions
                out = self.decoder(outputs, states, tgt_mask = current_targ_mask, tgt_key_padding_mask = current_pad_mask) 
                
                # add the last new prediction to the decoder inputs
                outputs = torch.cat((outputs, out[:, -1:]), dim = 1) # the prediction of the last output is the last to add (!)
            
            # let's take only the predictions (the last input will not be taken)
            outputs = outputs[:, 1:]
        
        # let us add padding index to the outputs
        if not target_mask is None: 
          target = copy.deepcopy(target.cpu())
          target = target.to(target_mask.device).masked_fill_(target_mask == 0, -100)

        # ---> Loss Calculation
        # let us calculate the loss of the size prediction
        # size_loss = 0
        # if not self.size_criterion is None:
            
            # size_loss = self.size_criterion(target_size, target_mask.sum(axis = -1).unsqueeze(1).type_as(next(self.parameters())))
            
        outputs = self.classifier(outputs)
        
        # let us permute the two last dimensions of the outputs
        outputs_ = outputs.permute(0, -1, -2)

        # calculate the loss
        loss = self.class_criterion(outputs_, target)

        outputs = torch.softmax(outputs, dim = -1)

        # calculate the predictionos
        outputs = copy.deepcopy(outputs.detach().cpu())
        predictions = torch.argmax(outputs, dim = -1).to(target_mask.device).masked_fill_(target_mask == 0, pad_token_id)

        return {'loss': loss, 'preds': predictions}
    
    def generate(self, input_, input_mask = None, temperature: float = 0, max_len: Union[int, None] = None):

        if self.training:

          raise GenerationException("You cannot generate when the model is on training mode!")

        # recuperate the max len
        max_len = max_len if not max_len is None else self.max_len
        
        # ---> Encoder prediction
        input_embed = self.embedding_layer(input_)
        
        # recuperate the last input (before position)
        last_input = input_embed[:, -1:]
       
        # add position to input_embedding
        input_embed = self.pe(input_embed)
        
        # recuperate the input mask for pytorch encoder
        pad_mask1 = (input_mask == False).to(next(self.parameters()).device) if not input_mask is None else None
        
        # let us compute the states
        input_embed = input_embed.type_as(next(self.encoder.parameters()))
        
        states = self.encoder(input_embed, src_key_padding_mask = pad_mask1)

        # ---> Decoder prediction
        # let us recuperate the maximum length
        # max_len = self.max_len if not self.max_len is None else 0

        # let's predict the size of the target and the target mask
        # if max_len > 0:

          # target_size = self.size_prediction(states).mean(axis = 1).round().clip(1, max_len)
        
        # else:

          # target_size = torch.max(self.size_prediction(states).mean(axis = 1).round(), torch.tensor(1.0))

        # target_ = copy.deepcopy(target_size.cpu())

        # target_mask = [torch.tensor(int(size[0])*[1] + [0] * max(max_len - int(size[0]), 0)) for size in target_.tolist()]

        # if max_len > 0:

          # target_mask = torch.stack(target_mask).to(next(self.parameters()).device, dtype = torch.bool)

        # else:

          # target_mask = pad_sequence(target_, batch_first = True).to(next(self.parameters()).device, dtype = torch.bool)
      
        # recuperate target mask for pytorch decoder            
        # pad_mask2 = (target_mask == 0).to(next(self.parameters()).device, dtype = torch.bool) if not target_mask is None else None
        
        # define the attention mask
        targ_mask = self.get_target_mask(max_len)
            
        # if we are in evaluation mode we will not use the target but the outputs to make prediction and it is
        # sequentially done (see comments)
        
        # let us recuperate the last input as the current outputs
        outputs = last_input.type_as(next(self.encoder.parameters()))
        
        # for each target that we want to predict
        for t in range(max_len):
            
            # recuperate the target mask of the current decoder input
            current_targ_mask = targ_mask[:t+1, :t+1] # all attentions between the elements before the last target
            
            # we do the same for the padding mask
            current_pad_mask = None
            
            # if not pad_mask2 is None:
                
                # current_pad_mask = pad_mask2[:, :t+1]
            
            # make new predictions
            out = self.decoder(outputs, states, tgt_mask = current_targ_mask, tgt_key_padding_mask = current_pad_mask) 
            
            # add the last new prediction to the decoder inputs
            outputs = torch.cat((outputs, out[:, -1:]), dim = 1) # the prediction of the last output is the last to add (!)
        
        # let's take only the predictions (the last input will not be taken)
        outputs = outputs[:, 1:]

        # ---> Predictions
        outputs = self.classifier(outputs)

        # calculate the resulted outputs with temperature
        if temperature > 0:

          outputs = torch.softmax(outputs / temperature, dim = -1)
        
        else:

          outputs = torch.softmax(outputs, dim = -1)

        # calculate the predictionos
        outputs = copy.deepcopy(outputs.detach().cpu())
        predictions = torch.argmax(outputs, dim = -1).to(next(self.parameters()).device)
       
        return predictions
    
    def generate_(self, input_, input_mask = None, temperature: float = 0, max_len: Union[int, None] = None):

        if self.training:

          raise GenerationException("You cannot generate when the model is on training mode!")

        # recuperate the max len
        max_len = max_len if not max_len is None else self.max_len
        
        # ---> Encoder prediction
        input_embed = self.embedding_layer(input_)
        
        # recuperate the last input (before position)
        last_input = input_[:, -1:]
       
        # add position to input_embedding
        input_embed = self.pe(input_embed)
        
        # recuperate the input mask for pytorch encoder
        pad_mask1 = (input_mask == False).to(next(self.parameters()).device) if not input_mask is None else None
        
        # let us compute the states
        input_embed = input_embed.type_as(next(self.encoder.parameters()))
        
        states = self.encoder(input_embed, src_key_padding_mask = pad_mask1)

        # define the attention mask
        targ_mask = self.get_target_mask(max_len)
        
        # if we are in evaluation mode we will not use the target but the outputs to make prediction and it is
        # sequentially done (see comments)
        
        # let us recuperate the last input as the current outputs
        tokens = last_input
        
        # for each target that we want to predict
        for t in range(max_len):
            
            # recuperate the target mask of the current decoder input
            current_targ_mask = targ_mask[:t+1, :t+1] # all attentions between the elements before the last target
            
            # we do the same for the padding mask
            current_pad_mask = None
            
            # if not pad_mask2 is None:
                
                # current_pad_mask = pad_mask2[:, :t+1]
            
            # pass the tokens to the embedding layer to get the embeddings
            tokens_embed = self.pe(self.embedding_layer(tokens)).type_as(next(self.encoder.parameters()))
            
            # make new predictions
            out = self.decoder(tokens_embed, states, tgt_mask = current_targ_mask, tgt_key_padding_mask = current_pad_mask) 
            
            # recuperate probabilities with or without temperature
            if temperature > 0:
                
              probs = torch.softmax(self.classifier(out[:, -1]) / temperature, dim = -1) 
            
            else:
                  
              probs = torch.softmax(self.classifier(out[:, -1]), dim = -1)
              
            # let us sample the next token
            next_token = torch.multinomial(probs, num_samples = 1)
            
            # add the last new prediction to the decoder inputs
            tokens = torch.cat((tokens, next_token), dim = -1) # the prediction of the last output is the last to add (!)
            
        # let's take only the predictions (the last input will not be taken)
        predictions = tokens[:, 1:]

        return predictions
    
    def beam_generate(self, input_, input_mask = None, temperature: float = 0, max_len: Union[int, None] = None, beam_size: int = 5):

        # let us initialize the batch size
        batch_size = input_.size(0)
        
        if self.training:

          raise GenerationException("You cannot generate when the model is on training mode!")

        # recuperate the max len
        max_len = max_len if not max_len is None else self.max_len
        
        # ---> Encoder prediction
        input_embed = self.embedding_layer(input_)
        
        # recuperate the last input (before position)
        last_input = input_[:, -1:]
       
        # add position to input_embedding
        input_embed = self.pe(input_embed)
        
        # recuperate the input mask for pytorch encoder
        pad_mask1 = (input_mask == False).to(next(self.parameters()).device) if not input_mask is None else None
        
        # let us compute the states
        input_embed = input_embed.type_as(next(self.encoder.parameters()))
        
        states = self.encoder(input_embed, src_key_padding_mask = pad_mask1)

        # define the attention mask
        targ_mask = self.get_target_mask(max_len)
        
        # if we are in evaluation mode we will not use the target but the outputs to make prediction and it is
        # sequentially done (see comments)
        
        # let us recuperate the last input as the current outputs
        tokens = last_input
        
        # generate predictions (beam search with the help of chatgpt)
        
        # let us initialize the beams
        beams = [tokens[i, -1:].expand(beam_size, -1) for i in range(batch_size)]
        
        # initialize the beam scores
        scores = torch.zeros((batch_size, beam_size), device=next(self.parameters()).device)
        
        # for each target that we want to predict
        for t in range(max_len):
            
          # initialize all of the candidates and the scores
          all_candidates = []
          all_scores = []
          
          # recuperate the target mask of the current decoder input
          current_targ_mask = targ_mask[:t+1, :t+1] # all attentions between the elements before the last target
          
          # we do the same for the padding mask
          current_pad_mask = None
          
          # iterate over the beams and batches to calculate make predictions 
          for be_idx in range(beam_size):
            
            # initialize the candidates and scores
            candidates = []
            candidate_scores = []
            
            for ba_idx in range(batch_size):
              
              # recuperate the current state
              current_state = states[ba_idx].unsqueeze(0)
              
              # recuperate the current sequence
              tokens = beams[ba_idx][be_idx].unsqueeze(0)
              
              # pass the tokens to the embedding layer to get the embeddings
              tokens_embed = self.pe(self.embedding_layer(tokens)).type_as(next(self.encoder.parameters()))
            
              # make new predictions
              out = self.decoder(tokens_embed, current_state, tgt_mask = current_targ_mask, tgt_key_padding_mask = current_pad_mask) 
              
              # recuperate probabilities with or without temperature
              if temperature > 0:
                  
                log_probs = F.log_softmax(self.classifier(out[:, -1]).squeeze() / temperature, dim = -1) 
              
              else:
                    
                log_probs = F.log_softmax(self.classifier(out[:, -1]).squeeze(), dim = -1)
              
              # get top k candidates
              beam_scores, beam_candidates = log_probs.topk(beam_size, dim = -1)
              
              # add the candidates to the set of candidates (do the same for the scores)
              candidates.append(beam_candidates)
              candidate_scores.append(beam_scores)
            
            # add the current set of candidates and scores to the global set
            all_candidates.append(torch.stack(candidates))
            all_scores.append(torch.stack(candidate_scores))
            
          # select top k candidates and scores from all beams
          all_candidates = torch.stack(all_candidates)
          all_scores = torch.stack(all_scores)
          topk_scores, topk_idx = all_scores.view(batch_size, -1).topk(beam_size)
          
          # Update beams and scores for the current iteration
          new_beams = []
          new_scores = []
          
          # iterate over the batches to update the beams and scores
          for ba_idx in range(batch_size):
            
            # recuperate candidates
            beam_candidates = all_candidates[:, ba_idx].reshape(-1)
            
            # recuperate indices
            selected_indices = topk_idx[ba_idx]
            
            # recuperate the beams
            selected_beams = selected_indices // beam_size
            
            # recuperate the tokens
            selected_tokens = beam_candidates[selected_indices % beam_size]
            
            new_beams.append([torch.concatenate((beams[ba_idx][selected_beams[i]], selected_tokens[i].unsqueeze(0)), dim = -1) for i in range(beam_size)])
            
            new_scores.append(topk_scores[ba_idx])
          
          # update the beams and scores
          beams = new_beams
          scores = new_scores
          
        # recuperate the top candidates for each sequence in the batch
        predictions = torch.stack([beams[i][0].squeeze() for i in range(batch_size)])
                                
        # let's take only the predictions (the last input will not be taken)
        predictions = predictions[:, 1:]

        return predictions
      
    def diverse_beam_generate(self, input_, input_mask = None, temperature: float = 0,
                              max_len: Union[int, None] = None, beam_size: int = 5,
                              beam_groups: int = 1, diversity_penalty: float = 0.5):

        # let us initialize the batch size
        batch_size = input_.size(0)
        
        if self.training:

          raise GenerationException("You cannot generate when the model is on training mode!")

        # recuperate the max len
        max_len = max_len if not max_len is None else self.max_len
        
        # ---> Encoder prediction
        input_embed = self.embedding_layer(input_)
        
        # recuperate the last input (before position)
        last_input = input_[:, -1:]
       
        # add position to input_embedding
        input_embed = self.pe(input_embed)
        
        # recuperate the input mask for pytorch encoder
        pad_mask1 = (input_mask == False).to(next(self.parameters()).device) if not input_mask is None else None
        
        # let us compute the states
        input_embed = input_embed.type_as(next(self.encoder.parameters()))
        
        states = self.encoder(input_embed, src_key_padding_mask = pad_mask1)

        # define the attention mask
        targ_mask = self.get_target_mask(max_len)
        
        # if we are in evaluation mode we will not use the target but the outputs to make prediction and it is
        # sequentially done (see comments)
        
        # let us recuperate the last input as the current outputs
        tokens = last_input
        
        # generate predictions (beam search with the help of chatgpt)
        
        # let us initialize the beams
        beams = [tokens[i, -1:].expand(beam_size, -1) for i in range(batch_size)]
        
        # initialize the beam scores
        scores = torch.zeros((batch_size, beam_size), device=next(self.parameters()).device)
        
        # for each target that we want to predict
        for t in range(max_len):
            
          # initialize all of the candidates and the scores
          all_candidates = []
          all_scores = []
          
          # recuperate the target mask of the current decoder input
          current_targ_mask = targ_mask[:t+1, :t+1] # all attentions between the elements before the last target
          
          # we do the same for the padding mask
          current_pad_mask = None
          
          # iterate over the beams and batches to calculate make predictions 
          for be_idx in range(beam_size):
            
            # initialize the candidates and scores
            candidates = []
            candidate_scores = []
            
            for ba_idx in range(batch_size):
              
              # recuperate the current state
              current_state = states[ba_idx].unsqueeze(0)
              
              # recuperate the current sequence
              tokens = beams[ba_idx][be_idx].unsqueeze(0)
              
              # pass the tokens to the embedding layer to get the embeddings
              tokens_embed = self.pe(self.embedding_layer(tokens)).type_as(next(self.encoder.parameters()))
            
              # make new predictions
              out = self.decoder(tokens_embed, current_state, tgt_mask = current_targ_mask, tgt_key_padding_mask = current_pad_mask) 
              
              # recuperate probabilities with or without temperature
              if temperature > 0:
                  
                log_probs = F.log_softmax(self.classifier(out[:, -1]).squeeze() / temperature, dim = -1) 
              
              else:
                    
                log_probs = F.log_softmax(self.classifier(out[:, -1]).squeeze(), dim = -1)
              
              # get top k candidates
              beam_scores, beam_candidates = log_probs.topk(beam_size, dim = -1)
              
              # add the candidates to the set of candidates (do the same for the scores)
              candidates.append(beam_candidates)
              candidate_scores.append(beam_scores)
            
            # add the current set of candidates and scores to the global set
            all_candidates.append(torch.stack(candidates))
            all_scores.append(torch.stack(candidate_scores))
            
          # select top k candidates and scores from all beams
          all_candidates = torch.stack(all_candidates)
          all_scores = torch.stack(all_scores)
          
          # reshape candidates and scores for efficient matrix operations
          all_candidates_flat = all_candidates.reshape(beam_size, -1)
          all_scores_flat = all_scores.reshape(beam_size, -1)
          
          # apply the diversity penalty to the scores for each beam group
          group_size = beam_size // beam_groups
          for group_idx in range(beam_groups):
            
            group_start = group_idx * group_size
            
            group_end = (group_idx + 1) * group_size
            
            group_candidates = all_candidates_flat[group_start:group_end]
            
            group_scores = all_scores_flat[group_start:group_end]
                    
            diversity_penalty_ = self.hamming_distance(group_candidates.unsqueeze(2), group_candidates.unsqueeze(1))
            
            penalty = diversity_penalty * diversity_penalty_.view(group_size, -1).sum(dim=0)
            
            group_scores -= penalty
          
          # reshape the scores back to the original shape
          all_scores = all_scores_flat.reshape(beam_size, batch_size, -1)
          
          topk_scores, topk_idx = all_scores.view(batch_size, -1).topk(beam_size)
          
          # Update beams and scores for the current iteration
          new_beams = []
          new_scores = []
          
          # iterate over the batches to update the beams and scores
          for ba_idx in range(batch_size):
            
            # recuperate candidates
            beam_candidates = all_candidates[:, ba_idx].reshape(-1)
            
            # recuperate indices
            selected_indices = topk_idx[ba_idx]
            
            # recuperate the beams
            selected_beams = selected_indices // beam_size
            
            # recuperate the tokens
            selected_tokens = beam_candidates[selected_indices % beam_size]
            
            new_beams.append([torch.concatenate((beams[ba_idx][selected_beams[i]], selected_tokens[i].unsqueeze(0)), dim = -1) for i in range(beam_size)])
            
            new_scores.append(topk_scores[ba_idx])
          
          # update the beams and scores
          beams = new_beams
          scores = new_scores
          
        # recuperate the top candidates for each sequence in the batch
        predictions = torch.stack([beams[i][0].squeeze() for i in range(batch_size)])
                                
        # let's take only the predictions (the last input will not be taken)
        predictions = predictions[:, 1:]

        return predictions
    
    def hamming_distance(self, sequence_1, sequence_2):
      
      # Calculate the hamming distance between two sequences
      return (sequence_1 != sequence_2).sum(axis = -1)
    
    def get_target_mask(self, attention_size: int):
        
        return torch.triu(torch.ones((attention_size, attention_size)), diagonal = 1).to(next(self.parameters()).device, dtype = torch.bool)


    def _copy_embedding_weights(self, module, input, output):
        # Copy the embedding weights to the last dense layer
        self.classifier.weight.data = module.weight.data
# %%
