
from wolof_translate.models.transformers.position import PositionalEncoding
from wolof_translate.models.transformers.size import SizePredict
from torch.nn.utils.rnn import pad_sequence
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
                 size_criterion = nn.MSELoss(),
                 n_features: int = 100,
                 n_layers: int = 2,
                 n_poses_max: int = 500,
                 projection_type: str = "embedding",
                 max_len: Union[int, None] = None, 
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
        
        self.size_criterion = size_criterion
        
        # let's initiate the mlp for predicting the target size
        self.size_prediction = SizePredict(
            self.enc_embed_dim,
            n_features=n_features,
            n_layers=n_layers,
            normalization=True, # we always use normalization
            drop_out=self.dropout
            )

        self.classifier = nn.Linear(self.dec_embed_dim, vocab_size)

        # let us share the weights between the embedding layer and classification
        # linear layer
        if share_weight:
          
          self.classifier.weight.data = self.embedding_layer.weight.data

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
        
        states = self.encoder(input_embed, src_key_padding_mask = pad_mask1)
   
        # ---> Decoder prediction
        # let's predict the size of the target 
        target_size = self.size_prediction(states).mean(axis = 1)
        
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
            
            outputs = self.decoder(target_embed, states, tgt_mask = targ_mask, tgt_key_padding_mask = pad_mask2)
            
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
        size_loss = 0
        if not self.size_criterion is None:
            
            size_loss = self.size_criterion(target_size, target_mask.sum(axis = -1).unsqueeze(1).type_as(next(self.parameters())))
            
        outputs = self.classifier(outputs)
        
        # let us permute the two last dimensions of the outputs
        outputs_ = outputs.permute(0, -1, -2)

        # calculate the loss
        loss = self.class_criterion(outputs_, target)

        outputs = torch.softmax(outputs, dim = -1)

        # calculate the predictionos
        outputs = copy.deepcopy(outputs.detach().cpu())
        predictions = torch.argmax(outputs, dim = -1).to(target_mask.device).masked_fill_(target_mask == 0, pad_token_id)

        return {'loss': loss + size_loss, 'preds': predictions}
    
    def generate(self, input_, input_mask = None, temperature: float = 0, pad_token_id:int = 3):

        if self.training:

          raise GenerationException("You cannot generate when the model is on training mode!")

        # ---> Encoder prediction
        input_embed = self.embedding_layer(input_)
        
        # recuperate the last input (before position)
        last_input = input_embed[:, -1:]
       
        # add position to input_embedding
        input_embed = self.pe(input_embed)
        
        # recuperate the input mask for pytorch encoder
        pad_mask1 = (input_mask == 0).bool().to(next(self.parameters()).device) if not input_mask is None else None
        
        # let us compute the states
        input_embed = input_embed.type_as(next(self.encoder.parameters()))
        
        states = self.encoder(input_embed, src_key_padding_mask = pad_mask1)

        # ---> Decoder prediction
        # let us recuperate the maximum length
        max_len = self.max_len if not self.max_len is None else 0

        # let's predict the size of the target and the target mask
        if max_len > 0:

          target_size = self.size_prediction(states).mean(axis = 1).round().clip(1, max_len)
        
        else:

          target_size = torch.max(self.size_prediction(states).mean(axis = 1).round(), torch.tensor(1.0))

        target_ = copy.deepcopy(target_size.cpu())

        target_mask = [torch.tensor(int(size[0])*[1] + [0] * max(max_len - int(size[0]), 0)) for size in target_.tolist()]

        if max_len > 0:

          target_mask = torch.stack(target_mask).to(next(self.parameters()).device, dtype = torch.bool)

        else:

          target_mask = pad_sequence(target_, batch_first = True).to(next(self.parameters()).device, dtype = torch.bool)
      
        # recuperate target mask for pytorch decoder            
        pad_mask2 = (target_mask == 0).to(next(self.parameters()).device, dtype = torch.bool) if not target_mask is None else None
        
        # define the attention mask
        targ_mask = self.get_target_mask(target_mask.size(1))
            
        # if we are in evaluation mode we will not use the target but the outputs to make prediction and it is
        # sequentially done (see comments)
        
        # let us recuperate the last input as the current outputs
        outputs = last_input.type_as(next(self.encoder.parameters()))
        
        # for each target that we want to predict
        for t in range(target_mask.size(1)):
            
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

        # ---> Predictions
        outputs = self.classifier(outputs)

        # calculate the resulted outputs with temperature
        if temperature > 0:

          outputs = torch.softmax(outputs / temperature, dim = -1)
        
        else:

          outputs = torch.softmax(outputs, dim = -1)

        # calculate the predictionos
        outputs = copy.deepcopy(outputs.detach().cpu())
        predictions = torch.argmax(outputs, dim = -1).to(target_mask.device).masked_fill_(target_mask == 0, pad_token_id)

        return predictions
    

    def get_target_mask(self, attention_size: int):
        
        return torch.triu(torch.ones((attention_size, attention_size)), diagonal = 1).to(next(self.parameters()).device, dtype = torch.bool)
