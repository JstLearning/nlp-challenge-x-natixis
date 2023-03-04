import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel


torch.set_default_dtype(torch.float32)

nontextual_cols = ['Index - 9',
 'Index - 8',
 'Index - 7',
 'Index - 6',
 'Index - 5',
 'Index - 4',
 'Index - 3',
 'Index - 2',
 'Index - 1',
 'Index - 0',
 'Index Name_CVIX Index',
 'Index Name_EURUSD Curncy',
 'Index Name_EURUSDV1M Curncy',
 'Index Name_MOVE Index',
 'Index Name_SPX Index',
 'Index Name_SRVIX Index',
 'Index Name_SX5E Index',
 'Index Name_V2X Index',
 'Index Name_VIX Index']

class AttentionWithContext(nn.Module):
    """
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    """
    
    def __init__(self, input_shape, return_coefficients=False, bias=True):
        super(AttentionWithContext, self).__init__()
        self.return_coefficients = return_coefficients

        self.W = nn.Linear(input_shape, input_shape, bias=bias)
        self.tanh = nn.Tanh()
        self.u = nn.Linear(input_shape, 1, bias=False)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.W.weight.data.uniform_(-initrange, initrange)
        self.W.bias.data.uniform_(-initrange, initrange)
        self.u.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz):
        # do not pass the mask to the next layers
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
    
    def forward(self, x, mask=None):
        # x has shape: (samples, steps, features)
        # mask has size (samples, steps, 1)
        
        uit = self.W(x) # fill the gap # compute uit = W . x  where x represents ht
        # uit is then of size (samples, steps, features) (Linear only modifies the last dimension)
        uit = self.tanh(uit)

        ait = self.u(uit)
        # ait is of size (samples, steps, 1)
        a = torch.exp(ait)
        
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            #TODO: Treat case if all masks are False.
            # Not impossible if all inputs are invalid (example: blank inputs)
            a = a*mask.double()
        
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        eps = 1e-9
        a = a / (torch.sum(a, axis=1, keepdim=True) + eps)
        weighted_input = torch.sum(a * x, axis=1) ### fill the gap ### # compute the attentional vector
        if self.return_coefficients:
            return weighted_input, a ### [attentional vector, coefficients] ### use torch.sum to compute s
        else:
            return weighted_input ### attentional vector only ###
        
    

class DocumentEncoder(nn.Module):
    def __init__(self, return_coefficients=False, bias=True, dropout=0):
        super(DocumentEncoder, self).__init__()
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bigru = nn.GRU(input_size=768,
                          hidden_size=32,
                          num_layers=1,
                          bias=bias,
                          batch_first=True,
                          bidirectional=True)
        self.attention = AttentionWithContext(
            input_shape=64,  return_coefficients=return_coefficients, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attention_mask=None):
        # Get a word embedding first. x is of shape (samples, steps=512)
        # attention_mask is of same size.
        x = self.text_encoder(x, attention_mask=attention_mask).last_hidden_state
        # x needs to have shape: (samples, steps=512, features=768)
        x, _ = self.bigru(x)
        # x needs to have shape: (samples, steps=512, features=64)
        x = self.dropout(x)
        x = self.attention(x)
        # x is now of size (samples, features=64) and represents a document.
        return x

class CorpusEncoder(nn.Module):
    def __init__(self, return_coefficients=False, bias=True, dropout=0.):
        super(CorpusEncoder, self).__init__()
        self.doc_encoder = DocumentEncoder(return_coefficients=False, bias=True, dropout=dropout)
        self.W = nn.Linear(in_features=64, out_features=32)
        self.dropout = nn.Dropout(dropout)
        self.attention = AttentionWithContext(
            input_shape=32,  return_coefficients=return_coefficients, bias=bias)
    
    def forward(self, x, attention_mask=None):
        # Get a document embedding first.
        # x and attention_mask are of size: (samples, nb_docs=4, steps=512)
        # We can reshape this into (samples * nb_docs, steps=512)
        batch_size, nb_docs = x.size(0), x.size(1)
        x = x.view(batch_size * nb_docs, -1)
        attention_mask_ = attention_mask.view(batch_size * nb_docs, -1)
        x = self.doc_encoder(x, attention_mask_)
        # x is now in shape (samples * nb_docs, features=64)
        x = self.W(x)
        x = self.dropout(x)
        # x is now in shape (samples * nb_docs, features=32)
        x = x.view(batch_size, nb_docs, -1)
        # x is now of shape: (samples, steps=nb_docs=4, features=32)
        # Note : About Corpus encoding
        # In order to filter out empty entries, we can use the previous attention mask.
        # The original mask is of size (batch_size, 4, 512) in the form:
        # [
        # [[1, 1, 1, 1, ..., 1, 1, 1, 1],
        # [1, 1, 1, 1, ..., 1, 1, 1, 1]
        # [1, 1, 0, 0, ..., 0, 0, 0, 0],
        # [1, 1, 0, 0, ..., 0, 0, 0, 0]],
        # ...,
        # ]
        # Therefore, we can filter the useless entries by using a mask
        # sum > 2. The new mask will then be of size (batch_size, 4, 1)
        # [
        # [[True],
        #  [True]
        #  [False],
        #  [False]],
        # ...,
        # ]
        attention_mask = torch.sum(attention_mask, dim=-1, keepdim=True).ge(3)
        x = self.attention(x, mask=attention_mask)
        # x is now of size (samples, features=32) and represents a corpus.
        return x

class MyModel(nn.Module):
    def __init__(self, dropout=.5):
        super(MyModel, self).__init__()
        self.corpus_enc_ecb = CorpusEncoder(dropout=dropout)
        self.corpus_enc_fed = CorpusEncoder(dropout=dropout)
        nb_nontext_features = len(nontextual_cols)
        self.fc1 = nn.Linear(32 * 2 + nb_nontext_features, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x_ecb, x_ecb_mask, x_fed, x_fed_mask, x_ind):
        # x_fed and x_ind of size (batch_size, 4, 512)

        x_ecb = self.corpus_enc_ecb(x_ecb, attention_mask=x_ecb_mask)
        x_fed = self.corpus_enc_fed(x_fed, attention_mask=x_fed_mask)

        # Both of the above are now of size (batch_size, features)
        # Cast to float because for some reason cat converts to dtype float64.
        # float converts to float32.
        x = torch.cat([x_ecb, x_fed, x_ind], dim=1).float()
        # x is now of size (batch_size, 2 * features + x_ind.size(1))
        # Classification
        # print(x.dtype)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        out = self.sigmoid(x)

        return out