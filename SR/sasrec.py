import pdb
import copy
import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as fn
from SR.utils_sr import SequentialRecommender, BPRLoss


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, args):
        super(SASRec, self).__init__(args)

        # load parameters info
        self.n_layers = 2
        self.n_heads = 2
        self.hidden_size = args.hidden_size # same as embedding_size
        self.inner_size = 256  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = 0.5
        self.attn_dropout_prob = 0.5
        self.hidden_act = 'gelu'
        self.layer_norm_eps = 1e-12

        self.initializer_range = 0.02
        self.loss_type = 'CE'
        self.args = args

        # define layers and loss
        self.item_embedding = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        if args.pretraining:
            self.text_embedding = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
            self.image_embedding = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)
        # parameters initialization

        if self.args.pretraining == True:
            self.replace_embedding()

        if self.args.ban_bp == True:
            for param in self.image_embedding.parameters():
                param.requires_grad = False
            for param in self.text_embedding.parameters():
                param.requires_grad = False


    def replace_embedding(self):
        ids_features_list = None

        image_features_list = torch.load(self.args.image_path)
        if isinstance(image_features_list, np.ndarray):
            image_features_list = torch.from_numpy(image_features_list).to(self.args.device)

        text_features_list = torch.load(self.args.text_path)
        if isinstance(text_features_list, np.ndarray):
            text_features_list = torch.from_numpy(text_features_list).to(self.args.device)

        if self.args.item_path is not None:
            ids_features_list = torch.load(self.args.item_path).to(self.args.device)

        # self.image_embedding.weight.data[1:, :] = fn.normalize(image_features_list)
        # self.text_embedding.weight.data[1:, :] = fn.normalize(text_features_list)
        self.image_embedding.weight.data[1:, :] = image_features_list
        self.text_embedding.weight.data[1:, :] = text_features_list
        if ids_features_list is not None:
            # self.item_embedding.weight.data[1:, :] = fn.normalize(ids_features_list)
            self.item_embedding.weight.data[1:, :] = ids_features_list

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # CF https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        if self.args.pretraining == True:
            item_emb = self.item_embedding(item_seq)
            image_emb = self.image_embedding(item_seq)
            text_emb = self.text_embedding(item_seq)
            item_emb = item_emb + image_emb + text_emb
        else:
            item_emb = self.item_embedding(item_seq)
        # item_emb = self.item_embedding(item_seq)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)


        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        # outputs = self.gather_indexes(outputs, item_seq_len - 1)
        output = output[:, -1, :]
        return output  # [B H]

    def cross_entropy(self, pos_logits, neg_logits):
        # item_seq = interaction[self.ITEM_SEQ]
        # item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # seq_out = self.forward(item_seq, item_seq_len)
        # pos_ids = interaction[self.POS_ITEM_ID]
        # neg_ids = interaction[self.NEG_ITEM_ID]
        # # [batch seq_len hidden_size]
        # pos_emb = self.model.item_embeddings(pos_ids)
        # neg_emb = self.model.item_embeddings(neg_ids)
        # # [batch*seq_len hidden_size]
        # pos = pos_emb.view(-1, pos_emb.size(2))
        # neg = neg_emb.view(-1, neg_emb.size(2))
        # seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        # pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        # neg_logits = torch.sum(neg * seq_emb, -1)
        # istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        # loss = torch.sum(
        #     - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
        #     torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        # ) / torch.sum(istarget)
        #
        # auc = torch.sum(
        #     ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        # ) / torch.sum(istarget)

        loss = torch.mean(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
        )

        auc = torch.sum(
            ((torch.sign(pos_logits - neg_logits) + 1) / 2)
        )

        return loss, auc

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            #
            if self.args.pretraining == True:
                pos_items_emb = self.item_embedding(pos_items)
                pos_image_emb = self.image_embedding(pos_items)
                pos_text_emb = self.text_embedding(pos_items)

                pos_items_emb = pos_items_emb + pos_image_emb + pos_text_emb

                neg_items_emb = self.item_embedding(neg_items)
                neg_image_emb = self.image_embedding(neg_items)
                neg_text_emb = self.text_embedding(neg_items)
                neg_items_emb = neg_items_emb + neg_image_emb + neg_text_emb
            else:
                pos_items_emb = self.item_embedding(pos_items)
                neg_items_emb = self.item_embedding(neg_items)

            # pos_items_emb = self.item_embedding(pos_items)
            # neg_items_emb = self.item_embedding(neg_items)

            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            # pos_score = torch.sum(pos_items_emb * seq_output, dim=-1)  # [B]
            # neg_score = torch.sum(neg_items_emb * seq_output, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            # loss, _ = self.cross_entropy(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            if self.args.train_tv == True:
                test_image_emb = self.image_embedding.weight
                test_text_emb = self.text_embedding.weight
                test_item_emb = test_item_emb + test_image_emb + test_text_emb
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            # loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        if self.args.train_tv == True:
            test_image_emb = self.image_embedding.weight
            test_text_emb = self.text_embedding.weight
            test_items_emb = test_items_emb + test_image_emb + test_text_emb
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

class TransformerEncoder(nn.Module):
    r"""One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and outputs hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether outputs all transformer layers' outputs

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' outputs, otherwise return a list only consists of the outputs of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The outputs of the point-wise feed-forward sublayer,
                                           is the outputs of the transformer layer.

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output

class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the outputs of the multi-head self-attention layer

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the outputs of the point-wise feed-forward layer

    """

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class MLP_layer(torch.nn.Module):
    def __init__(self, dim_latent):
        super(MLP_layer, self).__init__()
        self.dim_latent = dim_latent
        self.dim_feat = 128
        self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
        self.MLP_1 = nn.Linear(self.dim_latent, self.dim_latent)

    def forward(self, features):
        temp_features = self.MLP_1(fn.leaky_relu(self.MLP(features)))
        return temp_features