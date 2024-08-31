import pdb

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from torch.nn.init import normal_
import torch.nn.functional as F


def activation_layer(activation_name="relu", emb_dim=None):
    """Construct activation layers

    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation

    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "dice":
            activation = Dice(emb_dim)
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation


class Dice(nn.Module):
    r"""Dice activation function

    .. math::
        f(s)=p(s) \cdot s+(1-p(s)) \cdot \alpha s

    .. math::
        p(s)=\frac{1} {1 + e^{-\frac{s-E[s]} {\sqrt {Var[s] + \epsilon}}}}
    """

    def __init__(self, emb_size):
        super(Dice, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.alpha = torch.zeros((emb_size,))

    def forward(self, score):
        self.alpha = self.alpha.to(score.device)
        score_p = self.sigmoid(score)

        return self.alpha * (1 - score_p) * score + score_p * score


class MLPLayers(nn.Module):
    def __init__(
            self, layers, dropout=0.0, activation="relu", bn=False, init_method=None
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
                zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            if self.init_method == "norm":
                xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


class RegLoss(nn.Module):
    """RegLoss, L2 regularization on model parameters"""

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss

class FirstOrderLinear(nn.Module):
    def __init__(self, args):
        super(FirstOrderLinear, self).__init__()
        self.args = args
        self.user_embedding_fm = nn.Embedding(args.num_users, 1)
        self.item_embedding_fm = nn.Embedding(args.max_item, 1)
        self.image_embedding_fm = nn.Embedding(args.max_item, 1)
        self.text_embedding_fm = nn.Embedding(args.max_item, 1)
        self.class_embedding_fm = nn.Embedding(args.class_num, 1)

        self.bias = nn.Parameter(torch.zeros((1,)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding) or isinstance(module, nn.Conv1d):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, interaction, device):
        if self.args.pretraining == 'True':
            # your logic for pre_id=True
            uer_emb_fm = self.user_embedding_fm(interaction[0].to(device))
            id_emb_fm = self.item_embedding_fm(interaction[1].to(device))
            image_emb_fm = self.image_embedding_fm(interaction[1].to(device))
            text_emb_fm = self.text_embedding_fm(interaction[1].to(device))
            id_emb_fm = (image_emb_fm + id_emb_fm + text_emb_fm)
            class_emb_fm = self.class_embedding_fm(interaction[2].to(device))
        else:
            uer_emb_fm = self.user_embedding_fm(interaction[0].to(device))
            id_emb_fm = self.item_embedding_fm(interaction[1].to(device))
            class_emb_fm = self.class_embedding_fm(interaction[2].to(device))

        token_embedding = torch.cat((uer_emb_fm, id_emb_fm, class_emb_fm), dim=1)
        token_embedding = torch.sum(token_embedding, dim=1, keepdim=True)
        return token_embedding + self.bias

class xDeepFM(nn.Module):
    def __init__(self, args):
        super(xDeepFM, self).__init__()

        # load parameters info
        self.args = args
        self.embedding_size = args.embedding_size
        self.num_feature_field = args.num_feature_field
        self.mlp_hidden_size = [128, 128, 128]
        self.reg_weight = 1e-7
        self.dropout_prob = args.dropout_prob
        self.direct = False
        self.cin_layer_size = temp_cin_size = [120, 120, 120]
        self.args = args

        self.user_embedding = nn.Embedding(args.num_users, self.embedding_size)
        self.item_embedding = nn.Embedding(args.max_item, self.embedding_size)
        self.text_embedding = nn.Embedding(args.max_item, self.embedding_size)

        self.image_embedding = nn.Embedding(args.max_item, self.embedding_size)
        self.class_embedding = nn.Embedding(args.class_num, self.embedding_size)

        self.first_order_linear = FirstOrderLinear(args)

        # Check whether the size of the CIN layer is legal.
        if not self.direct:
            self.cin_layer_size = list(map(lambda x: int(x // 2 * 2), temp_cin_size))
            if self.cin_layer_size[:-1] != temp_cin_size[:-1]:
                self.logger.warning(
                    "Layer size of CIN should be even except for the last layer when direct is True."
                    "It is changed to {}".format(self.cin_layer_size)
                )

        # Create a convolutional layer for each CIN layer
        self.conv1d_list = nn.ModuleList()
        self.field_nums = [self.num_feature_field]
        for i, layer_size in enumerate(self.cin_layer_size):
            conv1d = nn.Conv1d(self.field_nums[-1] * self.field_nums[0], layer_size, 1)
            self.conv1d_list.append(conv1d)
            if self.direct:
                self.field_nums.append(layer_size)
            else:
                self.field_nums.append(layer_size // 2)

        # Create MLP layer
        size_list = (
                [self.embedding_size * self.num_feature_field] + self.mlp_hidden_size + [1]
        )
        self.mlp_layers = MLPLayers(size_list, dropout=self.dropout_prob)

        # Get the outputs size of CIN
        if self.direct:
            self.final_len = sum(self.cin_layer_size)
        else:
            self.final_len = (
                    sum(self.cin_layer_size[:-1]) // 2 + self.cin_layer_size[-1]
            )

        self.cin_linear = nn.Linear(self.final_len, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
        self.apply(self._init_weights)
        #
        if self.args.pretraining == 'True':
            self.replace_embedding()

    def replace_embedding(self):
        image_features_list = torch.load(self.args.image_path).to(self.args.device)
        text_features_list = torch.load(self.args.text_path).to(self.args.device)
        ids_features_list = torch.load(self.args.item_path).to(self.args.device)
        self.image_embedding.weight.data[1:, :] = image_features_list
        self.text_embedding.weight.data[1:, :] = text_features_list
        self.item_embedding.weight.data[1:, :] = ids_features_list

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding) or isinstance(module, nn.Conv1d):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def reg_loss(self, parameters):
        reg_loss = 0
        for name, parm in parameters:
            if name.endswith("weight"):
                reg_loss = reg_loss + parm.norm(2)
        return reg_loss

    def calculate_reg_loss(self):
        l2_reg = 0
        l2_reg = l2_reg + self.reg_loss(self.mlp_layers.named_parameters())
        l2_reg = l2_reg + self.reg_loss(self.first_order_linear.named_parameters())
        for conv1d in self.conv1d_list:
            l2_reg += self.reg_loss(conv1d.named_parameters())
        return l2_reg

    def compressed_interaction_network(self, input_features, activation="ReLU"):
        batch_size, _, embedding_size = input_features.shape
        hidden_nn_layers = [input_features]
        final_result = []
        for i, layer_size in enumerate(self.cin_layer_size):
            z_i = torch.einsum(
                "bhd,bmd->bhmd", hidden_nn_layers[-1], hidden_nn_layers[0]
            )
            z_i = z_i.view(
                batch_size, self.field_nums[0] * self.field_nums[i], embedding_size
            )
            z_i = self.conv1d_list[i](z_i)

            # Pass the CIN intermediate result through the activation function.
            if activation.lower() == "identity":
                output = z_i
            else:
                activate_func = activation_layer(activation)
                if activate_func is None:
                    output = z_i
                else:
                    output = activate_func(z_i)

            # Get the outputs of the hidden layer.
            if self.direct:
                direct_connect = output
                next_hidden = output
            else:
                if i != len(self.cin_layer_size) - 1:
                    next_hidden, direct_connect = torch.split(
                        output, 2 * [layer_size // 2], 1
                    )
                else:
                    direct_connect = output
                    next_hidden = 0

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)
        result = torch.cat(final_result, dim=1)
        result = torch.sum(result, -1)
        return result

    def forward(self, args, interaction, device):
        if args.pretraining == 'True':
            uer_emb = self.user_embedding(interaction[0].to(device)).unsqueeze(1)

            id_emb = self.item_embedding(interaction[1].to(device)).unsqueeze(1)
            image_emb = self.image_embedding(interaction[1].to(device)).unsqueeze(1)
            text_emb = self.text_embedding(interaction[1].to(device)).unsqueeze(1)
            id_emb = (id_emb + image_emb + text_emb)

            class_emb = self.class_embedding(interaction[2].to(device)).unsqueeze(1)
            uer_emb = F.normalize(uer_emb, p=2, dim=-1)
            id_emb = F.normalize(id_emb, p=2, dim=-1)
            class_emb = F.normalize(class_emb, p=2, dim=-1)

            xdeepfm_input = torch.concat((uer_emb, id_emb, class_emb), dim=1)
        else:
            uer_emb = self.user_embedding(interaction[0].to(device)).unsqueeze(1)
            id_emb = self.item_embedding(interaction[1].to(device)).unsqueeze(1)
            class_emb = self.class_embedding(interaction[2].to(device)).unsqueeze(1)
            uer_emb = F.normalize(uer_emb, p=2, dim=-1)
            id_emb = F.normalize(id_emb, p=2, dim=-1)
            class_emb = F.normalize(class_emb, p=2, dim=-1)
            xdeepfm_input = torch.concat((uer_emb, id_emb, class_emb), dim=1)

        # Get the outputs of CIN.
        cin_output = self.compressed_interaction_network(xdeepfm_input)
        cin_output = self.cin_linear(cin_output)

        # Get the outputs of MLP layer.
        batch_size = xdeepfm_input.shape[0]
        dnn_output = self.mlp_layers(xdeepfm_input.view(batch_size, -1))

        # Get predicted score.
        y_p = self.first_order_linear(interaction, device) + cin_output + dnn_output

        return y_p.squeeze(1)

    def calculate_loss(self, args, interaction, device):
        label = interaction[-1].float().to(device)
        output = self.forward(args, interaction, device)
        l2_reg = self.calculate_reg_loss()
        return self.loss(output, label) + self.reg_weight * l2_reg

    def predict(self, args, interaction, device):
        return self.sigmoid(self.forward(args, interaction, device))

