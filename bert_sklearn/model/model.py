import torch
import torch.nn as nn

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE

from .pytorch_pretrained import BertModel
from .pytorch_pretrained import BertPreTrainedModel


def LinearBlock(H1, H2, p):
    return nn.Sequential(
        nn.Linear(H1, H2),
        nn.BatchNorm1d(H2),
        nn.ReLU(),
        nn.Dropout(p))

def MLP(D, n, H, K, p):
    """
    MLP w batchnorm and dropout.

    Parameters
    ----------
    D : int, size of input layer
    n : int, number of hidden layers
    H : int, size of hidden layer
    K : int, size of output layer
    p : float, dropout probability
    """

    if n == 0:
        print("Defaulting to linear classifier/regressor")
        return nn.Linear(D, K)
    else:
        print("Using mlp with D=%d,H=%d,K=%d,n=%d"%(D, H, K, n))
        layers = [nn.BatchNorm1d(D),
                  LinearBlock(D, H, p)]
        for _ in range(n-1):
            layers.append(LinearBlock(H, H, p))
        layers.append(nn.Linear(H, K))
        return torch.nn.Sequential(*layers)


class BertPlusMLP(BertPreTrainedModel):
    """
    Bert model with MLP classifier/regressor head.

    Based on pytorch_pretrained_bert.modeling.BertForSequenceClassification

    Parameters
    ----------
    config : BertConfig
        stores configuration of BertModel

    model_type : string
         'text_classifier' | 'text_regressor' | 'token_classifier'

    num_labels : int
        For a classifier, this is the number of distinct classes.
        For a regressor his will be 1.

    num_mlp_layers : int
        the number of mlp layers. If set to 0, then defualts
        to the linear classifier/regresor in the original Google paper and code.

    num_mlp_hiddens : int
        the number of hidden neurons in each layer of the mlp.

    oversampler : str
        the algorithm to use to oversample examples of the minority class for unbalanced datasets.
        Must be one of "SMOTE", "ADASYN", "BorderlineSmote", "KMeansSMOTE", "SVMSMOTE", with case not important.
    """

    def __init__(self, config,
                 model_type="text_classifier",
                 num_labels=2,
                 num_mlp_layers=2,
                 num_mlp_hiddens=500,
                 oversampler=None):

        if oversampler is not None:
            assert oversampler in ["smote", "adasyn", "borderlinesmote", "kmeanssmote", "svmsmote"]

        super(BertPlusMLP, self).__init__(config)
        self.model_type = model_type
        self.num_labels = num_labels
        self.num_mlp_layers = num_mlp_layers
        self.num_mlp_hiddens = num_mlp_hiddens
        self.oversampler = self.__oversampler_for_str(oversampler)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert = BertModel(config)
        self.input_dim = config.hidden_size

        self.mlp = MLP(D=self.input_dim,
                       n=self.num_mlp_layers,
                       H=self.num_mlp_hiddens,
                       K=self.num_labels,
                       p=config.hidden_dropout_prob)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids=None, input_mask=None, labels=None, is_train=False):

        hidden, pooled_output = self.bert(input_ids,
                                          segment_ids,
                                          input_mask,
                                          output_all_encoded_layers=False)

        if self.model_type == "token_classifier":
            output = hidden
        else:
            output = pooled_output
            output = self.dropout(output)

        if is_train and self.oversampler:
            output, labels = self.oversampler.fit_resample(output.detach().numpy(), labels.detach().numpy())
            output = torch.from_numpy(output).to(input_ids.device)
            labels = torch.from_numpy(labels).to(input_ids.device)

        output = self.mlp(output)

        if labels is not None:
            if self.model_type == "text_classifier":
                loss_criterion = nn.CrossEntropyLoss(reduction='none')
                loss = loss_criterion(output.view(-1, output.shape[-1]), labels.view(-1))
            elif self.model_type == "text_regressor":
                loss_criterion = nn.MSELoss(reduction='none')
                output = torch.squeeze(output)
                loss = loss_criterion(output, labels)
            elif self.model_type == "token_classifier":
                loss_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
                loss = loss_criterion(output.view(-1, output.shape[-1]), labels.view(-1))
            return loss, output
        else:
            return output

    def __oversampler_for_str(self, oversampler_str, k_neighbors=5, m_neighbors=10):
        """
        Helper method for instantaiating an oversampler from a specified string.

        Parameters
        ----------
        oversampler_str : str
            the algorithm to use to oversample examples of the minority class for unbalanced datasets.
            Must be one of "SMOTE", "ADASYN", "BorderlineSmote", "KMeansSMOTE", "SVMSMOTE", with case not important.

        k_neighbors : int
             number of nearest neighbours to used to construct synthetic samples

        m_neighbors : int
            number of nearest neighbours to used to construct synthetic samples.
            Only used if oversampler_str == "BorderlineSMOTE" || SVMSMOTE
        """
        if oversampler_str:
            if oversampler_str.lower() == "smote":
                return SMOTE(k_neighbors=k_neighbors)
            elif oversampler_str.lower() == "adasyn":
                return ADASYN(n_neighbors=k_neighbors)
            elif oversampler_str.lower() == "borderlinesmote":
                return BorderlineSMOTE(k_neighbors=k_neighbors, m_neighbors=m_neighbors)
            elif oversampler_str.lower() == "kmeanssmote":
                return KMeansSMOTE(k_neighbors=k_neighbors)
            elif oversampler_str.lower() == "svmsmote":
                return SVMSMOTE(k_neighbors=k_neighbors, m_neighbors=m_neighbors)


