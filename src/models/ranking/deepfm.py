import torch
import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self, total_vocab_size, num_sparse_features, num_dense_features=0,
                 embedding_dim=16, hidden_dims=[64, 32], dropout=0.5):
        """
        DeepFM with Shared Embedding Matrix (Slot-based)
        :param total_vocab_size: Total number of unique slots (global index size)
        :param num_sparse_features: Number of sparse feature fields (slots per sample)
        :param num_dense_features: Number of dense/numerical features
        :param embedding_dim: Embedding dimension
        :param hidden_dims: DNN hidden layers
        """
        super(DeepFM, self).__init__()
        
        self.num_sparse_features = num_sparse_features
        self.num_dense_features = num_dense_features
        
        # 1. Sparse Feature Embeddings (Shared)
        # For FM 2nd order and Deep part
        self.embedding = nn.Embedding(total_vocab_size, embedding_dim)
        
        # 2. Linear Part (FM 1st order)
        # Sparse Linear: Embedding(vocab, 1)
        self.linear_sparse = nn.Embedding(total_vocab_size, 1)
        # Dense Linear: Linear(num_dense, 1)
        if num_dense_features > 0:
            self.linear_dense = nn.Linear(num_dense_features, 1)
        
        # 3. Deep Part
        # Input: Flattened Sparse Embeddings + Dense Features
        input_dim = num_sparse_features * embedding_dim + num_dense_features
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = dim
            
        self.dnn = nn.Sequential(*layers)
        self.dnn_linear = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, sparse_inputs, dense_inputs=None):
        """
        :param sparse_inputs: Tensor (batch_size, num_sparse_features) containing global slot indices
        :param dense_inputs: Tensor (batch_size, num_dense_features) containing float values
        """
        batch_size = sparse_inputs.size(0)
        
        # --- FM Component ---
        # 1. First Order (Linear)
        # Sparse: look up weights and sum
        linear_part = torch.sum(self.linear_sparse(sparse_inputs), dim=1) # (batch, 1)
        
        # Dense: matmul
        if self.num_dense_features > 0 and dense_inputs is not None:
            linear_part = linear_part + self.linear_dense(dense_inputs)
            
        # 2. Second Order (Interaction)
        # Retrieve embeddings: (batch, num_sparse, emb_dim)
        embeds_stack = self.embedding(sparse_inputs)
        
        sum_square = torch.pow(torch.sum(embeds_stack, dim=1), 2)
        square_sum = torch.sum(torch.pow(embeds_stack, 2), dim=1)
        
        fm_2nd_order = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True) # (batch, 1)
        
        # --- Deep Component ---
        dnn_input = embeds_stack.view(batch_size, -1) # Flatten sparse embeddings
        
        if self.num_dense_features > 0 and dense_inputs is not None:
            dnn_input = torch.cat([dnn_input, dense_inputs], dim=1)
            
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        
        # --- Combine ---
        total_logit = linear_part + fm_2nd_order + dnn_logit
        return torch.sigmoid(total_logit)
