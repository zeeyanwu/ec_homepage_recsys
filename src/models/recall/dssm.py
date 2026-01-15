import torch
import torch.nn as nn
import torch.nn.functional as F

class DSSM(nn.Module):
    def __init__(self, total_vocab_size, embedding_dim=32, hidden_dims=[64, 32], 
                 user_feature_count=3, item_feature_count=4):
        """
        DSSM Model for Recall with Shared Embedding Matrix (Slot-based)
        :param total_vocab_size: Total number of unique slots (global index size)
        :param embedding_dim: Dimension of embedding vectors
        :param hidden_dims: List of hidden layer dimensions for MLP towers
        :param user_feature_count: Number of feature fields for user tower
        :param item_feature_count: Number of feature fields for item tower
        """
        super(DSSM, self).__init__()
        
        # Shared Embedding Matrix
        self.embedding = nn.Embedding(total_vocab_size, embedding_dim)
        
        # Calculate input dimension for MLPs
        # Input is concatenation of embeddings for each field
        user_input_dim = user_feature_count * embedding_dim
        item_input_dim = item_feature_count * embedding_dim
        
        # User Tower MLP
        layers = []
        input_dim = user_input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        self.user_tower = nn.Sequential(*layers)
        
        # Item Tower MLP
        layers = []
        input_dim = item_input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        self.item_tower = nn.Sequential(*layers)
        
    def forward(self, user_inputs, item_inputs):
        """
        :param user_inputs: Tensor of shape (batch_size, user_feature_count) containing global slot indices
        :param item_inputs: Tensor of shape (batch_size, item_feature_count) containing global slot indices
        :return: cosine similarity score
        """
        # User Tower Forward
        # [batch_size, user_feat_count] -> [batch_size, user_feat_count, embed_dim]
        user_embeds = self.embedding(user_inputs)
        # Flatten: [batch_size, user_feat_count * embed_dim]
        user_concat = user_embeds.view(user_embeds.size(0), -1)
        user_vector = self.user_tower(user_concat)
        
        # Item Tower Forward
        # [batch_size, item_feat_count] -> [batch_size, item_feat_count, embed_dim]
        item_embeds = self.embedding(item_inputs)
        # Flatten: [batch_size, item_feat_count * embed_dim]
        item_concat = item_embeds.view(item_embeds.size(0), -1)
        item_vector = self.item_tower(item_concat)
        
        # Normalize vectors
        user_vector = F.normalize(user_vector, p=2, dim=1)
        item_vector = F.normalize(item_vector, p=2, dim=1)
        
        # Cosine Similarity (Dot product of normalized vectors)
        score = torch.sum(user_vector * item_vector, dim=1)
        return score, user_vector, item_vector

    def get_user_vector(self, user_inputs):
        user_embeds = self.embedding(user_inputs)
        user_concat = user_embeds.view(user_embeds.size(0), -1)
        user_vector = self.user_tower(user_concat)
        return F.normalize(user_vector, p=2, dim=1)

    def get_item_vector(self, item_inputs):
        item_embeds = self.embedding(item_inputs)
        item_concat = item_embeds.view(item_embeds.size(0), -1)
        item_vector = self.item_tower(item_concat)
        return F.normalize(item_vector, p=2, dim=1)
