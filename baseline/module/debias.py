import torch
import torch.nn as nn


def build_debias_model(model_class):
    class HawkesDebias(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.softplus = nn.Softplus()
            self.base_net = self._build_mlp(self.embedding_k, self.embedding_k//2, self.depth)
            self.excitation_net = self._build_mlp(self.embedding_k, self.embedding_k//2, self.depth)
            self.log_beta = nn.Parameter(torch.zeros(()))
            self.reset_parameters()

        @staticmethod
        def _build_mlp(input_dim, hidden_dim, depth):
            layers = []
            in_dim = input_dim
            for _ in range(max(depth, 1)):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.Softplus())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, 1, bias=False))
            return nn.Sequential(*layers)

        def reset_parameters(self):
            nn.init.normal_(self.item_embedding.weight, std=0.02)
            for module in list(self.base_net) + list(self.excitation_net):
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def current_beta(self):
            return self.softplus(self.log_beta) + 1e-6

        def prior_parameters_from_embeddings(self):
            z = self.item_embedding.weight
            mu = self.softplus(self.base_net(z)).squeeze(-1) + 1e-8
            alpha = self.softplus(self.excitation_net(z)).squeeze(-1) + 1e-8
            beta = self.current_beta()
            return mu, alpha, beta

        def prior(self, batch_items, pos_time, batch_time_all):
            item_vec = self.item_embedding(batch_items)
            beta = self.current_beta()
            query = pos_time.view(-1, 1, 1)
            mask = batch_time_all < query
            delta = (query - batch_time_all).clamp(min=0.0)
            h = (torch.exp(-beta * delta) * mask).sum(dim=-1)
            mB, C = h.shape 
            mu = self.softplus(self.base_net(item_vec)).reshape(mB, C) + 1e-8
            alpha = self.softplus(self.excitation_net(item_vec)).reshape(mB, C) + 1e-8
            return mu + alpha * h

    return HawkesDebias


def build_unshared_debias_model(model_class):
    class HawkesDebias(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.softplus = nn.Softplus()
            self.base_net = self._build_mlp(self.embedding_k, self.embedding_k//2, self.depth)
            self.excitation_net = self._build_mlp(self.embedding_k, self.embedding_k//2, self.depth)
            self.log_beta = nn.Parameter(torch.zeros(()))
            self.p_item_embedding = nn.Embedding(self.num_items, self.embedding_k)
            self.reset_parameters()

        @staticmethod
        def _build_mlp(input_dim, hidden_dim, depth):
            layers = []
            in_dim = input_dim
            for _ in range(max(depth, 1)):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.Softplus())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, 1, bias=False))
            return nn.Sequential(*layers)

        def reset_parameters(self):
            nn.init.normal_(self.p_item_embedding.weight, std=0.02)
            for module in list(self.base_net) + list(self.excitation_net):
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def current_beta(self):
            return self.softplus(self.log_beta) + 1e-6

        def prior_parameters_from_embeddings(self):
            z = self.p_item_embedding.weight
            mu = self.softplus(self.base_net(z)).squeeze(-1) + 1e-8
            alpha = self.softplus(self.excitation_net(z)).squeeze(-1) + 1e-8
            beta = self.current_beta()
            return mu, alpha, beta

        def prior(self, batch_items, pos_time, batch_time_all):
            item_vec = self.p_item_embedding(batch_items)
            beta = self.current_beta()
            query = pos_time.view(-1, 1, 1)
            mask = batch_time_all < query
            delta = (query - batch_time_all).clamp(min=0.0)
            h = (torch.exp(-beta * delta) * mask).sum(dim=-1)
            mB, C = h.shape 
            mu = self.softplus(self.base_net(item_vec)).reshape(mB, C) + 1e-8
            alpha = self.softplus(self.excitation_net(item_vec)).reshape(mB, C) + 1e-8
            return mu + alpha * h

    return HawkesDebias


def build_linear_debias_model(model_class):
    class HawkesDebias(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.softplus = nn.Softplus()
            self.base_net = self._build_mlp(self.embedding_k, self.embedding_k//2, self.depth)
            self.excitation_net = self._build_mlp(self.embedding_k, self.embedding_k//2, self.depth)
            self.log_beta = nn.Parameter(torch.zeros(()))
            self.reset_parameters()

        @staticmethod
        def _build_mlp(input_dim, hidden_dim, depth):
            layers = []
            layers.append(nn.Linear(input_dim, 1, bias=False))
            return nn.Sequential(*layers)

        def reset_parameters(self):
            nn.init.normal_(self.item_embedding.weight, std=0.02)
            for module in list(self.base_net) + list(self.excitation_net):
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def current_beta(self):
            return self.softplus(self.log_beta) + 1e-6

        def prior_parameters_from_embeddings(self):
            z = self.item_embedding.weight
            mu = self.softplus(self.base_net(z)).squeeze(-1) + 1e-8
            alpha = self.softplus(self.excitation_net(z)).squeeze(-1) + 1e-8
            beta = self.current_beta()
            return mu, alpha, beta

        def prior(self, batch_items, pos_time, batch_time_all):
            item_vec = self.item_embedding(batch_items)
            beta = self.current_beta()
            query = pos_time.view(-1, 1, 1)
            mask = batch_time_all < query
            delta = (query - batch_time_all).clamp(min=0.0)
            h = (torch.exp(-beta * delta) * mask).sum(dim=-1)
            mB, C = h.shape 
            mu = self.softplus(self.base_net(item_vec)).reshape(mB, C) + 1e-8
            alpha = self.softplus(self.excitation_net(item_vec)).reshape(mB, C) + 1e-8
            return mu + alpha * h

    return HawkesDebias


def build_unshared_linear_debias_model(model_class):
    class HawkesDebias(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.softplus = nn.Softplus()
            self.base_net = self._build_mlp(self.embedding_k, self.embedding_k//2, self.depth)
            self.excitation_net = self._build_mlp(self.embedding_k, self.embedding_k//2, self.depth)
            self.log_beta = nn.Parameter(torch.zeros(()))
            self.p_item_embedding = nn.Embedding(self.num_items, self.embedding_k)
            self.reset_parameters()

        @staticmethod
        def _build_mlp(input_dim, hidden_dim, depth):
            layers = []
            layers.append(nn.Linear(input_dim, 1, bias=False))
            return nn.Sequential(*layers)

        def reset_parameters(self):
            nn.init.normal_(self.p_item_embedding.weight, std=0.02)
            for module in list(self.base_net) + list(self.excitation_net):
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def current_beta(self):
            return self.softplus(self.log_beta) + 1e-6

        def prior_parameters_from_embeddings(self):
            z = self.p_item_embedding.weight
            mu = self.softplus(self.base_net(z)).squeeze(-1) + 1e-8
            alpha = self.softplus(self.excitation_net(z)).squeeze(-1) + 1e-8
            beta = self.current_beta()
            return mu, alpha, beta

        def prior(self, batch_items, pos_time, batch_time_all):
            item_vec = self.p_item_embedding(batch_items)
            beta = self.current_beta()
            query = pos_time.view(-1, 1, 1)
            mask = batch_time_all < query
            delta = (query - batch_time_all).clamp(min=0.0)
            h = (torch.exp(-beta * delta) * mask).sum(dim=-1)
            mB, C = h.shape 
            mu = self.softplus(self.base_net(item_vec)).reshape(mB, C) + 1e-8
            alpha = self.softplus(self.excitation_net(item_vec)).reshape(mB, C) + 1e-8
            return mu + alpha * h

    return HawkesDebias

