import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def _default_hidden_dim(out_feat, hidden_feat=None, hidden_factor=2, hidden_cap=256):
	if hidden_feat is not None:
		return int(hidden_feat)
	return int(min(hidden_cap, max(out_feat, out_feat * hidden_factor)))


class GraphLinearBlock(Module):

	"""
	Two-step graph propagation block with residual connection and layer norm.
	"""

	def __init__(self, in_feat, hidden_feat, out_feat, dropout=0.0, act=F.elu, use_layernorm=True):
		super(GraphLinearBlock, self).__init__()
		self.in_feat = int(in_feat)
		self.hidden_feat = int(hidden_feat)
		self.out_feat = int(out_feat)
		self.dropout = float(dropout)
		self.act = act
		self.use_layernorm = use_layernorm

		self.weight1 = Parameter(torch.FloatTensor(self.in_feat, self.hidden_feat))
		self.weight2 = Parameter(torch.FloatTensor(self.hidden_feat, self.out_feat))
		self.weight_residual = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

		self.norm_hidden = nn.LayerNorm(self.hidden_feat) if self.use_layernorm else nn.Identity()
		self.norm_out = nn.LayerNorm(self.out_feat) if self.use_layernorm else nn.Identity()

		self.reset_parameters()

	def reset_parameters(self):
		torch.nn.init.xavier_uniform_(self.weight1)
		torch.nn.init.xavier_uniform_(self.weight2)
		torch.nn.init.xavier_uniform_(self.weight_residual)

	def _graph_linear(self, feat, adj, weight):
		x = torch.mm(feat, weight)
		return torch.spmm(adj, x)

	def forward(self, feat, adj):
		hidden = self._graph_linear(feat, adj, self.weight1)
		hidden = self.norm_hidden(hidden)
		hidden = self.act(hidden)
		hidden = F.dropout(hidden, p=self.dropout, training=self.training)

		out = self._graph_linear(hidden, adj, self.weight2)
		residual = torch.mm(feat, self.weight_residual)
		out = self.norm_out(out + residual)
		out = self.act(out)
		out = F.dropout(out, p=self.dropout, training=self.training)
		return out


class Encoder(Module):

	"""
	Modality-specific GNN encoder with a deeper residual graph block.
	"""

	def __init__(self, in_feat, out_feat, hidden_feat=None, dropout=0.0, act=F.elu, use_layernorm=True):
		super(Encoder, self).__init__()
		hidden_feat = _default_hidden_dim(out_feat, hidden_feat=hidden_feat)
		self.block = GraphLinearBlock(
			in_feat=in_feat,
			hidden_feat=hidden_feat,
			out_feat=out_feat,
			dropout=dropout,
			act=act,
			use_layernorm=use_layernorm,
		)

	def forward(self, feat, adj):
		return self.block(feat, adj)


class Decoder(Module):

	"""
	Modality-specific GNN decoder mirroring the encoder depth.
	"""

	def __init__(self, in_feat, out_feat, hidden_feat=None, dropout=0.0, act=F.elu, use_layernorm=True):
		super(Decoder, self).__init__()
		hidden_feat = _default_hidden_dim(max(in_feat, out_feat), hidden_feat=hidden_feat)
		self.block = GraphLinearBlock(
			in_feat=in_feat,
			hidden_feat=hidden_feat,
			out_feat=out_feat,
			dropout=dropout,
			act=act,
			use_layernorm=use_layernorm,
		)

	def forward(self, feat, adj):
		return self.block(feat, adj)


class MLPProjector(Module):

	"""
	Projection head used by contrastive alignment losses.
	"""

	def __init__(self, in_feat, hidden_feat=None, out_feat=None, dropout=0.0):
		super(MLPProjector, self).__init__()
		out_feat = in_feat if out_feat is None else int(out_feat)
		hidden_feat = _default_hidden_dim(in_feat, hidden_feat=hidden_feat)
		self.layers = nn.Sequential(
			nn.Linear(in_feat, hidden_feat),
			nn.LayerNorm(hidden_feat),
			nn.ELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_feat, out_feat),
		)

	def forward(self, feat):
		return self.layers(feat)


class Encoder_overall(Module):

	"""\
	Overall encoder supporting two or three modalities.

	Parameters
	----------
	dim_in_feat_omics1 : int
		Dimension of input features for omics1.
	dim_in_feat_omics2 : int
		Dimension of input features for omics2.
	dim_out_feat_omics1 : int
		Dimension of latent representation for omics1.
	dim_out_feat_omics2 : int
		Dimension of latent representation for omics2.
	dim_in_feat_omics3 : int, optional
		Dimension of input features for omics3.
	dim_out_feat_omics3 : int, optional
		Dimension of latent representation for omics3.

	Returns
	-------
	results: a dictionary including latent representations and modality weights.
	"""

	def __init__(
		self,
		dim_in_feat_omics1,
		dim_out_feat_omics1,
		dim_in_feat_omics2,
		dim_out_feat_omics2,
		dim_in_feat_omics3=None,
		dim_out_feat_omics3=None,
		dropout=0.1,
		act=F.elu,
		hidden_dim=None,
		projector_dim=None,
		use_layernorm=True,
	):
		super(Encoder_overall, self).__init__()
		self.dropout = float(dropout)
		self.act = act

		self.modality_names = ["omics1", "omics2"]
		modality_dims = {
			"omics1": (int(dim_in_feat_omics1), int(dim_out_feat_omics1)),
			"omics2": (int(dim_in_feat_omics2), int(dim_out_feat_omics2)),
		}
		if dim_in_feat_omics3 is not None and dim_out_feat_omics3 is not None:
			self.modality_names.append("omics3")
			modality_dims["omics3"] = (int(dim_in_feat_omics3), int(dim_out_feat_omics3))

		self.spatial_encoders = nn.ModuleDict()
		self.feature_encoders = nn.ModuleDict()
		self.decoders = nn.ModuleDict()
		self.modality_attentions = nn.ModuleDict()
		self.projectors = nn.ModuleDict()

		for modality_name in self.modality_names:
			in_dim, out_dim = modality_dims[modality_name]
			self.spatial_encoders[modality_name] = Encoder(
				in_dim,
				out_dim,
				hidden_feat=hidden_dim,
				dropout=self.dropout,
				act=self.act,
				use_layernorm=use_layernorm,
			)
			self.feature_encoders[modality_name] = Encoder(
				in_dim,
				out_dim,
				hidden_feat=hidden_dim,
				dropout=self.dropout,
				act=self.act,
				use_layernorm=use_layernorm,
			)
			self.decoders[modality_name] = Decoder(
				out_dim,
				in_dim,
				hidden_feat=hidden_dim,
				dropout=self.dropout,
				act=self.act,
				use_layernorm=use_layernorm,
			)
			self.modality_attentions[modality_name] = AttentionLayer(out_dim, out_dim, dropout=self.dropout)
			self.projectors[modality_name] = MLPProjector(
				in_feat=out_dim,
				hidden_feat=hidden_dim,
				out_feat=projector_dim if projector_dim is not None else out_dim,
				dropout=self.dropout,
			)

		cross_out_dim = modality_dims["omics1"][1]
		self.atten_cross = AttentionLayer(cross_out_dim, cross_out_dim, dropout=self.dropout)

	def _build_input_dict(
		self,
		features_omics1,
		features_omics2,
		adj_spatial_omics1,
		adj_feature_omics1,
		adj_spatial_omics2,
		adj_feature_omics2,
		features_omics3=None,
		adj_spatial_omics3=None,
		adj_feature_omics3=None,
	):
		inputs = {
			"omics1": {
				"features": features_omics1,
				"adj_spatial": adj_spatial_omics1,
				"adj_feature": adj_feature_omics1,
			},
			"omics2": {
				"features": features_omics2,
				"adj_spatial": adj_spatial_omics2,
				"adj_feature": adj_feature_omics2,
			},
		}
		if "omics3" in self.modality_names:
			inputs["omics3"] = {
				"features": features_omics3,
				"adj_spatial": adj_spatial_omics3,
				"adj_feature": adj_feature_omics3,
			}
		return inputs

	def forward(
		self,
		features_omics1,
		features_omics2,
		adj_spatial_omics1,
		adj_feature_omics1,
		adj_spatial_omics2,
		adj_feature_omics2,
		features_omics3=None,
		adj_spatial_omics3=None,
		adj_feature_omics3=None,
	):
		inputs = self._build_input_dict(
			features_omics1,
			features_omics2,
			adj_spatial_omics1,
			adj_feature_omics1,
			adj_spatial_omics2,
			adj_feature_omics2,
			features_omics3=features_omics3,
			adj_spatial_omics3=adj_spatial_omics3,
			adj_feature_omics3=adj_feature_omics3,
		)

		latent_spatial = {}
		latent_feature = {}
		latent_modality = {}
		projected_modality = {}
		alpha_modality = {}

		for modality_name in self.modality_names:
			features = inputs[modality_name]["features"]
			adj_spatial = inputs[modality_name]["adj_spatial"]
			adj_feature = inputs[modality_name]["adj_feature"]

			latent_spatial[modality_name] = self.spatial_encoders[modality_name](features, adj_spatial)
			latent_feature[modality_name] = self.feature_encoders[modality_name](features, adj_feature)
			latent_modality[modality_name], alpha_modality[modality_name] = self.modality_attentions[modality_name](
				latent_spatial[modality_name],
				latent_feature[modality_name],
			)
			projected_modality[modality_name] = self.projectors[modality_name](latent_modality[modality_name])

		emb_latent_combined, alpha_cross = self.atten_cross(
			*[latent_modality[modality_name] for modality_name in self.modality_names]
		)

		results = {
			"emb_latent_combined": emb_latent_combined,
			"alpha": alpha_cross,
		}

		for modality_name in self.modality_names:
			adj_spatial = inputs[modality_name]["adj_spatial"]
			results[f"emb_latent_spatial_{modality_name}"] = latent_spatial[modality_name]
			results[f"emb_latent_feature_{modality_name}"] = latent_feature[modality_name]
			results[f"emb_latent_{modality_name}"] = latent_modality[modality_name]
			results[f"emb_proj_{modality_name}"] = projected_modality[modality_name]
			results[f"alpha_{modality_name}"] = alpha_modality[modality_name]
			results[f"emb_recon_{modality_name}"] = self.decoders[modality_name](emb_latent_combined, adj_spatial)

		for modality_name in self.modality_names:
			cross_recons = []
			for other_modality_name in self.modality_names:
				if other_modality_name == modality_name:
					continue
				translated = self.spatial_encoders[other_modality_name](
					self.decoders[other_modality_name](
						latent_modality[modality_name],
						inputs[other_modality_name]["adj_spatial"],
					),
					inputs[other_modality_name]["adj_spatial"],
				)
				cross_recons.append(translated)

			if len(cross_recons) == 1:
				results[f"emb_latent_{modality_name}_across_recon"] = cross_recons[0]
			else:
				results[f"emb_latent_{modality_name}_across_recon"] = torch.stack(cross_recons, dim=0).mean(dim=0)

		return results


class AttentionLayer(Module):

	"""
	Attention layer supporting an arbitrary number of embeddings.
	"""

	def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
		super(AttentionLayer, self).__init__()
		self.in_feat = in_feat
		self.out_feat = out_feat
		self.dropout = float(dropout)

		self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
		self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

		self.reset_parameters()

	def reset_parameters(self):
		torch.nn.init.xavier_uniform_(self.w_omega)
		torch.nn.init.xavier_uniform_(self.u_omega)

	def forward(self, *embs):
		if len(embs) == 1 and isinstance(embs[0], (list, tuple)):
			embs = tuple(embs[0])
		self.emb = torch.stack(embs, dim=1)
		self.emb = F.dropout(self.emb, p=self.dropout, training=self.training)
		self.v = torch.tanh(torch.matmul(self.emb, self.w_omega))
		self.vu = torch.matmul(self.v, self.u_omega)
		self.alpha = F.softmax(torch.squeeze(self.vu, dim=-1) + 1e-6, dim=1)
		emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))
		return torch.squeeze(emb_combined, dim=-1), self.alpha
