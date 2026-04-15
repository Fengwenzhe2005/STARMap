import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


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
		dropout=0.0,
		act=F.relu,
	):
		super(Encoder_overall, self).__init__()
		self.dropout = dropout
		self.act = act

		self.modality_names = ["omics1", "omics2"]
		modality_dims = {
			"omics1": (dim_in_feat_omics1, dim_out_feat_omics1),
			"omics2": (dim_in_feat_omics2, dim_out_feat_omics2),
		}
		if dim_in_feat_omics3 is not None and dim_out_feat_omics3 is not None:
			self.modality_names.append("omics3")
			modality_dims["omics3"] = (dim_in_feat_omics3, dim_out_feat_omics3)

		self.encoders = nn.ModuleDict()
		self.decoders = nn.ModuleDict()
		self.modality_attentions = nn.ModuleDict()
		for modality_name in self.modality_names:
			in_dim, out_dim = modality_dims[modality_name]
			self.encoders[modality_name] = Encoder(in_dim, out_dim)
			self.decoders[modality_name] = Decoder(out_dim, in_dim)
			self.modality_attentions[modality_name] = AttentionLayer(out_dim, out_dim)

		cross_out_dim = modality_dims["omics1"][1]
		self.atten_cross = AttentionLayer(cross_out_dim, cross_out_dim)

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
		alpha_modality = {}

		for modality_name in self.modality_names:
			features = inputs[modality_name]["features"]
			adj_spatial = inputs[modality_name]["adj_spatial"]
			adj_feature = inputs[modality_name]["adj_feature"]

			latent_spatial[modality_name] = self.encoders[modality_name](features, adj_spatial)
			latent_feature[modality_name] = self.encoders[modality_name](features, adj_feature)
			latent_modality[modality_name], alpha_modality[modality_name] = self.modality_attentions[modality_name](
				latent_spatial[modality_name],
				latent_feature[modality_name],
			)

		emb_latent_combined, alpha_cross = self.atten_cross(
			*[latent_modality[modality_name] for modality_name in self.modality_names]
		)

		results = {
			"emb_latent_combined": emb_latent_combined,
			"alpha": alpha_cross,
		}

		for modality_name in self.modality_names:
			adj_spatial = inputs[modality_name]["adj_spatial"]
			results[f"emb_latent_{modality_name}"] = latent_modality[modality_name]
			results[f"alpha_{modality_name}"] = alpha_modality[modality_name]
			results[f"emb_recon_{modality_name}"] = self.decoders[modality_name](emb_latent_combined, adj_spatial)

		for modality_name in self.modality_names:
			cross_recons = []
			for other_modality_name in self.modality_names:
				if other_modality_name == modality_name:
					continue
				translated = self.encoders[other_modality_name](
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


class Encoder(Module):

	"""\
	Modality-specific GNN encoder.
	"""

	def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
		super(Encoder, self).__init__()
		self.in_feat = in_feat
		self.out_feat = out_feat
		self.dropout = dropout
		self.act = act

		self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

		self.reset_parameters()

	def reset_parameters(self):
		torch.nn.init.xavier_uniform_(self.weight)

	def forward(self, feat, adj):
		x = torch.mm(feat, self.weight)
		x = torch.spmm(adj, x)
		return x


class Decoder(Module):

	"""\
	Modality-specific GNN decoder.
	"""

	def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
		super(Decoder, self).__init__()
		self.in_feat = in_feat
		self.out_feat = out_feat
		self.dropout = dropout
		self.act = act

		self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

		self.reset_parameters()

	def reset_parameters(self):
		torch.nn.init.xavier_uniform_(self.weight)

	def forward(self, feat, adj):
		x = torch.mm(feat, self.weight)
		x = torch.spmm(adj, x)
		return x


class AttentionLayer(Module):

	"""\
	Attention layer supporting an arbitrary number of embeddings.
	"""

	def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
		super(AttentionLayer, self).__init__()
		self.in_feat = in_feat
		self.out_feat = out_feat

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
		self.v = torch.tanh(torch.matmul(self.emb, self.w_omega))
		self.vu = torch.matmul(self.v, self.u_omega)
		self.alpha = F.softmax(torch.squeeze(self.vu, dim=-1) + 1e-6, dim=1)
		emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))
		return torch.squeeze(emb_combined, dim=-1), self.alpha
