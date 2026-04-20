import itertools

import torch
from tqdm import tqdm
import torch.nn.functional as F

from .model import Encoder_overall
from .preprocess import adjacent_matrix_preprocessing


class Train_STARMap:
	def __init__(
		self,
		data,
		datatype="SPOTS",
		device=torch.device("cpu"),
		random_seed=2022,
		learning_rate=0.0001,
		weight_decay=0.00,
		epochs=None,
		dim_input=3000,
		dim_output=64,
		weight_factors=None,
		dropout=0.1,
		hidden_dim=None,
		projector_dim=None,
		use_layernorm=True,
		contrastive_weight=0.2,
		contrastive_temperature=0.2,
		contrastive_sample_size=2048,
		spatial_smoothness_weight=0.05,
		feature_consistency_weight=0.1,
	):
		"""\
		Train STARMap with two or three modalities.
		"""
		self.data = data.copy()
		self.datatype = datatype
		self.device = device
		self.random_seed = random_seed
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.epochs = int(epochs) if epochs is not None else None
		self.dim_input = dim_input
		self.dim_output = dim_output
		self.dropout = float(dropout)
		self.hidden_dim = hidden_dim
		self.projector_dim = projector_dim
		self.use_layernorm = use_layernorm
		self.contrastive_weight = float(contrastive_weight)
		self.contrastive_temperature = float(contrastive_temperature)
		self.contrastive_sample_size = None if contrastive_sample_size is None else int(contrastive_sample_size)
		self.spatial_smoothness_weight = float(spatial_smoothness_weight)
		self.feature_consistency_weight = float(feature_consistency_weight)

		self.modality_names = ["omics1", "omics2"]
		if "adata_omics3" in self.data:
			self.modality_names.append("omics3")

		self.adata_modalities = {
			modality_name: self.data[f"adata_{modality_name}"]
			for modality_name in self.modality_names
		}

		self.adj = adjacent_matrix_preprocessing(
			self.adata_modalities["omics1"],
			self.adata_modalities["omics2"],
			self.adata_modalities.get("omics3"),
		)

		self.adj_spatial = {
			modality_name: self.adj[f"adj_spatial_{modality_name}"].coalesce().to(self.device)
			for modality_name in self.modality_names
		}
		self.adj_feature = {
			modality_name: self.adj[f"adj_feature_{modality_name}"].coalesce().to(self.device)
			for modality_name in self.modality_names
		}
		self.features = {
			modality_name: torch.FloatTensor(self.adata_modalities[modality_name].obsm["feat"].copy()).to(self.device)
			for modality_name in self.modality_names
		}

		self.dim_inputs = {
			modality_name: self.features[modality_name].shape[1]
			for modality_name in self.modality_names
		}

		if self.datatype == "SPOTS":
			default_epochs = 600
			default_weight_factors = [1, 5, 1, 1]
		elif self.datatype == "Stereo-CITE-seq":
			default_epochs = 1500
			default_weight_factors = [1, 10, 1, 10]
		elif self.datatype == "10x":
			default_epochs = 200
			default_weight_factors = [1, 5, 1, 10]
		elif self.datatype == "Spatial-epigenome-transcriptome":
			default_epochs = 1600
			default_weight_factors = [1, 5, 1, 1]
		else:
			default_epochs = 600
			default_weight_factors = [1, 5, 1, 1]

		if self.epochs is None:
			self.epochs = default_epochs

		if len(self.modality_names) == 3:
			default_weight_factors = [
				default_weight_factors[0],
				default_weight_factors[1],
				default_weight_factors[0],
				default_weight_factors[2],
				default_weight_factors[3],
				default_weight_factors[2],
			]

		self.weight_factors = weight_factors if weight_factors is not None else default_weight_factors

	def _run_model(self):
		model_kwargs = {
			"features_omics1": self.features["omics1"],
			"features_omics2": self.features["omics2"],
			"adj_spatial_omics1": self.adj_spatial["omics1"],
			"adj_feature_omics1": self.adj_feature["omics1"],
			"adj_spatial_omics2": self.adj_spatial["omics2"],
			"adj_feature_omics2": self.adj_feature["omics2"],
		}
		if "omics3" in self.modality_names:
			model_kwargs.update(
				{
					"features_omics3": self.features["omics3"],
					"adj_spatial_omics3": self.adj_spatial["omics3"],
					"adj_feature_omics3": self.adj_feature["omics3"],
				}
			)
		return self.model(**model_kwargs)

	def _sparse_dirichlet_energy(self, emb, adj):
		row, col = adj.indices()
		weights = adj.values()
		diff = emb[row] - emb[col]
		edge_energy = weights.unsqueeze(-1) * diff.pow(2)
		return edge_energy.sum(dim=1).mean()

	def _modality_contrastive_loss(self, projections):
		if len(projections) < 2 or self.contrastive_weight <= 0:
			return projections[0].new_tensor(0.0)

		num_obs = projections[0].shape[0]
		if self.contrastive_sample_size is not None and num_obs > self.contrastive_sample_size:
			sample_idx = torch.randperm(num_obs, device=projections[0].device)[: self.contrastive_sample_size]
			projections = [projection[sample_idx] for projection in projections]

		pair_losses = []
		for proj_a, proj_b in itertools.combinations(projections, 2):
			z_a = F.normalize(proj_a, p=2, dim=1)
			z_b = F.normalize(proj_b, p=2, dim=1)
			logits_ab = torch.mm(z_a, z_b.t()) / self.contrastive_temperature
			logits_ba = torch.mm(z_b, z_a.t()) / self.contrastive_temperature
			target = torch.arange(z_a.shape[0], device=z_a.device)
			loss_ab = F.cross_entropy(logits_ab, target)
			loss_ba = F.cross_entropy(logits_ba, target)
			pair_losses.append(0.5 * (loss_ab + loss_ba))

		return torch.stack(pair_losses).mean()

	def _compute_losses(self, results):
		recon_losses = []
		corr_losses = []
		feature_consistency_losses = []
		spatial_smoothness_losses = []
		projections = []

		for modality_idx, modality_name in enumerate(self.modality_names):
			recon_losses.append(
				self.weight_factors[modality_idx] * F.mse_loss(
					self.features[modality_name],
					results[f"emb_recon_{modality_name}"],
				)
			)
			corr_losses.append(
				self.weight_factors[len(self.modality_names) + modality_idx] * F.mse_loss(
					results[f"emb_latent_{modality_name}"],
					results[f"emb_latent_{modality_name}_across_recon"],
				)
			)
			feature_consistency_losses.append(
				F.mse_loss(
					results[f"emb_latent_spatial_{modality_name}"],
					results[f"emb_latent_feature_{modality_name}"],
				)
			)
			spatial_smoothness_losses.append(
				self._sparse_dirichlet_energy(
					results[f"emb_latent_{modality_name}"],
					self.adj_spatial[modality_name],
				)
			)
			projections.append(results[f"emb_proj_{modality_name}"])

		recon_loss = sum(recon_losses)
		corr_loss = sum(corr_losses)
		feature_consistency_loss = torch.stack(feature_consistency_losses).mean()
		spatial_smoothness_loss = torch.stack(spatial_smoothness_losses).mean()
		contrastive_loss = self._modality_contrastive_loss(projections)

		total_loss = (
			recon_loss
			+ corr_loss
			+ self.feature_consistency_weight * feature_consistency_loss
			+ self.spatial_smoothness_weight * spatial_smoothness_loss
			+ self.contrastive_weight * contrastive_loss
		)

		loss_dict = {
			"loss": total_loss,
			"recon_loss": recon_loss,
			"cross_recon_loss": corr_loss,
			"feature_consistency_loss": feature_consistency_loss,
			"spatial_smoothness_loss": spatial_smoothness_loss,
			"contrastive_loss": contrastive_loss,
		}
		return loss_dict

	def train(self):
		model_kwargs = {
			"dim_in_feat_omics1": self.dim_inputs["omics1"],
			"dim_out_feat_omics1": self.dim_output,
			"dim_in_feat_omics2": self.dim_inputs["omics2"],
			"dim_out_feat_omics2": self.dim_output,
			"dropout": self.dropout,
			"hidden_dim": self.hidden_dim,
			"projector_dim": self.projector_dim,
			"use_layernorm": self.use_layernorm,
		}
		if "omics3" in self.modality_names:
			model_kwargs.update(
				{
					"dim_in_feat_omics3": self.dim_inputs["omics3"],
					"dim_out_feat_omics3": self.dim_output,
				}
			)

		self.model = Encoder_overall(**model_kwargs).to(self.device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)
		self.model.train()

		loss_history = []
		loss_components_history = {
			"recon_loss": [],
			"cross_recon_loss": [],
			"feature_consistency_loss": [],
			"spatial_smoothness_loss": [],
			"contrastive_loss": [],
		}

		for epoch in tqdm(range(self.epochs)):
			self.model.train()
			self.optimizer.zero_grad()
			results = self._run_model()
			loss_dict = self._compute_losses(results)

			loss_history.append(loss_dict["loss"].item())
			for loss_name in loss_components_history:
				loss_components_history[loss_name].append(loss_dict[loss_name].item())

			loss_dict["loss"].backward()
			self.optimizer.step()

		print("Model training finished!\n")

		with torch.no_grad():
			self.model.eval()
			results = self._run_model()
			final_loss_dict = self._compute_losses(results)

		output = {
			"STARMap": F.normalize(results["emb_latent_combined"], p=2, eps=1e-12, dim=1).detach().cpu().numpy(),
			"alpha": results["alpha"].detach().cpu().numpy(),
			"loss_history": loss_history,
			"loss_components_history": loss_components_history,
			"final_losses": {key: value.item() for key, value in final_loss_dict.items()},
		}
		for modality_name in self.modality_names:
			output[f"emb_latent_{modality_name}"] = F.normalize(
				results[f"emb_latent_{modality_name}"],
				p=2,
				eps=1e-12,
				dim=1,
			).detach().cpu().numpy()
			output[f"alpha_{modality_name}"] = results[f"alpha_{modality_name}"].detach().cpu().numpy()

		return output


__all__ = ["Train_STARMap"]
