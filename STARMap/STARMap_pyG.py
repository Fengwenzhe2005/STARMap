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
			modality_name: self.adj[f"adj_spatial_{modality_name}"].to(self.device)
			for modality_name in self.modality_names
		}
		self.adj_feature = {
			modality_name: self.adj[f"adj_feature_{modality_name}"].to(self.device)
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
			default_weight_factors = [default_weight_factors[0], default_weight_factors[1], default_weight_factors[0], default_weight_factors[2], default_weight_factors[3], default_weight_factors[2]]

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

	def train(self):
		model_kwargs = {
			"dim_in_feat_omics1": self.dim_inputs["omics1"],
			"dim_out_feat_omics1": self.dim_output,
			"dim_in_feat_omics2": self.dim_inputs["omics2"],
			"dim_out_feat_omics2": self.dim_output,
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
		for epoch in tqdm(range(self.epochs)):
			self.model.train()
			results = self._run_model()

			recon_losses = []
			corr_losses = []
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

			loss = sum(recon_losses) + sum(corr_losses)

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

		print("Model training finished!\n")

		with torch.no_grad():
			self.model.eval()
			results = self._run_model()

		output = {
			"STARMap": F.normalize(results["emb_latent_combined"], p=2, eps=1e-12, dim=1).detach().cpu().numpy(),
			"alpha": results["alpha"].detach().cpu().numpy(),
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
