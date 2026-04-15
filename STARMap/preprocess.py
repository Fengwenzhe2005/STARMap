import os
import sys
import shutil
import tempfile
import subprocess
import urllib.request
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from pathlib import Path
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from scipy.io import mmwrite
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph


def _candidate_rscript_paths():
	candidates = []

	for env_key in ("R_SCRIPT", "RSCRIPT", "R_SCRIPT_PATH"):
		value = os.environ.get(env_key)
		if value:
			candidates.append(value)

	which_rscript = shutil.which("Rscript")
	if which_rscript:
		candidates.append(which_rscript)

	prefixes = []
	for prefix in (os.environ.get("CONDA_PREFIX"), sys.prefix):
		if prefix:
			prefixes.append(Path(prefix))

	for prefix in prefixes:
		candidates.extend(
			[
				prefix / "Scripts" / "Rscript.exe",
				prefix / "Lib" / "R" / "bin" / "Rscript.exe",
				prefix / "Lib" / "R" / "bin" / "x64" / "Rscript.exe",
			]
		)

	seen = set()
	existing = []
	for candidate in candidates:
		candidate = str(candidate)
		if candidate not in seen and Path(candidate).exists():
			seen.add(candidate)
			existing.append(candidate)
	return existing


def _find_rscript():
	candidates = _candidate_rscript_paths()
	if candidates:
		return candidates[0]
	raise RuntimeError(
		"Rscript.exe was not found. Please make sure R is installed in the active conda environment "
		"or available on PATH."
	)


def _infer_r_env_root(rscript_path):
	rscript_path = Path(rscript_path).resolve()
	for parent in [rscript_path.parent, *rscript_path.parents]:
		if (parent / "Scripts").exists() and (parent / "Lib" / "R").exists():
			return parent
	if "lib\\r\\bin" in str(rscript_path).lower():
		return rscript_path.parents[4]
	return rscript_path.parent


def _build_r_env(rscript_path):
	env_root = _infer_r_env_root(rscript_path)
	env = os.environ.copy()

	r_home = env_root / "Lib" / "R"
	if r_home.exists():
		env["R_HOME"] = str(r_home)

	path_entries = [
		env_root,
		env_root / "Library" / "mingw-w64" / "bin",
		env_root / "Library" / "usr" / "bin",
		env_root / "Library" / "bin",
		env_root / "Scripts",
		env_root / "Lib" / "R" / "bin",
		env_root / "Lib" / "R" / "bin" / "x64",
		Path(rscript_path).parent,
	]
	path_entries = [str(path) for path in path_entries if Path(path).exists()]
	env["PATH"] = os.pathsep.join(path_entries + [env.get("PATH", "")])
	return env


def _run_rscript(expr=None, script_path=None, capture_output=False, check=True):
	rscript_path = _find_rscript()
	env = _build_r_env(rscript_path)
	cmd = [rscript_path]
	if expr is not None:
		cmd.extend(["-e", expr])
	elif script_path is not None:
		cmd.append(str(script_path))
	else:
		raise ValueError("Either expr or script_path must be provided.")

	return subprocess.run(
		cmd,
		env=env,
		check=check,
		capture_output=capture_output,
		text=True,
	)


def construct_neighbor_graph(
	adata_omics1,
	adata_omics2,
	adata_omics3=None,
	datatype="SPOTS",
	n_neighbors=3,
	spatial_graph_mode="adaptive_gaussian",
	bandwidth_neighbor=None,
):
	"""
	Construct neighbor graphs, including feature graph and spatial graph.
	Feature graph is based expression data while spatial graph is based on cell/spot spatial coordinates.

	Parameters
	----------
	n_neighbors : int
		Number of neighbors.
	spatial_graph_mode : str
		Spatial graph construction mode. Use 'adaptive_gaussian' to keep
		distance-aware edge weights, or 'connectivity' to recover the
		original binary graph.
	bandwidth_neighbor : int, optional
		Which neighbor distance to use as the local Gaussian bandwidth.
		Defaults to n_neighbors when None.

	Returns
	-------
	data : dict
		AnnData objects with preprossed data for different omics.

	"""

	if datatype in ["Stereo-CITE-seq", "Spatial-epigenome-transcriptome"]:
		n_neighbors = 6

	cell_position_omics1 = adata_omics1.obsm["spatial"]
	adj_omics1 = construct_graph_by_coordinate(
		cell_position_omics1,
		n_neighbors=n_neighbors,
		graph_mode=spatial_graph_mode,
		bandwidth_neighbor=bandwidth_neighbor,
	)
	adata_omics1.uns["adj_spatial"] = adj_omics1

	cell_position_omics2 = adata_omics2.obsm["spatial"]
	adj_omics2 = construct_graph_by_coordinate(
		cell_position_omics2,
		n_neighbors=n_neighbors,
		graph_mode=spatial_graph_mode,
		bandwidth_neighbor=bandwidth_neighbor,
	)
	adata_omics2.uns["adj_spatial"] = adj_omics2

	if adata_omics3 is not None:
		cell_position_omics3 = adata_omics3.obsm["spatial"]
		adj_omics3 = construct_graph_by_coordinate(
			cell_position_omics3,
			n_neighbors=n_neighbors,
			graph_mode=spatial_graph_mode,
			bandwidth_neighbor=bandwidth_neighbor,
		)
		adata_omics3.uns["adj_spatial"] = adj_omics3
		feature_graph_omics1, feature_graph_omics2, feature_graph_omics3 = construct_graph_by_feature(
			adata_omics1,
			adata_omics2,
			adata_omics3,
		)
		adata_omics1.obsm["adj_feature"] = feature_graph_omics1
		adata_omics2.obsm["adj_feature"] = feature_graph_omics2
		adata_omics3.obsm["adj_feature"] = feature_graph_omics3
	else:
		feature_graph_omics1, feature_graph_omics2 = construct_graph_by_feature(adata_omics1, adata_omics2)
		adata_omics1.obsm["adj_feature"], adata_omics2.obsm["adj_feature"] = feature_graph_omics1, feature_graph_omics2

	data = {"adata_omics1": adata_omics1, "adata_omics2": adata_omics2}
	if adata_omics3 is not None:
		data["adata_omics3"] = adata_omics3

	return data


def pca(adata, use_reps=None, n_comps=10):

	"""Dimension reduction with PCA algorithm"""

	from sklearn.decomposition import PCA
	from scipy.sparse.csc import csc_matrix
	from scipy.sparse.csr import csr_matrix

	pca = PCA(n_components=n_comps)
	if use_reps is not None:
		feat_pca = pca.fit_transform(adata.obsm[use_reps])
	else:
		if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
			feat_pca = pca.fit_transform(adata.X.toarray())
		else:
			feat_pca = pca.fit_transform(adata.X)

	return feat_pca


def clr_normalize_each_cell(adata, inplace=True):

	"""Normalize count vector for each cell, i.e. for each row of .X"""

	import numpy as np
	import scipy

	def seurat_clr(x):
		s = np.sum(np.log1p(x[x > 0]))
		exp = np.exp(s / len(x))
		return np.log1p(x / exp)

	if not inplace:
		adata = adata.copy()

	x = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.asarray(adata.X)
	adata.X = np.apply_along_axis(
		seurat_clr,
		1,
		x,
	)
	return adata


def _compute_moran_i_scores(expr, adjacency):
	"""Compute Moran's I score for each gene."""

	n_obs = expr.shape[0]
	row_sums = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)
	total_weight = row_sums.sum()
	if total_weight <= 0:
		raise ValueError("Spatial adjacency has zero total weight.")

	if sp.issparse(expr):
		expr = expr.tocsr()
		sum_x = np.asarray(expr.sum(axis=0)).ravel().astype(np.float64)
		sumsq_x = np.asarray(expr.power(2).sum(axis=0)).ravel().astype(np.float64)
		weighted_sum_x = np.asarray(expr.T.dot(row_sums)).ravel().astype(np.float64)
		wx = adjacency.dot(expr)
		cross_term = np.asarray(expr.multiply(wx).sum(axis=0)).ravel().astype(np.float64)
	else:
		expr = np.asarray(expr, dtype=np.float64)
		sum_x = expr.sum(axis=0)
		sumsq_x = np.square(expr).sum(axis=0)
		weighted_sum_x = expr.T.dot(row_sums)
		wx = adjacency.dot(expr)
		cross_term = np.sum(expr * wx, axis=0)

	mean_x = sum_x / float(n_obs)
	numerator = cross_term - 2.0 * mean_x * weighted_sum_x + (mean_x ** 2) * total_weight
	denominator = sumsq_x - float(n_obs) * (mean_x ** 2)

	moran_i = np.full(expr.shape[1], -np.inf, dtype=np.float64)
	valid = denominator > 0
	moran_i[valid] = (float(n_obs) / total_weight) * numerator[valid] / denominator[valid]
	return moran_i


def _ensure_r_package(pkg_name):
	_run_rscript(
		expr=f"if(!requireNamespace('{pkg_name}', quietly=TRUE)) install.packages('{pkg_name}', repos='https://cloud.r-project.org')"
	)


def _ensure_r_packages(pkg_names):
	pkg_names = [pkg for pkg in pkg_names if pkg]
	if not pkg_names:
		return
	pkgs_literal = ", ".join([f'"{pkg}"' for pkg in pkg_names])
	expr = (
		f"pk <- c({pkgs_literal}); "
		"miss <- pk[!pk %in% rownames(installed.packages())]; "
		"if(length(miss)) install.packages(miss, repos='https://cloud.r-project.org')"
	)
	_run_rscript(expr=expr)


def _ensure_spark_package():
	try:
		_run_rscript(expr="library(SPARK);cat('SPARK ok\\n')")
		return
	except Exception:
		pass

	_ensure_r_packages(["Rcpp", "foreach", "doParallel", "Matrix", "CompQuadForm", "matlab", "pracma", "RcppArmadillo"])

	with tempfile.TemporaryDirectory(prefix="spark_install_") as tmpdir:
		tmpdir = Path(tmpdir)
		tarball_path = tmpdir / "SPARK_master.tar.gz"
		try:
			urllib.request.urlretrieve(
				"https://github.com/xzhoulab/SPARK/archive/refs/heads/master.tar.gz",
				tarball_path,
			)
		except Exception as exc:
			raise RuntimeError("Failed to download the official SPARK package source tarball from GitHub.") from exc

		try:
			_run_rscript(
				expr=f"install.packages('{tarball_path.as_posix()}', repos=NULL, type='source')",
				capture_output=True,
			)
		except subprocess.CalledProcessError as exc:
			output = (exc.stdout or "") + "\n" + (exc.stderr or "")
			if "'make' not found" in output or "make not found" in output.lower():
				raise RuntimeError(
					"Installing SPARK from source requires Windows build tools, but `make` was not found. "
					"Please install the official Rtools44 toolchain for R 4.4.x from "
					"https://cran.r-project.org/bin/windows/Rtools/rtools44/rtools.html "
					"and restart the notebook."
				) from exc
			raise RuntimeError(f"Failed to install SPARK from source.\n{output}") from exc

	try:
		_run_rscript(expr="library(SPARK);cat('SPARK ok\\n')")
	except subprocess.CalledProcessError as exc:
		output = (exc.stdout or "") + "\n" + (exc.stderr or "")
		raise RuntimeError(f"SPARK installation completed but the package still could not be loaded.\n{output}") from exc


def _run_sparkx(
	adata,
	spatial_key="spatial",
	option="mixture",
	num_cores=1,
	install_if_missing=True,
	verbose=True,
):
	"""
	Run SPARK-X in R and return the per-gene testing table.
	"""

	if spatial_key not in adata.obsm:
		raise KeyError(f"{spatial_key!r} is not found in adata.obsm.")

	if install_if_missing:
		_ensure_spark_package()

	with tempfile.TemporaryDirectory(prefix="starmap_sparkx_") as tmpdir:
		tmpdir = Path(tmpdir)
		count_path = tmpdir / "counts.mtx"
		gene_path = tmpdir / "genes.tsv"
		spot_path = tmpdir / "spots.tsv"
		loc_path = tmpdir / "locations.csv"
		out_path = tmpdir / "sparkx_results.csv"
		script_path = tmpdir / "run_sparkx.R"

		counts = adata.X.T.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(np.asarray(adata.X).T)
		mmwrite(str(count_path), counts)
		pd.Series(adata.var_names).to_csv(gene_path, index=False, header=False)
		pd.Series(adata.obs_names).to_csv(spot_path, index=False, header=False)

		loc_df = pd.DataFrame(
			np.asarray(adata.obsm[spatial_key], dtype=np.float64),
			index=adata.obs_names,
			columns=[f"coord_{i+1}" for i in range(np.asarray(adata.obsm[spatial_key]).shape[1])],
		)
		loc_df.to_csv(loc_path)

		verbose_flag = "TRUE" if verbose else "FALSE"
		r_script = f"""
		suppressPackageStartupMessages(library(Matrix))
		suppressPackageStartupMessages(library(SPARK))

		counts <- as(Matrix::readMM("{count_path.as_posix()}"), "dgCMatrix")
		genes <- readLines("{gene_path.as_posix()}")
		spots <- readLines("{spot_path.as_posix()}")
		loc <- as.matrix(read.csv("{loc_path.as_posix()}", row.names=1, check.names=FALSE))

		rownames(counts) <- genes
		colnames(counts) <- spots
		loc <- loc[spots, , drop=FALSE]

		res <- SPARK::sparkx(
			count_in=counts,
			locus_in=loc,
			numCores={int(num_cores)},
			option="{option}",
			verbose={verbose_flag}
		)

		out <- as.data.frame(res$res_mtest)
		out$gene <- rownames(out)
		write.csv(out, "{out_path.as_posix()}", row.names=FALSE)
		"""
		script_path.write_text(r_script, encoding="utf-8")

		try:
			_run_rscript(script_path=script_path)
		except subprocess.CalledProcessError as exc:
			output = (exc.stdout or "") + "\n" + (exc.stderr or "")
			raise RuntimeError(f"SPARK-X failed to run in R.\n{output}") from exc

		res = pd.read_csv(out_path)

	if "gene" not in res.columns:
		raise RuntimeError("SPARK-X output does not contain a gene column.")

	res = res.set_index("gene")
	res = res.reindex(adata.var_names)
	return res


def select_sparkx_svg_genes(
	adata,
	n_top_genes=3000,
	spatial_key="spatial",
	option="mixture",
	num_cores=1,
	rank_by="combinedPval",
	svg_key="spatially_variable",
	install_if_missing=True,
	verbose=True,
):
	"""
	Select SVGs using SPARK-X and return a boolean mask over ``adata.var_names``.
	"""

	res = _run_sparkx(
		adata,
		spatial_key=spatial_key,
		option=option,
		num_cores=num_cores,
		install_if_missing=install_if_missing,
		verbose=verbose,
	)

	if rank_by not in res.columns:
		raise KeyError(f"{rank_by!r} is not found in SPARK-X result columns: {list(res.columns)}")

	adata.var["sparkx_combinedPval"] = res["combinedPval"].to_numpy() if "combinedPval" in res.columns else np.nan
	adata.var["sparkx_adjustedPval"] = res["adjustedPval"].to_numpy() if "adjustedPval" in res.columns else np.nan

	rank_series = pd.to_numeric(res[rank_by], errors="coerce")
	adata.var[f"sparkx_{rank_by}"] = rank_series.to_numpy()

	svg_mask = np.zeros(adata.n_vars, dtype=bool)
	valid = rank_series.notna().to_numpy()
	valid_idx = np.where(valid)[0]
	if valid_idx.size > 0 and n_top_genes > 0:
		n_top_genes = min(int(n_top_genes), valid_idx.size)
		top_idx = valid_idx[np.argsort(rank_series.iloc[valid_idx].to_numpy())[:n_top_genes]]
		svg_mask[top_idx] = True

	adata.var[svg_key] = svg_mask
	print(f"SPARK-X SVG genes: {int(svg_mask.sum())}")
	return svg_mask


def select_hvg_svg_genes(
	adata,
	n_top_hvg=3000,
	n_top_svg=3000,
	min_cells=10,
	spatial_key="spatial",
	svg_neighbors=8,
	target_sum=1e4,
	hvg_flavor="seurat_v3",
	union_key="highly_variable_or_spatially_variable",
):
	"""
	Select genes using the union of HVGs and SVGs.

	Returns
	-------
	selected_mask : np.ndarray
		Boolean mask over ``adata.var_names`` for the HVG/SVG union gene set.
	"""

	if spatial_key not in adata.obsm:
		raise KeyError(f"{spatial_key!r} is not found in adata.obsm.")
	if svg_neighbors < 1:
		raise ValueError("svg_neighbors must be at least 1.")

	sc.pp.filter_genes(adata, min_cells=min_cells)

	n_top_hvg = min(int(n_top_hvg), adata.n_vars)
	n_top_svg = min(int(n_top_svg), adata.n_vars)

	sc.pp.highly_variable_genes(adata, flavor=hvg_flavor, n_top_genes=n_top_hvg)

	svg_adata = adata.copy()
	sc.pp.normalize_total(svg_adata, target_sum=target_sum)
	sc.pp.log1p(svg_adata)

	coords = np.asarray(svg_adata.obsm[spatial_key], dtype=np.float64)
	adjacency = kneighbors_graph(
		coords,
		n_neighbors=svg_neighbors,
		mode="connectivity",
		include_self=False,
	).tocsr()
	adjacency = adjacency.maximum(adjacency.T).astype(np.float64)

	moran_i = _compute_moran_i_scores(svg_adata.X, adjacency)
	adata.var["moran_i"] = moran_i

	svg_mask = np.zeros(adata.n_vars, dtype=bool)
	valid_idx = np.where(np.isfinite(moran_i))[0]
	if valid_idx.size > 0 and n_top_svg > 0:
		top_svg_idx = valid_idx[np.argsort(moran_i[valid_idx])[::-1][:n_top_svg]]
		svg_mask[top_svg_idx] = True

	adata.var["spatially_variable"] = svg_mask
	adata.var[union_key] = adata.var["highly_variable"].to_numpy() | svg_mask

	print(f"HVG genes: {int(adata.var['highly_variable'].sum())}")
	print(f"SVG genes: {int(adata.var['spatially_variable'].sum())}")
	print(f"HVG | SVG genes: {int(adata.var[union_key].sum())}")

	return adata.var[union_key].to_numpy()


def construct_graph_by_feature(*adatas, k=20, mode="connectivity", metric="correlation", include_self=False):

	"""Construct feature neighbor graphs using cosine similarity by default."""


	feature_graphs = [
		kneighbors_graph(adata.obsm["feat"], k, mode=mode, metric=metric, include_self=include_self)
		for adata in adatas
	]

	return tuple(feature_graphs)


def construct_graph_by_coordinate(
	cell_position,
	n_neighbors=3,
	graph_mode="adaptive_gaussian",
	bandwidth_neighbor=None,
	eps=1e-12,
):
	"""Constructing spatial neighbor graph according to spatial coordinates."""

	if n_neighbors < 1:
		raise ValueError("n_neighbors must be at least 1.")
	nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
	distances, indices = nbrs.kneighbors(cell_position)

	neighbor_indices = indices[:, 1:]
	neighbor_distances = distances[:, 1:]
	x = np.arange(cell_position.shape[0]).repeat(n_neighbors)
	y = neighbor_indices.flatten()

	if graph_mode == "connectivity":
		values = np.ones(x.size, dtype=np.float32)
	elif graph_mode == "adaptive_gaussian":
		if bandwidth_neighbor is None:
			bandwidth_neighbor = n_neighbors
		if bandwidth_neighbor < 1 or bandwidth_neighbor > n_neighbors:
			raise ValueError("bandwidth_neighbor must be between 1 and n_neighbors.")

		sigma = distances[:, bandwidth_neighbor].astype(np.float32)
		sigma = np.maximum(sigma, eps)
		sigma_i = sigma[:, None]
		sigma_j = sigma[neighbor_indices]
		values = np.exp(-(neighbor_distances**2) / (sigma_i * sigma_j + eps)).astype(np.float32).flatten()
	else:
		raise ValueError(f"Unsupported graph_mode: {graph_mode}")

	adj = pd.DataFrame(columns=["x", "y", "value"])
	adj["x"] = x
	adj["y"] = y
	adj["value"] = values
	return adj


def transform_adjacent_matrix(adjacent):
	n_spot = adjacent["x"].max() + 1
	adj = coo_matrix((adjacent["value"], (adjacent["x"], adjacent["y"])), shape=(n_spot, n_spot))
	return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):

	"""Convert a scipy sparse matrix to a torch sparse tensor."""

	sparse_mx = sparse_mx.tocoo().astype(np.float32)
	indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
	values = torch.from_numpy(sparse_mx.data)
	shape = torch.Size(sparse_mx.shape)
	return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_graph(adj):
	adj = sp.coo_matrix(adj)
	adj_ = adj + sp.eye(adj.shape[0])
	rowsum = np.array(adj_.sum(1))
	degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
	adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
	return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def adjacent_matrix_preprocessing(adata_omics1, adata_omics2, adata_omics3=None):
	"""Converting dense adjacent matrix to sparse adjacent matrix"""

	adatas = [adata_omics1, adata_omics2]
	if adata_omics3 is not None:
		adatas.append(adata_omics3)

	adj = {}
	for idx, adata in enumerate(adatas, start=1):
		adj_spatial = transform_adjacent_matrix(adata.uns["adj_spatial"]).tocsr()
		adj_spatial = adj_spatial.maximum(adj_spatial.T)
		adj_spatial = preprocess_graph(adj_spatial)

		adj_feature = adata.obsm["adj_feature"].copy().tocsr()
		adj_feature = adj_feature.maximum(adj_feature.T)
		adj_feature = preprocess_graph(adj_feature)

		adj[f"adj_spatial_omics{idx}"] = adj_spatial
		adj[f"adj_feature_omics{idx}"] = adj_feature

	return adj


def lsi(adata: anndata.AnnData, n_components: int = 20, use_highly_variable: Optional[bool] = None, **kwargs) -> None:
	r"""
	LSI analysis (following the Seurat v3 approach)
	"""
	if use_highly_variable is None:
		use_highly_variable = "highly_variable" in adata.var
	adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
	X = tfidf(adata_use.X)
	X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
	X_norm = np.log1p(X_norm * 1e4)
	X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
	X_lsi -= X_lsi.mean(axis=1, keepdims=True)
	X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
	adata.obsm["X_lsi"] = X_lsi[:, 1:]


def tfidf(X):
	r"""
	TF-IDF normalization (following the Seurat v3 approach)
	"""
	idf = X.shape[0] / X.sum(axis=0)
	if scipy.sparse.issparse(X):
		tf = X.multiply(1 / X.sum(axis=1))
		return tf.multiply(idf)
	tf = X / X.sum(axis=1, keepdims=True)
	return tf * idf


def fix_seed(seed):
	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	cudnn.deterministic = True
	cudnn.benchmark = False

	os.environ["PYTHONHASHSEED"] = str(seed)
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
