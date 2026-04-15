import os
import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AffinityPropagation, DBSCAN, KMeans, SpectralClustering
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import subprocess
import warnings


def _import_robjects(raise_error=True):
    try:
        import rpy2.robjects as robjects
        return robjects
    except Exception as exc:
        msg = (
            "rpy2 or R runtime is unavailable. "
            "Please install R, ensure Rscript is on PATH, and install rpy2."
        )
        if raise_error:
            raise RuntimeError(msg) from exc
        warnings.warn(msg)
        return None


def detect_r_home(raise_error=False):
    cmds = [
        ["Rscript", "-e", "cat(R.home())"],
        ["R", "-s", "-q", "-e", "cat(R.home())"],
    ]
    last_err = None
    for cmd in cmds:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=15)
            r_home = out.decode('utf-8', errors='ignore').strip()
            if r_home:
                os.environ['R_HOME'] = r_home
                print('R_HOME set to:', r_home)
                return r_home
        except Exception as exc:
            last_err = exc
    msg = f'Detect R_HOME failed, last error: {last_err}'
    if raise_error:
        raise RuntimeError(msg)
    warnings.warn(msg)
    return None

def ensure_r_pkg(pkg_name):
    cmd = ["Rscript", "-e", f"if(!requireNamespace('{pkg_name}', quietly=TRUE)) install.packages('{pkg_name}', repos='https://cloud.r-project.org')"]
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError as exc:
        raise RuntimeError("Rscript is not found in PATH.") from exc


def check_mclust(install_if_missing=True, raise_error=False):
    try:
        subprocess.check_call(["Rscript", "-e", "library(mclust);cat('mclust ok\n')"])
        return True
    except FileNotFoundError as exc:
        msg = "Rscript is not found in PATH. mclust is unavailable."
        if raise_error:
            raise RuntimeError(msg) from exc
        warnings.warn(msg)
        return False
    except subprocess.CalledProcessError as exc:
        if install_if_missing:
            try:
                ensure_r_pkg('mclust')
                subprocess.check_call(["Rscript", "-e", "library(mclust);cat('mclust ok\n')"])
                return True
            except Exception as install_exc:
                msg = f"mclust is unavailable: {install_exc}"
                if raise_error:
                    raise RuntimeError(msg) from install_exc
                warnings.warn(msg)
                return False
        msg = f"mclust is unavailable: {exc}"
        if raise_error:
            raise RuntimeError(msg) from exc
        warnings.warn(msg)
        return False


def _to_numpy_array(data, dtype=np.float64):
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    return np.asarray(data, dtype=dtype)


def align_adata_to_obs_names(adata, target_obs_names):
    """
    Subset and reorder ``adata`` so its observations follow ``target_obs_names``.
    """
    target_obs = pd.Index(target_obs_names)
    keep_mask = adata.obs_names.isin(target_obs)
    aligned = adata[np.asarray(keep_mask, dtype=bool)].copy()

    order_map = pd.Series(np.arange(target_obs.size), index=target_obs)
    aligned.obs["__order__"] = order_map.reindex(aligned.obs_names).to_numpy()
    if aligned.obs["__order__"].isna().any():
        missing_obs = aligned.obs_names[aligned.obs["__order__"].isna()].astype(str).tolist()
        raise ValueError(
            "Failed to align AnnData observations because some obs_names were not found "
            f"in target_obs_names: {missing_obs[:5]}"
        )

    aligned = aligned[np.argsort(aligned.obs["__order__"].to_numpy())].copy()
    aligned.obs = aligned.obs.drop(columns=["__order__"])
    return aligned


def build_embedding_adata(embedding, reference_adata, var_prefix="embedding", spatial_key="spatial"):
    """
    Build an ``AnnData`` view for a precomputed embedding aligned to ``reference_adata``.
    """
    embedding = _to_numpy_array(embedding, dtype=np.float64)
    if embedding.ndim != 2:
        raise ValueError(f"embedding must be 2-dimensional, got shape {embedding.shape}.")
    if embedding.shape[0] != reference_adata.n_obs:
        raise ValueError(
            f"Embedding row count ({embedding.shape[0]}) does not match "
            f"reference_adata.n_obs ({reference_adata.n_obs})."
        )

    embedding_adata = ad.AnnData(
        X=embedding,
        obs=reference_adata.obs.copy(),
        var=pd.DataFrame(index=[f"{var_prefix}_{idx}" for idx in range(embedding.shape[1])]),
    )
    if spatial_key in reference_adata.obsm:
        embedding_adata.obsm[spatial_key] = np.asarray(reference_adata.obsm[spatial_key]).copy()
    return embedding_adata


def attach_obsm_from_dict(adata, values, keys=None, copy=True):
    """
    Copy selected arrays from a dictionary-like object into ``adata.obsm``.
    """
    keys = list(values.keys()) if keys is None else list(keys)
    missing_keys = [key for key in keys if key not in values]
    if missing_keys:
        raise KeyError(f"Missing keys in values: {missing_keys}")

    attached_keys = []
    for key in keys:
        value = values[key]
        if copy:
            if hasattr(value, "detach") or hasattr(value, "cpu"):
                stored_value = _to_numpy_array(value)
            elif hasattr(value, "copy"):
                stored_value = value.copy()
            else:
                stored_value = np.array(value, copy=True)
        else:
            stored_value = value
        adata.obsm[key] = stored_value
        attached_keys.append(key)

    return attached_keys


def build_embedding_pca(adata, source_key, target_key=None, n_comps=20, random_seed=2022, verbose=True):
    """
    Build a PCA representation from an embedding stored in ``adata.obsm``.
    """
    if source_key not in adata.obsm:
        raise KeyError(f"{source_key!r} is not found in adata.obsm.")

    from sklearn.decomposition import PCA

    target_key = target_key or f"{source_key}_pca"
    data = _to_numpy_array(adata.obsm[source_key], dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"adata.obsm[{source_key!r}] must be 2-dimensional, got shape {data.shape}.")

    max_n_comps = min(data.shape[0], data.shape[1])
    if max_n_comps < 1:
        raise ValueError(f"adata.obsm[{source_key!r}] has invalid shape {data.shape} for PCA.")

    n_comps = min(n_comps, max_n_comps)
    adata.obsm[target_key] = PCA(n_components=n_comps, random_state=random_seed).fit_transform(data)
    if verbose:
        print(f"{target_key} shape:", adata.obsm[target_key].shape)
    return target_key


def add_ground_truth_labels(adata, label_path, label_key="ground_truth"):
    """
    Load line-delimited ground-truth labels and attach them to ``adata.obs``.
    """
    label_path = Path(label_path)
    with label_path.open("r", encoding="utf-8") as handle:
        labels = [line.strip() for line in handle if line.strip()]

    if len(labels) != adata.n_obs:
        raise ValueError(
            f"Ground-truth label count ({len(labels)}) does not match adata.n_obs ({adata.n_obs})."
        )

    adata.obs[label_key] = pd.Categorical(labels)
    return labels


def sort_mixed_labels(labels):
    """
    Sort labels so digit-only strings come first in numeric order, then others lexicographically.
    """
    return _sorted_mixed_labels(labels)


def prepare_starmap_modalities(
    adata_omics1,
    adata_omics2,
    adata_omics3=None,
    n_top_hvg=2000,
    n_top_svg=100,
    min_cells=10,
    spatial_key="spatial",
    svg_neighbors=6,
    feat_n_comps=None,
):
    """
    Select HVG/SVG genes and prepare ``.obsm['feat']`` for each modality.
    """
    from STARMap.preprocess import clr_normalize_each_cell, pca, select_hvg_svg_genes

    selected_gene_mask = select_hvg_svg_genes(
        adata_omics1,
        n_top_hvg=n_top_hvg,
        n_top_svg=n_top_svg,
        min_cells=min_cells,
        spatial_key=spatial_key,
        svg_neighbors=svg_neighbors,
    )

    if feat_n_comps is None:
        feat_n_comps = max(1, int(adata_omics2.n_vars) - 1)
    feat_n_comps = int(feat_n_comps)

    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)
    adata_omics1_high = adata_omics1[:, selected_gene_mask].copy()
    adata_omics1.obsm["feat"] = pca(adata_omics1_high, n_comps=feat_n_comps)

    adata_omics2 = clr_normalize_each_cell(adata_omics2)
    sc.pp.scale(adata_omics2)
    adata_omics2.obsm["feat"] = pca(adata_omics2, n_comps=feat_n_comps)

    if adata_omics3 is not None:
        sc.pp.scale(adata_omics3)
        adata_omics3.obsm["feat"] = pca(adata_omics3, n_comps=feat_n_comps)

    return {
        "adata_omics1": adata_omics1,
        "adata_omics2": adata_omics2,
        "adata_omics3": adata_omics3,
        "selected_gene_mask": selected_gene_mask,
        "feat_n_comps": feat_n_comps,
    }


def prepare_starmap_brain_modalities(
    adata_omics1,
    adata_omics2,
    adata_omics3=None,
    n_top_hvg=2000,
    n_top_svg=0,
    min_cells=10,
    spatial_key="spatial",
    svg_neighbors=6,
    atac_min_cells=10,
    atac_n_components=21,
    feat_n_comps=20,
):
    from STARMap.preprocess import lsi, pca, select_hvg_svg_genes

    selected_gene_mask = select_hvg_svg_genes(
        adata_omics1,
        n_top_hvg=n_top_hvg,
        n_top_svg=n_top_svg,
        min_cells=min_cells,
        spatial_key=spatial_key,
        svg_neighbors=svg_neighbors,
    )

    feat_n_comps = int(feat_n_comps)

    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)
    adata_omics1_high = adata_omics1[:, selected_gene_mask].copy()
    adata_omics1.obsm["feat"] = pca(adata_omics1_high, n_comps=feat_n_comps)

    sc.pp.filter_genes(adata_omics2, min_cells=atac_min_cells)
    atac_n_components = max(int(atac_n_components), feat_n_comps + 1)
    atac_n_components = min(atac_n_components, adata_omics2.n_obs, adata_omics2.n_vars)
    if atac_n_components < 2:
        raise ValueError("ATAC input has too few features for LSI.")
    lsi(adata_omics2, n_components=atac_n_components, use_highly_variable=False)
    adata_omics2.obsm["feat"] = np.asarray(adata_omics2.obsm["X_lsi"], dtype=np.float64)[:, :feat_n_comps]

    if adata_omics3 is not None:
        sc.pp.scale(adata_omics3)
        adata_omics3.obsm["feat"] = pca(adata_omics3, n_comps=feat_n_comps)

    return {
        "adata_omics1": adata_omics1,
        "adata_omics2": adata_omics2,
        "adata_omics3": adata_omics3,
        "selected_gene_mask": selected_gene_mask,
        "feat_n_comps": feat_n_comps,
    }


def run_embedding_clustering_evaluation(
    adata,
    embedding_key,
    label_path,
    n_clusters,
    method="mclust",
    truth_key="ground_truth",
    pca_n_comps=20,
    random_seed=2022,
    pred_key=None,
):
    """
    Cluster an embedding, compare it to ground truth, and return bundled results.
    """
    pred_key = pred_key or embedding_key
    build_embedding_pca(
        adata,
        source_key=embedding_key,
        target_key=f"{embedding_key}_pca",
        n_comps=pca_n_comps,
        random_seed=random_seed,
        verbose=False,
    )
    add_ground_truth_labels(adata, label_path, label_key=truth_key)
    clustering(
        adata,
        key=f"{embedding_key}_pca",
        add_key=pred_key,
        n_clusters=n_clusters,
        method=method,
        use_pca=False,
    )

    scores = evaluate_clustering(adata, pred_key=pred_key, truth_key=truth_key, verbose=False)
    labels_true = adata.obs[truth_key].astype(str).to_numpy()
    label_order = sort_mixed_labels(pd.unique(labels_true).tolist())
    confusion = pd.crosstab(
        pd.Series(labels_true, name="truth"),
        pd.Series(adata.obs[pred_key].astype(str).to_numpy(), name="pred"),
    )

    return {
        "embedding_key": embedding_key,
        "pred_key": pred_key,
        "truth_key": truth_key,
        "scores": scores,
        "labels_true": labels_true,
        "label_order": label_order,
        "confusion": confusion,
    }


def plot_umap_ground_truth_vs_prediction(
    adata,
    pred_key,
    truth_key="ground_truth",
    use_rep=None,
    n_neighbors=10,
    figsize=(8, 3),
    point_size=20,
    truth_title="Ground truth",
    pred_title=None,
    w_pad=0.4,
):
    """
    Compute UMAP from a representation and plot ground truth vs predicted labels.
    """
    neighbors_kwargs = {"n_neighbors": n_neighbors}
    if use_rep is not None:
        neighbors_kwargs["use_rep"] = use_rep
    sc.pp.neighbors(adata, **neighbors_kwargs)
    sc.tl.umap(adata)

    fig, ax_list = plt.subplots(1, 2, figsize=figsize)
    sc.pl.umap(
        adata,
        color=truth_key,
        ax=ax_list[0],
        title=truth_title,
        s=point_size,
        show=False,
    )
    sc.pl.umap(
        adata,
        color=pred_key,
        ax=ax_list[1],
        title=pred_title or pred_key,
        s=point_size,
        show=False,
    )
    plt.tight_layout(w_pad=w_pad)
    plt.show()
    return fig, ax_list


def evaluate_clustering(adata, pred_key, truth_key="ground_truth", verbose=True):
    """
    Compute clustering metrics against a ground-truth label column.
    """
    if pred_key not in adata.obs:
        raise KeyError(f"{pred_key!r} is not found in adata.obs.")
    if truth_key not in adata.obs:
        raise KeyError(f"{truth_key!r} is not found in adata.obs.")

    labels_pred = adata.obs[pred_key].astype(str)
    labels_true = adata.obs[truth_key].astype(str)
    scores = {
        "ARI": metrics.adjusted_rand_score(labels_true, labels_pred),
        "NMI": metrics.normalized_mutual_info_score(labels_true, labels_pred),
        "AMI": metrics.adjusted_mutual_info_score(labels_true, labels_pred),
        "Homogeneity Score": metrics.homogeneity_score(labels_true, labels_pred),
        "Completeness Score": metrics.completeness_score(labels_true, labels_pred),
        "V-Measure": metrics.v_measure_score(labels_true, labels_pred),
    }

    if verbose:
        for name, value in scores.items():
            print(f"{name}: {value:.4f}")

    return scores


def _sorted_mixed_labels(labels):
    def _sort_key(label):
        label = str(label)
        return (0, int(label)) if label.isdigit() else (1, label)

    return sorted([str(label) for label in labels], key=_sort_key)


def summarize_best_matching_clusters(
    adata,
    pred_key,
    truth_key="ground_truth",
    rare_max_count=None,
    rare_fraction_threshold=None,
    selection_metric="overlap",
    exclude_truth_labels=None,
):
    """
    For each truth label, find the predicted cluster that maximizes the chosen
    selection metric and compute binary precision / recall / F1 / MCC.
    """
    if pred_key not in adata.obs:
        raise KeyError(f"{pred_key!r} is not found in adata.obs.")
    if truth_key not in adata.obs:
        raise KeyError(f"{truth_key!r} is not found in adata.obs.")

    labels_true = adata.obs[truth_key].astype(str)
    labels_pred = adata.obs[pred_key].astype(str)
    contingency = pd.crosstab(labels_true, labels_pred)
    contingency = contingency.reindex(
        index=_sorted_mixed_labels(contingency.index),
        columns=_sorted_mixed_labels(contingency.columns),
    )

    valid_metrics = {"overlap", "f1", "mcc", "recall", "precision"}
    if selection_metric not in valid_metrics:
        raise ValueError(f"selection_metric must be one of {sorted(valid_metrics)}.")

    total_count = int(contingency.to_numpy().sum())
    result_rows = []
    for truth_label in contingency.index:
        true_mask = (labels_true == truth_label).to_numpy()
        truth_count = int(true_mask.sum())
        fraction = truth_count / float(total_count) if total_count > 0 else 0.0

        best_entry = None
        for cluster_label in contingency.columns:
            pred_mask = (labels_pred == str(cluster_label)).to_numpy()
            overlap_count = int(np.sum(true_mask & pred_mask))
            cluster_count = int(pred_mask.sum())
            precision = metrics.precision_score(true_mask, pred_mask, zero_division=0)
            recall = metrics.recall_score(true_mask, pred_mask, zero_division=0)
            f1 = metrics.f1_score(true_mask, pred_mask, zero_division=0)
            mcc = metrics.matthews_corrcoef(true_mask, pred_mask)

            score_value = {
                "overlap": overlap_count,
                "f1": f1,
                "mcc": mcc,
                "recall": recall,
                "precision": precision,
            }[selection_metric]

            entry = (
                score_value,
                f1,
                mcc,
                recall,
                precision,
                overlap_count,
                -cluster_count,
                str(cluster_label),
                cluster_count,
            )
            if best_entry is None or entry > best_entry:
                best_entry = entry

        if best_entry is None:
            continue

        (
            _,
            best_f1,
            best_mcc,
            best_recall,
            best_precision,
            overlap_count,
            _neg_cluster_count,
            best_cluster,
            cluster_count,
        ) = best_entry

        result_rows.append(
            {
                "truth_label": str(truth_label),
                "truth_count": truth_count,
                "truth_fraction": fraction,
                "best_cluster": best_cluster,
                "cluster_count": cluster_count,
                "overlap_count": overlap_count,
                "overlap_ratio_in_truth": overlap_count / truth_count if truth_count > 0 else 0.0,
                "precision": best_precision,
                "recall": best_recall,
                "f1": best_f1,
                "mcc": best_mcc,
            }
        )

    summary = pd.DataFrame(result_rows)
    summary = summary.sort_values(["truth_count", "truth_label"]).reset_index(drop=True)

    rare_mask = pd.Series(True, index=summary.index)
    if rare_max_count is not None:
        rare_mask &= summary["truth_count"] <= int(rare_max_count)
    if rare_fraction_threshold is not None:
        rare_mask &= summary["truth_fraction"] < float(rare_fraction_threshold)
    rare_summary = summary.loc[rare_mask].reset_index(drop=True)
    if exclude_truth_labels is not None:
        exclude_truth_labels = {str(label) for label in exclude_truth_labels}
        rare_summary = rare_summary.loc[~rare_summary["truth_label"].isin(exclude_truth_labels)].reset_index(drop=True)

    return summary, rare_summary, contingency


def plot_rare_truth_vs_best_cluster_panels(
    adata,
    rare_summary,
    label_key="ground_truth",
    pred_key="STARMap",
    basis="spatial",
    figsize=(7, 3),
    spatial_point_size=25,
    w_pad=0.3,
):
    """
    For each rare truth label, plot a pair of spatial panels:
    left = truth label, right = model-found best cluster.
    """
    if label_key not in adata.obs:
        raise KeyError(f"{label_key!r} is not found in adata.obs.")
    if pred_key not in adata.obs:
        raise KeyError(f"{pred_key!r} is not found in adata.obs.")
    if rare_summary.empty:
        raise ValueError("rare_summary is empty; nothing to plot.")

    basis = _resolve_spatial_basis(adata, basis=basis)
    label_series = adata.obs[label_key].astype(str)
    pred_series = adata.obs[pred_key].astype(str)

    for row in rare_summary.itertuples(index=False):
        rare_label = str(row.truth_label)
        best_cluster = str(row.best_cluster)
        print(
            f"Rare label {rare_label}: "
            f"n={row.truth_count}, frac={row.truth_fraction:.2%}, "
            f"best_cluster={best_cluster}, F1={row.f1:.3f}, MCC={row.mcc:.3f}"
        )
        truth_mask = (label_series == rare_label).to_numpy()
        pred_mask = (pred_series == best_cluster).to_numpy()

        truth_temp_key = "__rare_truth_view__"
        pred_temp_key = "__rare_pred_view__"
        adata.obs[truth_temp_key] = pd.Categorical(
            np.where(truth_mask, "1", "0"),
            categories=["0", "1"],
        )
        adata.obs[pred_temp_key] = pd.Categorical(
            np.where(pred_mask, "1", "0"),
            categories=["0", "1"],
        )
        adata.uns[f"{truth_temp_key}_colors"] = np.array(["#d9d9d9", "#d62728"], dtype=object)
        adata.uns[f"{pred_temp_key}_colors"] = np.array(["#d9d9d9", "#1f77b4"], dtype=object)

        plot_spatial_cluster_comparison(
            adata,
            pred_key=pred_temp_key,
            truth_key=truth_temp_key,
            basis=basis,
            pred_title="Model found",
            truth_title="Ground truth",
            figsize=figsize,
            point_size=spatial_point_size,
            w_pad=w_pad,
        )

        del adata.obs[truth_temp_key]
        del adata.obs[pred_temp_key]
        del adata.uns[f"{truth_temp_key}_colors"]
        del adata.uns[f"{pred_temp_key}_colors"]

    return rare_summary["truth_label"].astype(str).tolist()


def _resolve_spatial_basis(adata, basis="spatial"):
    if basis in adata.obsm:
        return basis

    for key in ["spatial", "spatial_coords", "spatial_coord"]:
        if key in adata.obsm:
            return key

    raise KeyError(f"{basis!r} is not found in adata.obsm, and no spatial fallback key is available.")


def plot_spatial_cluster_comparison(
    adata,
    pred_key,
    truth_key="ground_truth",
    basis="spatial",
    pred_title=None,
    truth_title="Ground truth",
    figsize=(7, 3),
    point_size=25,
    w_pad=0.3,
):
    """
    Plot ground-truth and predicted spatial clusters using the same style as the notebook snippet.
    """
    basis = _resolve_spatial_basis(adata, basis=basis)
    has_truth = truth_key in adata.obs

    if has_truth:
        fig, ax_list = plt.subplots(1, 2, figsize=figsize)
        sc.pl.embedding(
            adata,
            basis=basis,
            color=truth_key,
            ax=ax_list[0],
            title=truth_title,
            s=point_size,
            show=False,
        )
        sc.pl.embedding(
            adata,
            basis=basis,
            color=pred_key,
            ax=ax_list[1],
            title=pred_title or pred_key,
            s=point_size,
            show=False,
        )
        axes = ax_list
    else:
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0] / 2.0, figsize[1]))
        sc.pl.embedding(
            adata,
            basis=basis,
            color=pred_key,
            ax=ax,
            title=pred_title or pred_key,
            s=point_size,
            show=False,
        )
        axes = [ax]

    plt.tight_layout(w_pad=w_pad)
    plt.show()
    return fig, axes

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """
    Clustering using the mclust algorithm.
    Fixed: GPU compatible, Explicit Matrix, Correct Column Naming.
    """
    np.random.seed(random_seed)
    if not check_mclust(install_if_missing=True, raise_error=False):
        raise RuntimeError("mclust is unavailable in current environment.")

    robjects = _import_robjects(raise_error=True)
    robjects.r.library("mclust")
    robjects.r['set.seed'](random_seed)

    data = adata.obsm[used_obsm]
    if hasattr(data, 'cpu'):
        data = data.cpu().detach().numpy()
    data = np.array(data, dtype=np.float64, order='C')

    nr, nc = data.shape
    r_vec = robjects.FloatVector(data.ravel())
    r_mat = robjects.r['matrix'](r_vec, nrow=nr, ncol=nc, byrow=True)
    r_colnames = robjects.StrVector([f"V{i+1}" for i in range(nc)])
    r_mat = robjects.r['colnames<-'](r_mat, r_colnames)

    r_G = robjects.IntVector([num_cluster] if isinstance(num_cluster, int) else num_cluster)
    res = robjects.r['Mclust'](data=r_mat, G=r_G, modelNames=modelNames)

    res_names = list(res.names)
    idx = res_names.index('classification') if 'classification' in res_names else -2
    mclust_res = np.array(res[idx])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int').astype('category')
    return adata


def clustering(
    adata,
    n_clusters=7,
    key="emb",
    add_key="STARMap",
    method="mclust",
    start=0.1,
    end=3.0,
    increment=0.01,
    use_pca=False,
    n_comps=20,
):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden',
        'louvain', 'kmeans', 'dbscan', and 'spectral'. The default is
        'mclust'.
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.

    Returns
    -------
    None.

    """

    if use_pca:
        build_embedding_pca(adata, source_key=key, target_key=key + "_pca", n_comps=n_comps, verbose=False)

    rep_key = key + "_pca" if use_pca else key
    embedding = _to_numpy_array(adata.obsm[rep_key], dtype=np.float64)

    if method == "mclust":
        try:
            if use_pca:
                adata = mclust_R(adata, used_obsm=key + "_pca", num_cluster=n_clusters)
            else:
                adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
            adata.obs[add_key] = adata.obs["mclust"]
            return
        except RuntimeError as exc:
            warnings.warn(f"{exc} Falling back to leiden clustering.")
            method = "leiden"

    if method == "leiden":
        if use_pca:
            res = search_res(adata, n_clusters, use_rep=key + "_pca", method=method, start=start, end=end, increment=increment)
        else:
            res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs["leiden"]
    elif method == "louvain":
        if use_pca:
            res = search_res(adata, n_clusters, use_rep=key + "_pca", method=method, start=start, end=end, increment=increment)
        else:
            res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs["louvain"]
    elif method == "kmeans":
        labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=20).fit_predict(embedding)
        adata.obs[add_key] = pd.Categorical(labels.astype(str))
    elif method == "spectral":
        labels = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            assign_labels="kmeans",
            random_state=0,
            n_neighbors=min(20, max(2, embedding.shape[0] - 1)),
        ).fit_predict(embedding)
        adata.obs[add_key] = pd.Categorical(labels.astype(str))
    elif method == "dbscan":
        labels = DBSCAN(eps=0.7, min_samples=10, metric="cosine").fit_predict(embedding)
        adata.obs[add_key] = pd.Categorical(labels.astype(str))
    else:
        raise ValueError(
            "method must be one of 'mclust', 'leiden', 'louvain', 'kmeans', 'dbscan', or 'spectral'."
        )


def search_res(adata, n_clusters, method="leiden", use_rep="emb", start=0.1, end=3.0, increment=0.01):
	"""\
	Searching corresponding resolution according to given cluster number

	Parameters
	----------
	adata : anndata
		AnnData object of spatial data.
	n_clusters : int
		Targetting number of clusters.
	method : string
		Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.
	use_rep : string
		The indicated representation for clustering.
	start : float
		The start value for searching.
	end : float
		The end value for searching.
	increment : float
		The step size to increase.

	Returns
	-------
	res : float
		Resolution.

	"""
	if method not in ["leiden", "louvain"]:
		raise ValueError("method must be 'leiden' or 'louvain'.")

	print("Searching resolution...")
	label = 0
	res = None
	count_unique = -1
	sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
	for res in sorted(list(np.arange(start, end, increment)), reverse=True):
		if method == "leiden":
			sc.tl.leiden(adata, random_state=0, resolution=res)
			count_unique = len(pd.DataFrame(adata.obs["leiden"]).leiden.unique())
			print("resolution={}, cluster number={}".format(res, count_unique))
		elif method == "louvain":
			sc.tl.louvain(adata, random_state=0, resolution=res)
			count_unique = len(pd.DataFrame(adata.obs["louvain"]).louvain.unique())
			print("resolution={}, cluster number={}".format(res, count_unique))
		if count_unique == n_clusters:
			label = 1
			break

	assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

	return res
