import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import *
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.stats import gmean, mannwhitneyu, spearmanr

def make_lfcs(ad, raw_layer, norm_layer, cl_level = "leiden", pseudocount_threshold = 0.05, cl_pct = 0.05, 
              padj_threshold = 0.05, log2FC_threshold = 0.5):
    
    def calculate_size_factor(samp, gene_is = ad.var_names):
        g_means = gmean([np.array(ad[samp,gene_is].layers[raw_layer].sum(axis = 0).squeeze()),
                         np.array(ad[~samp,gene_is].layers[raw_layer].sum(axis = 0).squeeze())])
        return np.nanmedian(np.array(ad[samp,gene_is].layers[raw_layer].sum(axis = 0).squeeze())/g_means)
    
    
    df = pd.DataFrame()
    pseudocount = np.quantile(np.mean(ad.layers[raw_layer], axis = 0), pseudocount_threshold)
    print(pseudocount)
    for cl in np.unique(ad.obs[cl_level]):
        print(cl)

        ## for each gene, number of cells with nonzero counts
        nnz_counts = np.array(np.sum(ad[ad.obs[cl_level] == cl,:].layers[raw_layer] > 0, axis = 0)).squeeze()
        ## indices for genes where > 5% of cells in the cluster have nonzero counts
        gene_is = np.where(nnz_counts >= (cl_pct*(np.sum(ad.obs[cl_level] == cl))))[0]

        ## get Pearson normalized counts for cluster and non cluster cells in these genes
        normalized_cl = ad[ad.obs[cl_level] == cl, gene_is].layers[norm_layer]
        normalized_non_cl = ad[ad.obs[cl_level] != cl,gene_is].layers[norm_layer]

        ## generate size factor, *not* taking into effect these genes with low expression counts
        cl_size_factor = calculate_size_factor(ad.obs[cl_level] == cl, gene_is)
        noncl_size_factor = calculate_size_factor(ad.obs[cl_level] != cl, gene_is)

        # get pseudobulk counts for genes adjusted by size factor, for cluster cells and non cluster cells
        cl_sums = np.array(np.sum(ad[ad.obs[cl_level] == cl, gene_is].layers[raw_layer],
                                  0)/cl_size_factor).squeeze()
        non_cl_sums = np.array(np.sum(ad[ad.obs[cl_level] != cl, gene_is].layers[raw_layer],
                                      0)/noncl_size_factor).squeeze()

        ## generate LFC in the adjusted pseudobulk counts
        log2FC = np.log2((cl_sums + pseudocount) / (non_cl_sums + pseudocount))

        ## only generate pvals for genes with hihg LFCs
        log2FC_high = np.where(abs(log2FC) > log2FC_threshold)[0]
        pvals = [mannwhitneyu(normalized_cl[:,i], normalized_non_cl[:,i])[1] for i in log2FC_high]
        pvals_corrected = [p * len(log2FC_high) for p in pvals]    
        log2FC_data = {'log2FC': log2FC.tolist(),
                       'cluster_means': cl_sums, 
                       'noncluster_means': non_cl_sums,
                       'pval': np.repeat(1, len(log2FC)), 
                       'padj': np.repeat(1, len(log2FC)), 
                       'gene': ad.var_names[gene_is]}
        log2FC_data = pd.DataFrame(log2FC_data)
        log2FC_data["pval"][log2FC_high] = pvals
        log2FC_data["padj"][log2FC_high] = pvals_corrected
        log2FC_data = log2FC_data.sort_values(by='log2FC', ascending=False)
        print('  results: log2FC > %s: %s genes; log2FC < -%s: %s genes' %
          (log2FC_threshold, sum((log2FC_data['log2FC'] > log2FC_threshold) & 
                                 (log2FC_data['padj'] < padj_threshold)),
          log2FC_threshold, sum((log2FC_data['log2FC'] < -log2FC_threshold) & 
                                 (log2FC_data['padj'] < padj_threshold))))
        log2FC_data["cluster"] = cl
        df = df.append(log2FC_data)
    return df


def plot_diff_expr(lfc_data, adata, cl_name= "leiden", top_n = 5, figsize=(50, 20), dpi = 100,
                   wspace=0.5, hspace=0.2, nrows=2, color='lightblue', only_up = False):
    fig = plt.figure(figsize=figsize, dpi = dpi)
    plot_clusters = np.unique(lfc_data["cluster"])
    ncols = int(np.ceil(len(plot_clusters) / nrows))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=wspace, hspace=hspace)
    i = 0
    for cluster in np.unique(lfc_data["cluster"]):
            logFC_cluster = lfc_data[lfc_data["cluster"] == cluster]
            genes_up = logFC_cluster[logFC_cluster['log2FC'] > 0].sort_values('log2FC', ascending = False)[:top_n]
            genes_down = logFC_cluster[logFC_cluster['log2FC'] < 0].sort_values('log2FC', ascending = False)[-top_n:]
            genes_both = pd.concat([genes_up, genes_down])
            if only_up:
                genes_both = genes_up
            # print(",".join(genes_up["gene"]))
            ax = plt.subplot(gs[i])
            i += 1
            cluster_size = sum(adata.obs[cl_name] == cluster)
            title = ('cluster %s:\n%s cells' % (cluster, cluster_size))
            if cluster_size >= 1:
                genes_both[::-1].plot(y='log2FC', # figsize=(3, len(genes_both) / 4),
                                      kind='barh',
                                      ax=ax,
                                      width=0.9, color=color, grid=False, legend=False,
                                      title=title)
            plt.xlabel('log2FC')
            plt.ylabel('')
            locs, _ =  ticks = plt.yticks()
            plt.yticks(locs, labels = genes_both[::-1]["gene"])

def make_dendrogram(genes_to_use, ad, lay = "pearson_theta_1", cluster_level = "leiden", out_name="dendrogram.pdf"):
    dendrogram_input = ad[:, genes_to_use]
    dendrogram_input = pd.DataFrame(
        dendrogram_input.layers[lay], columns = dendrogram_input.var_names, 
        index = dendrogram_input.obs[cluster_level]).reset_index().groupby(cluster_level).mean()
    print(dendrogram_input.shape)
    ## create distance matrix with cosine 
    dist_matrix = linkage(dendrogram_input, metric='cosine', method = "average")
    print(dist_matrix.shape)
    plt.figure(figsize=(10,3))
    plt.grid(False)
    set_link_color_palette(['black'])
    dendrogram(dist_matrix, above_threshold_color='black', leaf_font_size=16, show_contracted=True,
              labels = dendrogram_input.index)
    plt.savefig(out_name)
    plt.show()

cd4_genes = ["Nkg7", "Ccl5", "Slamf6", "Lef1", "Gzma", "Gzmb", "Id2", "Itgae", "Jaml", "Sell", "Tcf7"]
def make_corr_df(ad):
    genes = ad.var_names
    spearman_rhos = []
    for g in genes:
        g_vals = np.array(ad[:, g].layers["pearson_theta_1"]).flatten()
        spearman_rhos.append(spearmanr(ad.obs["normalized biotin"], g_vals))
    df = pd.DataFrame({"gene": genes, "corr": [ c[0] for c in spearman_rhos]})
    df["interesting gene"] = df.gene.isin(cd4_genes)
    df = df.sort_values("corr") 
    df["i"] = [i/10 for i in range(len(genes))]
    return df

def make_corr_plot(df):
    return ggplot(df[~df["interesting gene"]], aes(x="i", y="corr")) + geom_point(color = "#696969") + \
theme_minimal() + geom_point(df[df["interesting gene"]], aes(x="i", y="corr"), color="#8b1c62") + geom_text(
        df[df["interesting gene"]], aes(label="gene"), color="#8b1c62", size=12, nudge_y=0, nudge_x = 130, 
        adjust_text = {"autoalign": "x", "arrowprops": { "arrowstyle": '-', "color": "black", "lw": 0.2}}) + \
xlim(0, 1600) + labs(x=" ", y="Correlation") + theme(dpi=200, axis_text_x=element_blank(),
                                                     legend_position="none", figure_size=(5,3)) 
