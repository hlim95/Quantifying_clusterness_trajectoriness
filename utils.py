import matplotlib.pyplot as plt
from anndata import AnnData
import scanpy_modified as scanpy
from tqdm import tqdm
import numpy as np
from numpy import inf
from ripser import Rips
from scipy.spatial import distance
from numpy import linalg as LA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import sklearn
import pandas as pd
from sklearn.manifold import TSNE
import umap
import glob, os
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import time
import sys
from scipy import sparse
from scipy.stats import rankdata
from scipy.stats import entropy
from sklearn.decomposition import PCA
eps = sys.float_info.epsilon

def preprocessing(data):
    data = density_downsampling(data, od = 0.01, td = 1)
    data = normalize(data)
    return(data)

def normalize(data):
    normalized = (data - np.min(data, axis = 0)) / (np.max(data, axis = 0) - np.min(data, axis = 0))
    return normalized 

def density_downsampling(Data, od = 0.01, td = 1):
    np.random.seed(0)
    dist = distance.pdist(Data, metric='euclidean')    
    dist_m = distance.squareform(dist)
    sorted_dist_m = np.sort(dist_m)
    median_min_dist = np.median(sorted_dist_m[:,1])
    dist_thres = 5 * median_min_dist

    local_densities = np.sum(1*(dist_m < dist_thres),0)
    #print(local_densities)
    OD = np.quantile(local_densities, od)
    TD = np.quantile(local_densities, td)
    #print(OD,TD)
    IDX_TO_KEEP = []
    for i in range(len(local_densities)):
        if local_densities[i] < OD:
            continue
        elif local_densities[i] > TD:
            if np.random.uniform(0,1) < TD/local_densities[i]:
                 IDX_TO_KEEP.append(i)
        else:
            IDX_TO_KEEP.append(i)
    downsampled_data = Data[IDX_TO_KEEP,:]
    return downsampled_data


def features_dpt_entropy(data, num_bins = 10, visualize = False):
    num_data = len(data)
    data = AnnData(data)
    data.uns['iroot'] = 0
    scanpy.pp.neighbors(data,n_neighbors=max(10,int(0.005*num_data)), method='umap',knn=True)
    scanpy.tl.diffmap(data)
    scanpy.tl.dpt(data)
    tmp = np.stack(data.obs['dpt_distances'])
    tmp[tmp == inf] = 1.5 * np.max(tmp[tmp != inf]) 
    tmp[tmp == -inf] = -1 * np.min(tmp[tmp != -inf]) 
    a = plt.hist(tmp[np.triu(tmp, 1) != 0], bins = num_bins)
    hs = a[0]/np.sum(a[0])
  
    ent = entropy(hs, base=num_bins)
    if visualize == True:
        plt.show()
    elif visualize == False:
        plt.close()
    return ent


def features_homology_dpt_entropy(data, num_bins = 3, visualize = False):
    data = AnnData(data)
    data.uns['iroot'] = 0
    #scanpy.pp.neighbors(data,n_neighbors=5, method='umap', knn=True)
    scanpy.pp.neighbors(data)
    scanpy.tl.diffmap(data)
    scanpy.tl.dpt(data)
    tmp = np.stack(data.obs['dpt_distances'])
    tmp[tmp == inf] = np.random.normal(1.5, 0.1) * np.max(tmp[tmp != inf]) 
    tmp[tmp == -inf] = -1 * np.min(tmp[tmp != -inf]) 
    rips = Rips(maxdim=0,verbose = False)
    diagrams = rips.fit_transform(sparse.csr_matrix(tmp),distance_matrix = True)
    a = plt.hist(diagrams[0][:-1,1], bins = num_bins)
    hs = a[0]/np.sum(a[0])   
    ent = entropy(hs, base=num_bins)
    ent = np.log(ent)
    if visualize == True:
        plt.show()
    elif visualize == False:
        plt.close()
    return ent

def features_vector(data, metric = 'euclidean'):
    _, dim = data.shape
    if dim > 5:
        pca = PCA(n_components=5)
        data = pca.fit_transform(data)
    _, dim = data.shape
    X = data
    num_clusters = int(len(X)/20)
    SCORES = 0
    num_rep = 3
    for rep in range(num_rep):
        kmeans = KMeans(n_clusters=num_clusters).fit(X)
        for m in range(num_clusters):

            clusters = kmeans.cluster_centers_.tolist()
            all_dist = sklearn.metrics.pairwise_distances(clusters, metric=metric)
            threshold = np.percentile(all_dist[np.triu(all_dist, 1) !=0], 20)
            index = m
            dist = sklearn.metrics.pairwise_distances(np.array(clusters[index]).reshape(1, -1),clusters, metric=metric)[0]
            current_index = index
            kmean_order = []
            for i in range(len(clusters)):
                current_kmean = clusters[current_index]
                kmean_order.append(current_kmean)
                clusters.remove(clusters[current_index])
                dist_current = sklearn.metrics.pairwise_distances(np.array(current_kmean).reshape(1, -1), clusters, metric=metric)[0]
                if len(dist_current) == 1:
                    break
                next_index = np.argsort(dist_current)[0]  
                if dist_current[next_index] > threshold:
                    break
                current_index = next_index
            vectors = []
            for j in range(len(kmean_order)-1):
                d1 = np.array(kmean_order[j])
                d2 = np.array(kmean_order[j+1])
                vectors.append(d2-d1)
            a = np.sum(vectors,axis = 0)
            try:
                norm = LA.norm(np.sum(vectors,axis = 0),ord=dim)
                SCORES += norm
            except Exception:
                pass
    return SCORES/(num_clusters*num_rep)

def features_ripley_dpt_v2(data, visualize = False):
    X = data
    n, dim = X.shape
    MIN_MAX = []
    for i in range(dim):
        dim_min = np.min(X[:,i])
        dim_max = np.max(X[:,i])
        MIN_MAX.append((dim_min,dim_max))
    num_repeats = 1
    rScore = 0 
    for i in range(num_repeats):
        SDATA = []
        for i in range(len(MIN_MAX)):
            shuffled_data = np.random.uniform(MIN_MAX[i][0],MIN_MAX[i][1],n)
            SDATA.append(shuffled_data)
           
        SX = np.array(SDATA).T

        data = AnnData(X)
        data.uns['iroot'] = 0
        scanpy.pp.neighbors(data)
        scanpy.tl.diffmap(data)
        scanpy.tl.dpt(data)
        DX = np.stack(data.obs['dpt_distances'])

        data = AnnData(SX)
        data.uns['iroot'] = 0
        scanpy.pp.neighbors(data)
        scanpy.tl.diffmap(data)
        scanpy.tl.dpt(data)
        DSX = np.stack(data.obs['dpt_distances'])        
        DX[DX == inf] = 1.5 * np.max(DX[DX != inf]) 
        DSX[DSX == inf] = 1.5 * np.max(DSX[DSX != inf]) 

        xs = np.linspace(0,np.nanmax(DX[DX != -np.inf])+1,100)
        T_K_DX = []
        for i in range(len(xs)):
            th = xs[i]
            K_DX = np.sum( np.sum(1*(DX < th), axis = 0)/n) 
            T_K_DX.append(K_DX)
    
    
        xs = np.linspace(0,np.nanmax(DSX[DSX != -np.inf])+1,100)
        T_K_DSX = []
        for i in range(len(xs)):
            th = xs[i]
            K_DSX = np.sum( np.sum(1*(DSX < th), axis = 0)/n) 
            T_K_DSX.append(K_DSX)

        dx_n = normalize(T_K_DX)
        dsx_n = normalize(T_K_DSX)
        sum_n = np.array(dx_n) + np.array(dsx_n)
        dx_n = dx_n[sum_n != 2]
        dsx_n = dsx_n[sum_n != 2]

        Score = np.trapz(np.abs(dx_n - dsx_n), dx = 1/len(dx_n))
        rScore += Score
    return rScore / num_repeats


def features_avg_connection_dpt(df):
    SCORE = []
    c = density_downsampling(df,od = 0.03, td = 0.3)
    K = np.linspace(0.03, 1, 20)    
    k_scores = []
    for k in K:
        sc = generate_score_k_dpt(c, k)
        k_scores.append(sc)
    score = np.trapz(k_scores, K/np.max(K))
    return score

def generate_score_k_dpt(tmp_data, k):
    if len(tmp_data) > 200:
        num_repeats = 5
        final_score = 0
        for i in range(num_repeats):
            idx = np.random.randint(0,len(tmp_data), size=200)
            t_data = tmp_data[idx,:]

            data = AnnData(t_data)
            data.uns['iroot'] = 0
            scanpy.pp.neighbors(data)
            scanpy.tl.diffmap(data)
            scanpy.tl.dpt(data)
            DX = np.stack(data.obs['dpt_distances'])
            DX[DX == inf] = 1.5 * np.max(DX[DX != inf]) 
            
            K = int(len(t_data) * k)

            knn_distance_based = (
                NearestNeighbors(n_neighbors=int(len(t_data) * k), metric="precomputed")
                    .fit(DX)
            )
            
            A = knn_distance_based.kneighbors_graph(DX).toarray()
            
            SA = 1*((A + A.T) > 1.5)
            old_total = 0 
            for i in range(10):
                total = np.sum(SA)
                if total == old_total:
                    break
                else:
                    old_total = total
                SA = np.matmul(SA,SA)
                SA = 1*(SA > 1)

            avg_connect = np.sum(SA,axis=0) / len(SA)
            final_score += np.median(avg_connect)

        return final_score/num_repeats

    else:
        data = AnnData(tmp_data)
        data.uns['iroot'] = 0
        scanpy.pp.neighbors(data)
        scanpy.tl.diffmap(data)
        scanpy.tl.dpt(data)
        DX = np.stack(data.obs['dpt_distances'])
        DX[DX == inf] = 1.5 * np.max(DX[DX != inf]) 
        
        

        knn_distance_based = (
            NearestNeighbors(n_neighbors=int(len(tmp_data) * k), metric="precomputed")
                .fit(DX)
        )
        
        A = knn_distance_based.kneighbors_graph(DX).toarray()
        
        SA = 1*((A + A.T) > 1.5)
        old_total = 0 
        for i in range(10):
            total = np.sum(SA)
            if total == old_total:
                break
            else:
                old_total = total
            SA = np.matmul(SA,SA)
            SA = 1*(SA > 1)
        avg_connect = np.sum(SA,axis=0) / len(SA)
        final_score = np.median(avg_connect)
    
        return final_score

