import json

import random
import numpy as np
import torch
import fire
torch.set_grad_enabled(False)
from torchvision import transforms
from typing import Sequence
import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(__file__), ".."))
import os
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
import torch.nn.functional as F

from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts, DatasetCatalog, MetadataCatalog
from detectron2.data import transforms as T
from tqdm.auto import tqdm
from torchvision.transforms import functional as tvF
import torchvision as tv
from detectron2.data.dataset_mapper import DatasetMapper

from lib.detr_mapper import DetrDatasetMapper
from fast_pytorch_kmeans import KMeans
from lib.categories import ALL_CLS_DICT
import lib.data.fewshot
import lib.data.ovdshot
import lib.data.lvis

import matplotlib.pyplot as plt
from skimage.filters import gaussian
import cv2
import  faiss
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns



pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
pixel_std = torch.Tensor([58.395, 57.120, 57.375]).view(3, 1, 1)
normalize_image = lambda x: (x - pixel_mean) / pixel_std
denormalize_image = lambda x: (x * pixel_std) + pixel_mean


def compress(tensor, n_clst=5):
    if len(tensor) <= n_clst:
        # may be normalize this
        # the raw tokens are not normalized
        return tensor
    else:
        kmeans = KMeans(n_clusters=n_clst, verbose=False, mode='cosine')
        kmeans.fit(tensor)
        return kmeans.centroids
    
    
def save_img(img, path):
    tv.utils.save_image(denormalize_image(img) / 255, path)

def iround(x): return int(round(x))

def crop(img, box, enlarge=0.2):
    h,w = img.shape[1:]

    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2  
    lx = (box[2] - box[0]) * (1 + enlarge)
    ly = (box[3] - box[1]) * (1 + enlarge)

    x0 = max(int(round(cx - lx / 2)), 0)
    x1 = min(int(round(cx + lx / 2)), w)
    y0 = max(int(round(cy - ly / 2)), 0)
    y1 = min(int(round(cy + ly / 2)), h)

    return img[:, y0:y1, x0:x1]


def resize_with_largest_edge(img, size=224):
    h, w = img.shape[1:]
    if h >= w:
        ratio = w / h
        h = size
        w = iround(h * ratio)
    else:
        ratio = h / w
        w = size
        h = iround(ratio * w)
    return tvF.resize(img, (h, w), interpolation=tvF.InterpolationMode.BICUBIC)

def resize_to_closest_14x(img):
    h, w = img.shape[1:]
    h, w = max(iround(h / 14), 1) * 14, max(iround(w / 14), 1) * 14
    return tvF.resize(img, (h, w), interpolation=tvF.InterpolationMode.BICUBIC)


def to_mask(boxes, height, width):
    result = torch.zeros(len(boxes), height, width)
    boxes = torch.round(boxes).long()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        result[i, y1:y2, x1:x2] = 1
    return result.bool()


# Step 1: Randomly generate boxes that do not intersect with annotations
def generate_mask(image, gt_boxes, num_b=10, min_size=30, max_size=200, max_iter=100, patch_size=14):
    '''
    Randomly generate num_b masks that do not intersect with any ground truth boxes for negative sample generation

    Args:
        image (torch.Tensor): Input image tensor with shape [C, H, W]
        gt_boxes (torch.Tensor): Ground truth box tensor with shape [N, 4] in [x1, y1, x2, y2] format
        num_b (int): Number of background samples to extract per image
        min_size (int): Minimum size of the box
        max_size (int): Maximum size of the box
        max_iter (int): Maximum iterations for generating valid boxes
        patch_size (int): Patch size of the backbone network
    '''
    # Get dimensions from image tensor (C, H, W)
    _, h, w = image.shape
    mask = np.zeros((1, h, w), dtype=np.uint8)  # Initialize mask

    for _ in range(num_b):
        valid_box = False
        count = 0
        while not valid_box and count < max_iter:
            count += 1

            # Generate random top-left coordinates and width/height of the box
            # Ensure the box does not exceed image boundaries
            x = random.randint(0, w - max_size)
            y = random.randint(0, h - max_size)
            width = random.randint(min_size, max_size)
            height = random.randint(min_size, max_size)

            # Calculate bottom-right coordinates
            x2 = x + width
            y2 = y + height

            # Check for intersection with any ground truth box
            # gt_boxes format: [N, 4], each box is [x1, y1, x2, y2]
            intersects = False
            for box in gt_boxes:
                bx1, by1, bx2, by2 = box
                # Rectangle intersection detection
                if (x < bx2 and x2 > bx1 and
                        y < by2 and y2 > by1):
                    intersects = True
                    break

            if not intersects:
                # Mark this valid background region in the mask
                mask[:, y:y2, x:x2] = 1
                valid_box = True

        # Skip current sample if no valid box found within max iterations
        if count >= max_iter:
            print(f"Warning: Exceeded maximum iterations {max_iter}, failed to generate valid background box")

    # Convert mask to tensor and adjust size to match patch size
    mask = torch.as_tensor(mask, dtype=torch.float32)
    # Downsample to patch level (1, H/patch_size, W/patch_size)
    mask = F.interpolate(
        mask.unsqueeze(0),
        size=(h // patch_size, w // patch_size),
        mode='nearest'
    ).squeeze(0)
    # Reshape for subsequent processing
    mask = mask.reshape(-1, (h // patch_size) * (w // patch_size))

    return mask
    
def get_dataloader(dname, aug=False, split=0, idx=0):
    if aug:
        print("ENABLE AUGMENTATION")
        augmentation = [
            T.RandomBrightness(0.9, 1.1),
            T.RandomContrast(0.9, 1.1),
            T.RandomSaturation(0.9, 1.1),
            T.RandomFlip(),
            T.ResizeShortestEdge(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ]
        augmentation_with_crop=[
            T.RandomBrightness(0.9, 1.1),
            T.RandomContrast(0.9, 1.1),
            T.RandomSaturation(0.9, 1.1),
            T.RandomFlip(),
            T.ResizeShortestEdge(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            T.RandomCrop(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            T.ResizeShortestEdge(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ]
    else:
        augmentation=[
            T.ResizeShortestEdge(
                short_edge_length=800,
                max_size=1333,
            ),
        ]
        augmentation_with_crop=[]

    dataset_dicts = get_detection_dataset_dicts(dname)
    if split > 0:
        size = len(dataset_dicts) // split
        if idx < split - 1:
            dataset_dicts = dataset_dicts[size * idx: size * (idx + 1)]
        else:
            dataset_dicts = dataset_dicts[size * idx:]
    
    return build_detection_test_loader(dataset=dataset_dicts,
                            mapper=DatasetMapper(
                    augmentations=augmentation,
                    is_train=True,
                    image_format="RGB"
            ), num_workers=4)

def extract_top_clusters(centroids, labels, top_n=8):
    """
    Extract the top N clusters by sample count and their feature vectors

    Parameters:
        centroids: Cluster centroid data
        labels: Cluster labels
        top_n: Number of top clusters to extract, default is 8

    Returns:
        top_cluster_ids: IDs of the top N clusters
        top_centroids: Corresponding feature vectors of cluster centroids
        cluster_sizes: Sample count of each cluster
    """
    # Count sample number for each cluster
    unique_labels, cluster_sizes = np.unique(labels, return_counts=True)

    # Sort by sample count in descending order and get top_n cluster IDs
    sorted_indices = np.argsort(cluster_sizes)[::-1]  # Indices for descending sort
    top_cluster_ids = unique_labels[sorted_indices[:top_n]]  # Take top 8
    top_sizes = cluster_sizes[sorted_indices[:top_n]]  # Corresponding sample counts

    # Print information of top 8 clusters
    print(f"Top {top_n} largest clusters:")
    for i, (cluster_id, size) in enumerate(zip(top_cluster_ids, top_sizes)):
        print(f"Rank {i + 1}: Cluster ID {cluster_id}, Sample count {size}")

    # Extract corresponding feature vectors from centroids
    top_centroids = centroids[top_cluster_ids]

    return top_cluster_ids, top_centroids, top_sizes


def visualize_top_clusters_with_samples(
        data_np,  # Original sample data (shape: [N, D])
        labels,  # Cluster labels for all samples (shape: [N,])
        top_centroids,  # Centroids of top K clusters (shape: [K, D])
        top_cluster_ids,  # IDs of top K clusters (shape: [K,])
        sample_ratio=0.3,  # Sampling ratio (controls number of visualized samples)
        perplexity=30,  # t-SNE parameter, affects local structure preservation
        random_state=42  # Random seed for reproducibility
):
    """
    Visualize sample distribution and corresponding centroid positions of top K clusters (based on t-SNE dimensionality reduction)

    Parameters:
        data_np: Original sample data containing feature vectors of all samples
        labels: Cluster label corresponding to each sample
        top_centroids: Centroids of top K clusters
        top_cluster_ids: IDs of top K clusters
        sample_ratio: Sampling ratio from each cluster (between 0-1)
        perplexity: Perplexity parameter for t-SNE (recommended 5-50)
        random_state: Random seed to ensure stable visualization results
    """
    # Get number of clusters K
    K = len(top_centroids)
    assert K == len(top_cluster_ids), "Number of centroids must match number of cluster IDs"
    print(f"Visualizing top {K} clusters (including samples and centroids)")

    # --------------------------
    # 1. Filter samples belonging to top K clusters
    # --------------------------
    # Create mask: filter samples belonging to top K clusters
    mask = np.isin(labels, top_cluster_ids)
    cluster_samples = data_np[mask]  # Samples belonging to top K clusters
    cluster_labels = labels[mask]  # Corresponding labels

    # Sample each cluster individually (avoid large clusters overshadowing small ones)
    sampled_samples = []
    sampled_labels = []
    for cid in top_cluster_ids:
        c_samples = cluster_samples[cluster_labels == cid]
        # Sample by ratio for each cluster, keep at least 5 samples (avoid empty clusters)
        n_sample = max(5, int(len(c_samples) * sample_ratio))
        if len(c_samples) <= n_sample:
            # Take all samples when insufficient
            sampled_samples.append(c_samples)
            sampled_labels.append(np.full(len(c_samples), cid))
        else:
            # Random sampling
            idx = np.random.choice(len(c_samples), n_sample, replace=False)
            sampled_samples.append(c_samples[idx])
            sampled_labels.append(np.full(n_sample, cid))

    # Merge sampled samples and labels
    sampled_samples = np.vstack(sampled_samples)
    sampled_labels = np.hstack(sampled_labels)
    print(f"Total sampled samples: {len(sampled_samples)} (from top {K} clusters)")

    # --------------------------
    # 2. t-SNE dimensionality reduction (unified space for samples + centroids)
    # --------------------------
    # Merge samples and centroids to ensure same t-SNE space
    combined_data = np.vstack([sampled_samples, top_centroids])
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init='pca',
        learning_rate='auto'
    )
    combined_2d = tsne.fit_transform(combined_data)

    # Separate 2D coordinates of samples and centroids
    samples_2d = combined_2d[:-K]  # First N rows are samples
    centroids_2d = combined_2d[-K:]  # Last K rows are centroids

    # --------------------------
    # 3. Visualize samples and centroids
    # --------------------------
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")

    # Generate color palette
    palette = sns.color_palette("husl", K)
    # Create mapping from cluster ID to color index (ensure one-to-one correspondence)
    cid_to_idx = {cid: i for i, cid in enumerate(top_cluster_ids)}

    # Plot sample points
    for i, cid in enumerate(top_cluster_ids):
        # Filter samples of current cluster
        sample_mask = (sampled_labels == cid)
        # Total sample size of this cluster (in original data)
        total_size = np.sum(labels == cid)

        plt.scatter(
            samples_2d[sample_mask, 0],
            samples_2d[sample_mask, 1],
            color=palette[i],
            label=f"Cluster {cid} (total: {total_size})",
            alpha=0.6,
            s=50,
            edgecolors="none",
            zorder=2
        )

    # Plot centroids (marked with star)
    for i, cid in enumerate(top_cluster_ids):
        plt.scatter(
            centroids_2d[i, 0],
            centroids_2d[i, 1],
            color=palette[i],
            marker="*",
            s=400,
            edgecolors="black",
            linewidths=2,
            zorder=3  # Place above sample points
        )

    # Chart settings
    plt.title(f"Top {K} Clusters (Samples + Centroids)", fontsize=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"top_{K}_clusters_with_samples.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --------------------------
    # 4. Output cluster quality metrics
    # --------------------------
    print("\nCluster Quality Metrics:")
    # 1. Sample size of each cluster
    for cid in top_cluster_ids:
        print(f"Cluster {cid}: Total samples = {np.sum(labels == cid)}")

    # 2. Average intra-cluster distance (average distance from samples to centroid)
    for i, cid in enumerate(top_cluster_ids):
        # Get all samples of this cluster
        all_samples = data_np[labels == cid]
        if len(all_samples) == 0:
            continue
        # Calculate average distance
        centroid = top_centroids[i]
        avg_dist = np.mean(np.linalg.norm(all_samples - centroid, axis=1))
        print(f"Cluster {cid}: Average intra-cluster distance = {avg_dist:.4f} (smaller = more compact)")


def cluster_features(data, K_base=200, min_niter=50, max_niter=300):
    """
    Improved intelligent clustering function that determines optimal K value using multiple evaluation metrics

    Parameters:
        data: torch.Tensor or numpy array with shape (D, N)
        D: Feature dimension (1024)
        N: Number of samples (>10000)

    Returns:
        dict: Dictionary containing clustering results and evaluation metrics
    """
    if isinstance(data, torch.Tensor):
        data_np = data.T.cpu().numpy().astype('float32')
        data_np = np.ascontiguousarray(data_np)
    else:
        data_np = data.T.astype('float32')
        data_np = np.ascontiguousarray(data_np)

    N, D = data_np.shape

    # Check GPU availability
    gpu_count = faiss.get_num_gpus()
    print(f"Detected GPU count: {gpu_count}")
    if gpu_count == 0:
        print("Warning: No GPU detected! Clustering will run on CPU.")

    # Automatically determine k value range
    max_possible_k = min(K_base, int(np.sqrt(N)))
    k_range = range(2, max_possible_k + 1, 1)
    metrics = {
        'k_values': list(k_range),
        'inertia': [],
        'db_index': [],
        'silhouette': [],
        'calinski': [],
        'best_k': None,
        'centroids': None,
        'labels': None
    }
    # --------------------------
    # Dynamic niter selection (for 1024-dim features)
    # --------------------------
    # For high-dimensional data (1024D), more iterations are needed for convergence
    # Larger N or K requires more iterations
    if N <= 10000:
        niter = min(max(min_niter, 80), max_niter)  # Small dataset: 80 iterations
    elif N <= 50000:
        niter = min(max(min_niter, 120), max_niter)  # Medium dataset: 120 iterations
    elif N <= 200000:
        niter = min(max(min_niter, 180), max_niter)  # Large dataset: 180 iterations
    else:
        niter = min(max(min_niter, 250), max_niter)  # Very large dataset: 250 iterations

    clustering_params = {
        'niter': niter,
        'verbose': False,
        'seed': 42,
        'gpu': (gpu_count > 0)
    }

    # Multi-k value testing
    for k in k_range:
        kmeans = faiss.Kmeans(D, k, **clustering_params)
        kmeans.train(data_np)
        _, labels = kmeans.index.search(data_np, 1)
        labels = labels.flatten()

        # Calculate multiple evaluation metrics
        metrics['inertia'].append(kmeans.obj[-1])
        metrics['db_index'].append(davies_bouldin_score(data_np, labels))
        metrics['silhouette'].append(silhouette_score(data_np, labels))
        metrics['calinski'].append(calinski_harabasz_score(data_np, labels))
        print(f"\nK={k} completed")

    # Comprehensive evaluation to determine optimal k value
    db_scores = np.array(metrics['db_index'])
    silhouette_scores = np.array(metrics['silhouette'])
    calinski_scores = np.array(metrics['calinski'])

    # Normalize all metrics
    normalized_db = (db_scores - db_scores.min()) / (db_scores.max() - db_scores.min())
    normalized_sil = 1 - (silhouette_scores - silhouette_scores.min()) / (
                silhouette_scores.max() - silhouette_scores.min())
    normalized_cal = 1 - (calinski_scores - calinski_scores.min()) / (calinski_scores.max() - calinski_scores.min())

    # Weighted comprehensive score
    combined_scores = 0.4 * normalized_db + 0.3 * normalized_sil + 0.3 * normalized_cal
    best_k_idx = np.argmin(combined_scores)
    best_k = k_range[best_k_idx]

    print(f"Automatically determined niter: {clustering_params['niter']} (based on N={N} and K={best_k})")

    # Final clustering
    final_kmeans = faiss.Kmeans(D, best_k, **clustering_params)
    final_kmeans.train(data_np)
    _, final_labels = final_kmeans.index.search(data_np, 1)
    final_labels = final_labels.flatten()

    # Extract top n largest clusters
    top_cluster_ids, top_centroids, top_sizes = extract_top_clusters(final_kmeans.centroids, final_labels, 8)

    # Update metrics with top 8 cluster information
    metrics.update({
        'best_k': best_k,
        'centroids': final_kmeans.centroids,
        'labels': final_labels,
        'combined_scores': combined_scores,
        'topn_cluster_ids': top_cluster_ids,  # IDs of top n clusters
        'topn_centroids': top_centroids,  # Feature vectors of top n clusters
        'topn_cluster_sizes': top_sizes  # Sample counts of top n clusters
    })

    print(f"\nOptimal K selected: {best_k}")

    # Directly visualize using top K centroids
    visualize_top_clusters_with_samples(
        data_np=data_np,
        labels=final_labels,
        top_centroids=top_centroids,
        top_cluster_ids=top_cluster_ids,
        sample_ratio=0.3,  # Sample 30% of samples from each cluster
        perplexity=30,  # t-SNE parameter suitable for medium-scale data
        random_state=42  # Fixed random seed for reproducible results
    )
    # Visualization
    visualize_metrics(metrics, k_range)

    return metrics


def visualize_metrics(metrics, k_range):
    """Improved visualization function"""
    plt.figure(figsize=(15, 10))

    # Elbow method plot
    plt.subplot(2, 2, 1)
    plt.plot(k_range, metrics['inertia'], 'bo-', linewidth=2, markersize=8)
    plt.title('Elbow Method', fontsize=14, pad=20)
    plt.xlabel('Number of clusters', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # DB index plot
    plt.subplot(2, 2, 2)
    plt.plot(k_range, metrics['db_index'], 'rD-', linewidth=2, markersize=8)
    plt.title('Davies-Bouldin Index', fontsize=14, pad=20)
    plt.xlabel('Number of clusters', fontsize=12)
    plt.ylabel('DB Index', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Silhouette score plot
    plt.subplot(2, 2, 3)
    plt.plot(k_range, metrics['silhouette'], 'g^-', linewidth=2, markersize=8)
    plt.title('Silhouette Score', fontsize=14, pad=20)
    plt.xlabel('Number of clusters', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Combined evaluation score plot
    plt.subplot(2, 2, 4)
    plt.plot(k_range, metrics['combined_scores'], 'ms-', linewidth=2, markersize=8)
    plt.title('Combined Evaluation Score', fontsize=14, pad=20)
    plt.xlabel('Number of clusters', fontsize=12)
    plt.ylabel('Combined Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add optimal K value marker
    best_k = metrics['best_k']
    if best_k:
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.axvline(x=best_k, color='k', linestyle='--',
                        label=f'Optimal K={best_k}')
            plt.legend()

    plt.tight_layout()
    plt.savefig('clustering_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

def main(model='vitl14', dataset='fs_coco17_support_novel_30shot', use_bbox='yes',
            epochs=1, device=0, n_clst=200, split=0, idx=0, out_dir=None, without_mask=False):

    use_bbox = use_bbox == 'yes'
    dataset_name = dataset
    model_name = model

    ckpt = "/home/jack/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth"
    from dinov2.models.vision_transformer import vit_large
    model = vit_large(patch_size=14, img_size=518, init_values=1.0, block_chunks=0)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dataloader = get_dataloader(dataset_name, split=split, idx=idx)

    D = DatasetCatalog.get(dataset_name)
    thing_cats = {b['category_id'] for a in D for b in a.get('annotations', [])}
    print(f'Found thing categories: {len(thing_cats)}')
    
    if device != 'cpu':
        device = int(device)

    model = model.to(device)

    background_features = []

    with tqdm(total=epochs * len(dataloader)) as bar:
        for _ in range(epochs):
            for item_i, item in enumerate(dataloader):
                item = item[0]
                if 'instances' in item:
                    instances = item['instances'].to(device)
                image = item['image'].clone()
                image14x = resize_to_closest_14x(normalize_image(image)).to(device)

                # Generate mask
                mask = generate_mask(image14x, gt_boxes=instances.gt_boxes.tensor).to(device)

                target_mask_size = image14x.shape[1] // 14, image14x.shape[2] // 14

                r = model.get_intermediate_layers(image14x[None, ...],
                                        return_class_token=True, reshape=True)
                patch_tokens = r[0][0][0] # c, h, w
                features = patch_tokens.flatten(1)
                masked_patch_tokens = features * mask
                filtered_patch_tokens = masked_patch_tokens[:, mask.squeeze(0) == 1]
                background_features.append(filtered_patch_tokens.cpu())

    background_features= torch.cat(background_features, dim=-1).numpy()
    d, n = background_features.shape
    print('\nDone. Extracted background {} vectors of dimensionality = {}'.format(n, d))
    #kmeans_features,k = cluster_features(background_features, K_base=n_clst)
    dic_k = cluster_features(background_features, K_base=n_clst)
    kmeans_features = dic_k['centroids']
    k =dic_k['best_k']
    # Normalize the tensor along dim=1 using F.normalize
    prototypes = F.normalize(torch.from_numpy(kmeans_features), dim=1)  # Normalize the feature vectors
    classes = ['bg_class_{}'.format(i + 1) for i in range(prototypes.shape[0])]

    category_dict = {
        'prototypes': prototypes,
        'label_names': classes
    }
    name = f'weights/background/{dataset_name}/'+dataset_name + '.' + model_name+f'_{k}'+'.pth'
    dir_path = os.path.dirname(name)
    os.makedirs(dir_path, exist_ok=True)
    torch.save(category_dict, name)
    print(f'Saved prototypes to {name}')


if __name__ == "__main__":
    fire.Fire(main)
