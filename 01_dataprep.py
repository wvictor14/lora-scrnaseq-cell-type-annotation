"""
Data preparation for RA synovial scRNA-seq cell-type annotation.
Loads SCP279, maps clusters to hierarchical labels (Levels 1-3),
generates per-donor marker lists via DE, and saves training examples.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('prepare_data')

# Paths
DATA_DIR = Path('data/SCP279')
EXPR_PATH = DATA_DIR / 'expression/exprMatrix.tsv.gz'
META_PATH = DATA_DIR / 'metadata/meta.txt'
OUTPUT_DIR = Path('data')

# Example generation
RANDOM_SEED = 42
TOP_N_GENES = 20
MIN_CELLS_BASE = 10          # step 3: min cells for a "real" per-donor example
MIN_CELLS_AUGMENT = 5        # lower bar to be eligible as an augmentation source
LEVEL3_TARGET_PER_LABEL = 40  # 4 labels x 40 ~= 160, matches "~150 examples/level"
MAX_AUGMENT_PER_GROUP = 8    # cap bootstrap draws from any single (donor,label) group
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# Cluster to hierarchical labels (single source of truth)
CLUSTER_LABELS = {
    'SC-T1': {'level1': 'Immune', 'level2': 'T/NK', 'level3': 'Tconv'},
    'SC-T2': {'level1': 'Immune', 'level2': 'T/NK', 'level3': 'Treg'},
    'SC-T3': {'level1': 'Immune', 'level2': 'T/NK', 'level3': 'Tph'},
    'SC-T4': {'level1': 'Immune', 'level2': 'T/NK', 'level3': 'CD8 T'},
    'SC-T5': {'level1': 'Immune', 'level2': 'T/NK', 'level3': 'CD8 T'},
    'SC-T6': {'level1': 'Immune', 'level2': 'T/NK', 'level3': 'CD8 T'},
    'SC-M1': {'level1': 'Immune', 'level2': 'Myeloid', 'level3': None},
    'SC-M2': {'level1': 'Immune', 'level2': 'Myeloid', 'level3': None},
    'SC-M3': {'level1': 'Immune', 'level2': 'Myeloid', 'level3': None},
    'SC-M4': {'level1': 'Immune', 'level2': 'Myeloid', 'level3': None},
    'SC-B1': {'level1': 'Immune', 'level2': 'B/Plasma', 'level3': None},
    'SC-B2': {'level1': 'Immune', 'level2': 'B/Plasma', 'level3': None},
    'SC-B3': {'level1': 'Immune', 'level2': 'B/Plasma', 'level3': None},
    'SC-B4': {'level1': 'Immune', 'level2': 'B/Plasma', 'level3': None},
    'SC-F1': {'level1': 'Stromal', 'level2': 'Fibroblast', 'level3': None},
    'SC-F2': {'level1': 'Stromal', 'level2': 'Fibroblast', 'level3': None},
    'SC-F3': {'level1': 'Stromal', 'level2': 'Fibroblast', 'level3': None},
    'SC-F4': {'level1': 'Stromal', 'level2': 'Fibroblast', 'level3': None},
}


def load_expression_matrix(path: Path) -> pd.DataFrame:
    """Load gzip-compressed expression matrix (genes × cells)."""
    logger.info('Loading expression matrix from %s', path)
    expr = pd.read_csv(path, sep='\t', index_col=0)
    logger.info('Loaded %d genes × %d cells', expr.shape[0], expr.shape[1])
    return expr


def load_metadata(path: Path) -> pd.DataFrame:
    """Load metadata (gzip-mislabeled as .txt). Filter to RA synovium only."""
    logger.info('Loading metadata from %s', path)
    meta = pd.read_csv(path, sep='\t', compression='gzip')

    # Skip header row (TYPE, group)
    meta = meta[meta['NAME'] != 'TYPE'].reset_index(drop=True)

    # RA synovium clusters: SC-T*, SC-M*, SC-B*, SC-F*
    ra_clusters = set(CLUSTER_LABELS.keys())
    meta = meta[meta['Cluster'].isin(ra_clusters)].reset_index(drop=True)

    logger.info('Filtered to RA synovium: %d cells', len(meta))
    return meta


def extract_donor_id(cell_name: str) -> str:
    """Extract donor ID from cell name (prefix before first underscore)."""
    return cell_name.split('_')[0]


def create_anndata(expr: pd.DataFrame, meta: pd.DataFrame) -> ad.AnnData:
    """
    Create AnnData object with expression and metadata.
    Ensure cell names match between expr and meta.
    """
    logger.info('Creating AnnData object')

    # Transpose expr to cells × genes format
    expr_t = expr.T

    # Validate cell names match
    if not (expr_t.index == meta['NAME'].values).all():
        raise ValueError('Cell names do not match between expression and metadata')

    # Create AnnData: cells × genes
    adata = ad.AnnData(X=expr_t.values, var=pd.DataFrame(index=expr.index))
    adata.obs['cell_name'] = meta['NAME'].values
    adata.obs['cluster'] = meta['Cluster'].values
    adata.obs['donor'] = adata.obs['cell_name'].apply(extract_donor_id)

    # Add hierarchical labels from consolidated mapping
    label_df = pd.DataFrame.from_dict(CLUSTER_LABELS, orient='index')
    adata.obs = adata.obs.join(label_df, on='cluster')

    # Keep raw counts
    adata.layers['raw_counts'] = adata.X.copy()

    logger.info('Created AnnData: %d obs × %d var', adata.n_obs, adata.n_vars)
    logger.info('Donors: %d unique', adata.obs['donor'].nunique())
    logger.info('Level 3 counts:')
    logger.info('\n%s', adata.obs['level3'].value_counts())

    return adata


def normalize_and_log(adata: ad.AnnData, target_sum: int = 10_000) -> None:
    """
    Normalize total counts per cell and log1p transform.
    Used only for DE; raw counts preserved in layers['raw_counts'].
    Modifies adata in place.
    """
    logger.info('Normalizing to %d counts/cell and log1p transforming', target_sum)

    # Copy raw counts to X for normalization
    adata.X = adata.layers['raw_counts'].copy()

    # Normalize total
    sc.pp.normalize_total(adata, target_sum=target_sum, inplace=True)

    # Log1p
    sc.pp.log1p(adata)

    logger.info('Normalization complete')


def rank_genes_one_vs_rest(
    level_cells: ad.AnnData, label_col: str, target_label: str, top_n: int = TOP_N_GENES
) -> list[str] | None:
    """
    One-vs-rest Wilcoxon DE for `target_label` against all other labels present
    in `level_cells`. Returns top-n gene symbols (scanpy's score-descending
    order), or None if fewer than 2 distinct labels are present (no valid rest).
    """
    if level_cells.obs[label_col].nunique() < 2:
        return None

    sc.tl.rank_genes_groups(
        level_cells, groupby=label_col, groups=[target_label], reference='rest', method='wilcoxon'
    )
    return list(level_cells.uns['rank_genes_groups']['names'][target_label][:top_n])


def build_level_examples(adata: ad.AnnData, level: int) -> list[dict]:
    """
    Step 3: for each donor x each label present (min MIN_CELLS_BASE cells) at
    this level, one-vs-rest DE restricted to this level's cells -> one example.
    """
    label_col = f'level{level}'
    level_cells = adata[adata.obs[label_col].notna()]
    logger.info('Level %d: %d cells with a label (subset for DE)', level, level_cells.n_obs)

    examples = []
    for donor in sorted(level_cells.obs['donor'].unique()):
        donor_cells = level_cells[level_cells.obs['donor'] == donor]
        label_counts = donor_cells.obs[label_col].value_counts()

        if label_counts.size < 2:
            continue  # no "rest" to contrast against

        for label, n_cells in label_counts.items():
            if n_cells < MIN_CELLS_BASE:
                continue
            genes = rank_genes_one_vs_rest(donor_cells, label_col, label)
            if genes is not None:
                examples.append({'genes': genes, 'label': label, 'donor': donor, 'level': level})

    logger.info('Level %d: %d base examples from per-donor DE', level, len(examples))
    return examples


def augment_level3(level_cells: ad.AnnData, examples: list[dict]) -> list[dict]:
    """
    Step 3 (Level 3 augmentation): Treg/Tph are thin across donors. Bootstrap-
    resample cells within (donor, label) groups and redo DE to reach
    ~LEVEL3_TARGET_PER_LABEL examples per label. Reference ("rest") cells for
    the donor are left unsampled; only the target label's group is resampled.
    """
    label_col = 'level3'
    rng = np.random.default_rng(RANDOM_SEED)
    counts_by_label = pd.Series([e['label'] for e in examples]).value_counts()
    augmented = list(examples)

    for label in level_cells.obs[label_col].dropna().unique():
        n_existing = counts_by_label.get(label, 0)
        n_needed = LEVEL3_TARGET_PER_LABEL - n_existing
        if n_needed <= 0:
            continue

        # candidate (donor, label) groups with enough cells to bootstrap from
        candidates = []
        for donor in sorted(level_cells.obs['donor'].unique()):
            donor_cells = level_cells[level_cells.obs['donor'] == donor]
            if donor_cells.obs[label_col].nunique() < 2:
                continue
            n_cells = (donor_cells.obs[label_col] == label).sum()
            if n_cells >= MIN_CELLS_AUGMENT:
                candidates.append(donor)

        if not candidates:
            logger.warning('Level 3 label %r: no augmentation candidates, staying at %d examples', label, n_existing)
            continue

        n_added = 0
        draws_per_donor = {donor: 0 for donor in candidates}
        while n_added < n_needed and any(d < MAX_AUGMENT_PER_GROUP for d in draws_per_donor.values()):
            for donor in candidates:
                if n_added >= n_needed:
                    break
                if draws_per_donor[donor] >= MAX_AUGMENT_PER_GROUP:
                    continue

                donor_cells = level_cells[level_cells.obs['donor'] == donor].copy()
                target_idx = np.where(donor_cells.obs[label_col].values == label)[0]
                n_cells = len(target_idx)

                # bootstrap-resample the target label's cells; leave rest as-is
                resampled_idx = rng.choice(target_idx, size=n_cells, replace=True)
                keep_mask = donor_cells.obs[label_col].values != label
                keep_mask[target_idx] = False
                bootstrap_cells = donor_cells[keep_mask | np.isin(np.arange(donor_cells.n_obs), resampled_idx)].copy()

                genes = rank_genes_one_vs_rest(bootstrap_cells, label_col, label)
                if genes is not None:
                    augmented.append({'genes': genes, 'label': label, 'donor': donor, 'level': 3})
                    n_added += 1
                draws_per_donor[donor] += 1

        if n_existing + n_added < LEVEL3_TARGET_PER_LABEL:
            logger.warning(
                'Level 3 label %r: reached %d/%d examples (candidates exhausted)',
                label, n_existing + n_added, LEVEL3_TARGET_PER_LABEL,
            )
        logger.info('Level 3 label %r: +%d augmented examples (%d total)', label, n_added, n_existing + n_added)

    return augmented


def _shuffled(items: list, rng: np.random.Generator) -> list:
    items = list(items)
    order = rng.permutation(len(items))
    return [items[i] for i in order]


def split_donors(donors: list[str], adata: ad.AnnData, seed: int = RANDOM_SEED) -> dict[str, str]:
    """
    Seeded shuffle + slice into ~70/15/15 train/val/test, by donor.

    Level 3 labels (Treg/Tph especially) have very few eligible donors overall
    (as few as 3 for Treg) -> a plain random split can leave val or test with
    zero examples for a label, silently breaking downstream macro-F1. So first
    reserve one eligible donor per level-3 label (scarcest label first) into
    test, then val, before randomly filling the rest.
    """
    rng = np.random.default_rng(seed)
    level3_obs = adata.obs[adata.obs['level3'].notna()]
    labels = list(level3_obs['level3'].dropna().unique())
    eligible_by_label = {
        label: level3_obs[level3_obs['level3'] == label].groupby('donor').size().pipe(
            lambda s: s[s >= MIN_CELLS_AUGMENT].index.tolist()
        )
        for label in labels
    }
    labels_by_scarcity = sorted(labels, key=lambda label: len(eligible_by_label[label]))

    donor_split: dict[str, str] = {}
    for label in labels_by_scarcity:
        for split in ('test', 'val'):
            already_has_rep = any(donor_split.get(d) == split for d in eligible_by_label[label])
            if already_has_rep:
                continue
            candidates = _shuffled(
                [d for d in eligible_by_label[label] if d not in donor_split], rng
            )
            if not candidates:
                logger.warning(
                    'Level 3 label %r: no unassigned eligible donor left to reserve for %s split',
                    label, split,
                )
                continue
            donor_split[candidates[0]] = split

    n = len(donors)
    n_train_target = round(n * TRAIN_FRAC)
    n_val_target = round(n * VAL_FRAC)
    n_train_assigned = sum(v == 'train' for v in donor_split.values())
    n_val_assigned = sum(v == 'val' for v in donor_split.values())
    n_train_needed = max(0, n_train_target - n_train_assigned)
    n_val_needed = max(0, n_val_target - n_val_assigned)

    remaining = _shuffled([d for d in donors if d not in donor_split], rng)
    for donor in remaining[:n_train_needed]:
        donor_split[donor] = 'train'
    for donor in remaining[n_train_needed:n_train_needed + n_val_needed]:
        donor_split[donor] = 'val'
    for donor in remaining[n_train_needed + n_val_needed:]:
        donor_split[donor] = 'test'

    logger.info(
        'Donor split: %d train, %d val, %d test',
        sum(v == 'train' for v in donor_split.values()),
        sum(v == 'val' for v in donor_split.values()),
        sum(v == 'test' for v in donor_split.values()),
    )
    return donor_split


def save_examples(
    examples: list[dict], donor_split: dict[str, str], level: int, output_dir: Path = OUTPUT_DIR
) -> None:
    """Write data/examples_{level}_{split}.jsonl, grouped by each example's donor's split."""
    by_split: dict[str, list[dict]] = {'train': [], 'val': [], 'test': []}
    for example in examples:
        by_split[donor_split[example['donor']]].append(example)

    for split, split_examples in by_split.items():
        path = output_dir / f'examples_{level}_{split}.jsonl'
        with open(path, 'w') as f:
            for example in split_examples:
                f.write(json.dumps(example) + '\n')

        label_counts = pd.Series([e['label'] for e in split_examples]).value_counts().to_dict()
        logger.info('Level %d %s: %d examples, by label: %s', level, split, len(split_examples), label_counts)


def main() -> ad.AnnData:
    """Main pipeline: load, map labels, normalize."""
    # Load data
    expr = load_expression_matrix(EXPR_PATH)
    meta = load_metadata(META_PATH)

    # Create AnnData
    adata = create_anndata(expr, meta)

    # Normalize for DE (keep raw in layers)
    normalize_and_log(adata)

    # Quick QC
    sparsity = 1 - (adata.X != 0).sum() / adata.X.size
    logger.info('Expression matrix sparsity: %.2f%%', sparsity * 100)
    logger.info('Cells per donor:')
    logger.info('\n%s', adata.obs.groupby('donor').size().describe())

    # Save checkpoint
    output_path = OUTPUT_DIR / 'adata_labeled.h5ad'
    adata.write_h5ad(output_path)
    logger.info('Saved checkpoint to %s', output_path)

    # Step 3-4: per-donor DE examples, Level 3 augmentation, donor split, jsonl export
    donor_split = split_donors(sorted(adata.obs['donor'].unique()), adata)

    for level in (1, 2, 3):
        examples = build_level_examples(adata, level)
        if level == 3:
            examples = augment_level3(adata[adata.obs['level3'].notna()], examples)
        save_examples(examples, donor_split, level)

    # Gotcha (build-plan.md): assert no donor appears in more than one split
    donors_by_split = {}
    for level in (1, 2, 3):
        for split in ('train', 'val', 'test'):
            path = OUTPUT_DIR / f'examples_{level}_{split}.jsonl'
            file_donors = {json.loads(line)['donor'] for line in path.read_text().splitlines()}
            donors_by_split.setdefault(split, set()).update(file_donors)

    assert not (donors_by_split['train'] & donors_by_split['val']), 'donor leaked between train/val'
    assert not (donors_by_split['train'] & donors_by_split['test']), 'donor leaked between train/test'
    assert not (donors_by_split['val'] & donors_by_split['test']), 'donor leaked between val/test'
    logger.info('No donor split leakage across train/val/test.')

    return adata


if __name__ == '__main__':
    adata = main()
