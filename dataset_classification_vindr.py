import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Label mapping
BIRADS_MAP = {
    'BI-RADS 1': 0,
    'BI-RADS 2': 1,
    'BI-RADS 3': 2,
    'BI-RADS 4': 3,
    'BI-RADS 5': 4,
}

DENSITY_MAP = {
    'DENSITY A': 0,
    'DENSITY B': 1,
    'DENSITY C': 2,
    'DENSITY D': 3,
}


class MakeDataset_VinDr_classification(Dataset):
    """
    Dataset class untuk VinDr-Mammo Multi-Task Classification.
    Setiap sample adalah pasangan ipsilateral (CC, MLO) dari satu sisi payudara.
    Return: (img_cc, img_mlo, label_birads, label_density)
    """

    def __init__(self, image_dir, label_dir_csv, transform=None,
                 mode='train', split_size=0.2, target_size=224):
        """
        Args:
            image_dir   : Path ke folder dataset_preprocessed
            label_dir_csv: Path ke breast-level_annotations.csv
            transform   : Torchvision transforms
            mode        : 'train', 'val', atau 'test'
            split_size  : Proporsi val dari training split (0.2 = 20%)
            target_size : Ukuran gambar setelah resize
        """
        self.image_dir   = image_dir
        self.transform   = transform
        self.target_size = target_size
        self.mode        = mode

        df = pd.read_csv(label_dir_csv)

        # Pisah berdasarkan kolom split di CSV
        # Kolom split di CSV sudah 80:20 (training=train, test=test)
        if mode == 'train':
            df_split = df[df['split'] == 'training'].reset_index(drop=True)
        else:  # test
            df_split = df[df['split'] == 'test'].reset_index(drop=True)

        self.pairs = self._make_pairs(df_split)

        print(f"[{mode.upper()}] Total pasangan ipsilateral: {len(self.pairs)}")

    def _make_pairs(self, df):
        """
        Membentuk pasangan (CC, MLO) ipsilateral dari dataframe.
        Pasangan dikelompokkan berdasarkan (study_id, laterality).
        Pasangan yang tidak lengkap (tidak ada CC atau MLO) diabaikan.
        """
        pairs = []
        grouped = df.groupby(['study_id', 'laterality'])

        for (study_id, laterality), group in grouped:
            cc_rows  = group[group['view_position'] == 'CC']
            mlo_rows = group[group['view_position'] == 'MLO']

            # Harus ada tepat 1 CC dan 1 MLO
            if len(cc_rows) != 1 or len(mlo_rows) != 1:
                continue

            cc_row  = cc_rows.iloc[0]
            mlo_row = mlo_rows.iloc[0]

            # Ambil label dari CC row (CC dan MLO harus punya label yang sama)
            birads_str  = str(cc_row['breast_birads']).strip()
            density_str = str(cc_row['breast_density']).strip()

            # Skip jika label tidak dikenal
            if birads_str not in BIRADS_MAP or density_str not in DENSITY_MAP:
                continue

            label_birads  = BIRADS_MAP[birads_str]
            label_density = DENSITY_MAP[density_str]

            cc_path  = os.path.join(self.image_dir, study_id, cc_row['image_id']  + '.png')
            mlo_path = os.path.join(self.image_dir, study_id, mlo_row['image_id'] + '.png')

            # Skip jika file tidak ada
            if not os.path.exists(cc_path) or not os.path.exists(mlo_path):
                continue

            pairs.append({
                'cc_path'      : cc_path,
                'mlo_path'     : mlo_path,
                'label_birads' : label_birads,
                'label_density': label_density,
            })

        return pairs

    def _load_image(self, path):
        """Load gambar grayscale (sudah di-preprocess 224x224) lalu konversi ke RGB (3 channel)."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Gambar tidak ditemukan: {path}")
        # Gambar sudah berukuran target_size x target_size dari preprocessing
        # Resize hanya sebagai fallback jika ukuran tidak sesuai
        if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
            img = cv2.resize(img, (self.target_size, self.target_size))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # (H, W, 3)
        return img

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        img_cc  = self._load_image(pair['cc_path'])
        img_mlo = self._load_image(pair['mlo_path'])

        # Konversi ke PIL untuk kompatibilitas dengan torchvision transforms
        from PIL import Image
        img_cc  = Image.fromarray(img_cc)
        img_mlo = Image.fromarray(img_mlo)

        if self.transform:
            # Augmentasi diterapkan sama pada CC dan MLO
            # Gunakan seed yang sama agar konsisten
            seed = torch.randint(0, 2**32, (1,)).item()

            torch.manual_seed(seed)
            img_cc = self.transform(img_cc)

            torch.manual_seed(seed)
            img_mlo = self.transform(img_mlo)
        else:
            import torchvision.transforms.functional as TF
            img_cc  = TF.to_tensor(img_cc)
            img_mlo = TF.to_tensor(img_mlo)

        label_birads  = torch.tensor(pair['label_birads'],  dtype=torch.long)
        label_density = torch.tensor(pair['label_density'], dtype=torch.long)

        return img_cc, img_mlo, label_birads, label_density
