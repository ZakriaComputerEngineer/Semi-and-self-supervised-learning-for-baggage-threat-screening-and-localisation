import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split


def mask_transform_fn(mask):
    """Transforms the mask by converting it into a tensor of 0s and 1s."""
    return torch.tensor(np.array(mask) > 0, dtype=torch.float32)


class ImageMaskDataset(Dataset):
    def __init__(self, root_dir, image_size=(256, 256)):
        """
        Dataset class for loading images and corresponding masks.
        Args:
        - root_dir (str): Root directory containing images and masks
        - image_size (tuple[int, int]): Tuple specifying image resize dimensions
        """
        assert os.path.exists(
            root_dir), f"Error: Directory '{root_dir}' does not exist."

        self.image_paths, self.mask_paths = self._load_paths(root_dir)
        self.valid_pairs = self._validate_pairs()
        print(f"Found {len(self.valid_pairs)} valid image-mask pairs")

        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Lambda(mask_transform_fn)
        ])

    def _load_paths(self, root_dir):
        """
        Load paths for images and their corresponding masks from multiple dataset directories.

        Args:
            root_dir (str): Root directory containing multiple dataset folders

        Returns:
            tuple[list, list]: Lists of image and mask file paths
        """
        image_paths = []
        mask_paths = []

        # Get all dataset directories
        dataset_dirs = [d for d in os.listdir(
            root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        for dataset_dir in dataset_dirs:
            dataset_path = os.path.join(root_dir, dataset_dir)
            images_dir = os.path.join(dataset_path, 'images')
            masks_dir = os.path.join(dataset_path, 'masks')

            # Skip if images or masks directory doesn't exist
            if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
                print(
                    f"Warning: Skipping {dataset_dir} - missing images or masks directory")
                continue

            # Get list of image files
            image_files = sorted([f for f in os.listdir(images_dir)
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

            # Get list of mask files
            mask_files = sorted([f for f in os.listdir(masks_dir)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

            # Verify matching numbers of images and masks
            if len(image_files) != len(mask_files):
                print(f"Warning: Skipping {dataset_dir} - number of images ({len(image_files)}) "
                      f"does not match number of masks ({len(mask_files)})")
                continue

            # Add full paths to lists
            image_paths.extend([os.path.join(images_dir, f)
                               for f in image_files])
            mask_paths.extend([os.path.join(masks_dir, f) for f in mask_files])

            print(f"Added {len(image_files)} pairs from {dataset_dir}")

        if not image_paths:
            raise ValueError(f"No valid image-mask pairs found in {root_dir}")

        print(
            f"Found total of {len(image_paths)} image-mask pairs across all datasets")
        return image_paths, mask_paths

    def _validate_image(self, image_path):
        """Validate if an image file is readable"""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except:
            print(f"Warning: Corrupted or unreadable image file: {image_path}")
            return False

    def _validate_pairs(self):
        """Validate all image-mask pairs and return only valid ones"""
        valid_pairs = []
        for img_path, mask_path in tqdm(zip(self.image_paths, self.mask_paths),
                                        desc="Validating image-mask pairs",
                                        total=len(self.image_paths)):
            if self._validate_image(img_path) and self._validate_image(mask_path):
                valid_pairs.append((img_path, mask_path))
            else:
                print(
                    f"Skipping corrupted pair:\nImage: {img_path}\nMask: {mask_path}\n")
        return valid_pairs

    def __getitem__(self, idx):
        """
        Return a single image-mask pair.
        Args:
        - idx (int): Index of the desired image-mask pair
        Returns:
        - tuple[torch.Tensor, torch.Tensor]: Transformed image and mask
        """
        try:
            image_path, mask_path = self.valid_pairs[idx]

            # Load and convert image
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_tensor = self.image_transform(image)

            # Load and convert mask
            with Image.open(mask_path) as mask:
                mask = mask.convert("L")
                mask_tensor = self.mask_transform(mask)

            return image_tensor, mask_tensor

        except Exception as e:
            print(
                f"Error loading pair {idx}:\nImage: {image_path}\nMask: {mask_path}\nError: {str(e)}")
            # Return a zero tensor of appropriate size as fallback
            return torch.zeros(3, 256, 256), torch.zeros(256, 256)

    def __len__(self):
        return len(self.valid_pairs)


class DatasetLoader:
    def __init__(self, image_size, root_dir, split_ratios=[0.7, 0.15, 0.15], seed=42,
                 Shuffle=True, batch_size=16, num_workers=2, task='pretext'):
        """
        Initialize DatasetLoader with optional augmentation.

        Args:
            augment (string): Whether to apply augmentation to the dataset depeding on the task type (pretext or downstream)
        """
        self.image_size = image_size
        self.task = task

        if task == 'pretext':
            # Perform augmentation before creating the dataset
            self.augment_dataset(root_dir)
            # Use augmented directory for dataset
            self.dataset = ImageMaskDataset(
                os.path.join(root_dir, 'augmented'), (image_size, image_size))
        else:
            self.dataset = ImageMaskDataset(root_dir, (image_size, image_size))

        self.split_ratios = split_ratios
        self.seed = seed
        self.shuffle_data = Shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

    def augment_dataset(self, root_dir):
        """Apply augmentation to the dataset."""
        # Create augmented directory
        augmented_dir = os.path.join(root_dir, 'augmented')
        os.makedirs(augmented_dir, exist_ok=True)

        # Parameters for augmentation
        angles = [45, 90, 270]
        scale_factors = [0.7, 1.4]
        shift_values = [(25, 0), (-25, 0), (0, 25), (0, -25)]
        flip_codes = [0, 1, -1]
        target_resolution = (768, 576)

        for dataset_dir in os.listdir(root_dir):
            dataset_path = os.path.join(root_dir, dataset_dir)
            if not os.path.isdir(dataset_path) or dataset_dir == 'augmented':
                continue

            images_dir = os.path.join(dataset_path, 'images')
            masks_dir = os.path.join(dataset_path, 'masks')

            if not (os.path.exists(images_dir) and os.path.exists(masks_dir)):
                continue

            # Create output directories for augmented data
            aug_dataset_dir = os.path.join(augmented_dir, dataset_dir)
            aug_images_dir = os.path.join(aug_dataset_dir, 'images')
            aug_masks_dir = os.path.join(aug_dataset_dir, 'masks')
            os.makedirs(aug_images_dir, exist_ok=True)
            os.makedirs(aug_masks_dir, exist_ok=True)

            # Copy original images and masks
            for img_name in os.listdir(images_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Load image and mask
                    img_path = os.path.join(images_dir, img_name)
                    mask_path = os.path.join(masks_dir, img_name)

                    if not os.path.exists(mask_path):
                        continue

                    image = cv2.imread(img_path)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                    # Save original
                    cv2.imwrite(os.path.join(aug_images_dir, img_name), image)
                    cv2.imwrite(os.path.join(aug_masks_dir, img_name), mask)

                    # Rotations
                    for angle in angles:
                        rotated_img, rotated_mask = self.rotate_image_and_mask(
                            image, mask, angle, target_resolution)
                        cv2.imwrite(os.path.join(aug_images_dir,
                                                 f"{os.path.splitext(img_name)[0]}_rot{angle}.jpg"), rotated_img)
                        cv2.imwrite(os.path.join(aug_masks_dir,
                                                 f"{os.path.splitext(img_name)[0]}_rot{angle}.png"), rotated_mask)

                    # Scaling
                    for scale in scale_factors:
                        scaled_img, scaled_mask = self.scale_image_and_mask(
                            image, mask, scale, target_resolution)
                        cv2.imwrite(os.path.join(aug_images_dir,
                                                 f"{os.path.splitext(img_name)[0]}_scale{scale}.jpg"), scaled_img)
                        cv2.imwrite(os.path.join(aug_masks_dir,
                                                 f"{os.path.splitext(img_name)[0]}_scale{scale}.png"), scaled_mask)

                    # Shifts
                    for shift_x, shift_y in shift_values:
                        shifted_img, shifted_mask = self.shift_image_and_mask(
                            image, mask, shift_x, shift_y, target_resolution)
                        cv2.imwrite(os.path.join(aug_images_dir,
                                                 f"{os.path.splitext(img_name)[0]}_shift{shift_x}_{shift_y}.jpg"), shifted_img)
                        cv2.imwrite(os.path.join(aug_masks_dir,
                                                 f"{os.path.splitext(img_name)[0]}_shift{shift_x}_{shift_y}.png"), shifted_mask)

                    # Flips
                    for flip_code in flip_codes:
                        flipped_img, flipped_mask = self.flip_image_and_mask(
                            image, mask, flip_code, target_resolution)
                        cv2.imwrite(os.path.join(aug_images_dir,
                                                 f"{os.path.splitext(img_name)[0]}_flip{flip_code}.jpg"), flipped_img)
                        cv2.imwrite(os.path.join(aug_masks_dir,
                                                 f"{os.path.splitext(img_name)[0]}_flip{flip_code}.png"), flipped_mask)

    @staticmethod
    def rotate_image_and_mask(image, mask, angle, target_resolution):
        # Implementation of rotate_image_and_mask function
        if angle == 90:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            rotated_mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
            rotated_mask = cv2.rotate(mask, cv2.ROTATE_180)
        elif angle == 270:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rotated_mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 45:
            (h, w) = image.shape[:2]
            diag_length = int(np.sqrt(h**2 + w**2))
            padding = (diag_length - h) // 2

            padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding,
                                              borderType=cv2.BORDER_REPLICATE)
            padded_mask = cv2.copyMakeBorder(mask, padding, padding, padding, padding,
                                             borderType=cv2.BORDER_REPLICATE)

            (ph, pw) = padded_image.shape[:2]
            center = (pw // 2, ph // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(padded_image, M, (pw, ph),
                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            rotated_mask = cv2.warpAffine(padded_mask, M, (pw, ph),
                                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            start_x, start_y = (pw - w) // 2, (ph - h) // 2
            rotated_image = rotated_image[start_y:start_y +
                                          h, start_x:start_x + w]
            rotated_mask = rotated_mask[start_y:start_y +
                                        h, start_x:start_x + w]
        else:
            rotated_image = image.copy()
            rotated_mask = mask.copy()

        resized_image = cv2.resize(
            rotated_image, target_resolution, interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(
            rotated_mask, target_resolution, interpolation=cv2.INTER_NEAREST)

        return resized_image, resized_mask

    @staticmethod
    def scale_image_and_mask(image, mask, scale_factor, target_resolution):
        scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor,
                                  interpolation=cv2.INTER_LINEAR)
        scaled_mask = cv2.resize(mask, None, fx=scale_factor, fy=scale_factor,
                                 interpolation=cv2.INTER_NEAREST)

        resized_image = cv2.resize(
            scaled_image, target_resolution, interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(
            scaled_mask, target_resolution, interpolation=cv2.INTER_NEAREST)

        return resized_image, resized_mask

    @staticmethod
    def shift_image_and_mask(image, mask, shift_x, shift_y, target_resolution):
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        shifted_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        resized_image = cv2.resize(
            shifted_image, target_resolution, interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(
            shifted_mask, target_resolution, interpolation=cv2.INTER_NEAREST)

        return resized_image, resized_mask

    @staticmethod
    def flip_image_and_mask(image, mask, flip_code, target_resolution):
        flipped_image = cv2.flip(image, flip_code)
        flipped_mask = cv2.flip(mask, flip_code)

        resized_image = cv2.resize(
            flipped_image, target_resolution, interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(
            flipped_mask, target_resolution, interpolation=cv2.INTER_NEAREST)

        return resized_image, resized_mask

    def get_dataloaders(self):
        """Split dataset and return DataLoaders for train, val, and test."""
        # Rest of the method remains unchanged
        dataset_size = len(self.dataset)
        train_size = int(dataset_size * self.split_ratios[0])
        val_size = int(dataset_size * self.split_ratios[1])
        test_size = dataset_size - train_size - val_size

        generator = torch.Generator().manual_seed(self.seed)

        if self.shuffle_data:
            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset, [train_size, val_size, test_size], generator=generator)
        else:
            train_dataset = torch.utils.data.Subset(
                self.dataset, range(0, train_size))
            val_dataset = torch.utils.data.Subset(
                self.dataset, range(train_size, train_size + val_size))
            test_dataset = torch.utils.data.Subset(
                self.dataset, range(train_size + val_size, dataset_size))

        print("Creating DataLoaders...")
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader
