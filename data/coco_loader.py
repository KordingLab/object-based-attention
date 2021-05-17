from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from collections import defaultdict
import torch
import torchvision
import pickle
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple, List

coco = None

def get_filtered_metadata_deprecated(savefile = "metadata.p"):
    #get all image IDs
    img_ids = coco.getImgIds()

    #get a list of annotations
    list_of_anns = [coco.loadAnns(coco.getAnnIds(imgIds = i, iscrowd = False)) for i in img_ids]
    list_of_anns = [a for a in list_of_anns if len(a) > 0]

    
    filtered_images = []
    filtered_annotations = []
    for i, anns in enumerate(list_of_anns):
        img_metadata = coco.loadImgs(anns[0]["image_id"])[0]
        img_size = img_metadata["height"] * img_metadata["width"]
        filtered_anns = [ann for ann in anns if filter_objects_by_annotation(ann, img_size)]

        #sort by area
        filtered_anns = sorted(filtered_anns, key = lambda x: -x["area"])

        if i % 500 == 0: 
            print("[%d\% Complete]\t ITEMS FOUND:%s"%((i/len(list_of_anns))*100, len(filtered_images)))
        
        if len(filtered_anns) > 0: 
            final_anns = []
            accumulated_mask = np.zeros_like(coco.annToMask(filtered_anns[0]))
            for ann in filtered_anns: 
                mask = coco.annToMask(ann)
                accumulated_mask += mask
                if np.sum(accumulated_mask > 1) > 0.01* img_size:
                    break
                else: 
                    final_anns.append(ann)

            filtered_annotations.append(final_anns)
            filtered_images.append(final_anns[0]["image_id"])
    
    metadata = {"images": filtered_images, "annotations": filtered_annotations}
    pickle.dump(metadata, open("%s"%(savefile), "wb"))
    return filtered_images, filtered_annotations

def get_filtered_metadata(savefile = "metadata.p"):
    #get all image IDs
    img_ids = coco.getImgIds()

    #get a list of annotations
    list_of_anns = [coco.loadAnns(coco.getAnnIds(imgIds = i, iscrowd = False)) for i in img_ids]
    list_of_anns = [a for a in list_of_anns if len(a) > 0]

    
    filtered_images = []
    filtered_annotations = []
    for i, anns in enumerate(list_of_anns):
        img_metadata = coco.loadImgs(anns[0]["image_id"])[0]
        img_size = img_metadata["height"] * img_metadata["width"]
        filtered_anns = [ann for ann in anns if filter_objects_by_annotation(ann, img_size)]

        #sort by area
        filtered_anns = sorted(filtered_anns, key = lambda x: -x["area"])

        if i % 500 == 0: 
            print("ITERATION: %s\t ITEMS:%s"%(i, len(filtered_images)))
        
        if len(filtered_anns) >= 2: 
            final_anns = []
            accumulated_mask = np.zeros_like(coco.annToMask(filtered_anns[0]))
            for ann in filtered_anns: 
                mask = coco.annToMask(ann)
                accumulated_mask += mask
                if np.sum(accumulated_mask > 1) > 0.01* img_size:
                    break
                else: 
                    final_anns.append(ann)
                    
            final_anns = final_anns[:2]
            if len(final_anns) == 2: 
                filtered_annotations.append(final_anns)
                filtered_images.append(final_anns[0]["image_id"])
    
    metadata = {"images": filtered_images, "annotations": filtered_annotations}
    pickle.dump(metadata, open("%s"%(savefile), "wb"))
    return filtered_images, filtered_annotations

def center_pad(img, fill = 0):
    '''
    Center and pad an image to conform 
    to a square input size
    '''
    height = img.shape[0]
    width = img.shape[1]
    if height > width:
        diff = height - width
        new_img = np.ones((height, height, 3)) * fill
        new_img[:, int(diff/2) : int(diff/2) + width, :] = img

    else: 
        diff = width - height
        new_img = np.ones((width, width, 3)) * fill
        new_img[int(diff/2): int(diff/2) + height, :, :] = img
    
    return new_img

def get_category_counts(indices, filtered_ids, filtered_annotations, maxobjects = None): 
    '''
    Given a list of indices, return the 
    count of each category and supercategory present
    '''
    cat_count = defaultdict(lambda: 0)
    supercat_count = defaultdict(lambda: 0)

    for i in indices:
        index = filtered_ids[i]
        anns = filtered_annotations[i]
        
        for ann in anns:
            target = ann["category_id"]

            category = coco.cats[target]["name"]
            supercategory = coco.cats[target]["supercategory"]

            cat_count[category] += 1
            supercat_count[supercategory] += 1
    
    return cat_count, supercat_count

def display_dict(d, percent = False): 
    padlen = 20
    if percent:
        total = sum([d[k] for k in d.keys()])
        
        for k in sorted(d.keys()):
            if len(k) < padlen: 
                newk = k + " "*(padlen - len(k))
            print("%s %.4f"%(newk, d[k]/total))
    else: 
        for k in sorted(d.keys()): 
            if len(k) < padlen: 
                newk = k + " "*(padlen - len(k))
            print(newk, d[k])

#a function to weight the class labels by prevalence
def class_weight(d, normalize = True): 
    weights = defaultdict(lambda: 1)
    total = sum([d[k] for k in d.keys()])
    for k in d.keys():
        
        weights[k] = total/d[k]
    
    totalweights = sum([weights[k] for k in weights.keys()])
    
    if normalize: 
        final_weights = {k: weights[k]/totalweights for k in weights.keys()}
    else: 
        final_weights = weights
        
    return final_weights

def get_dicts(): 
    '''
    Returns: a dictionary of dictionaries containing
    handy conversions from category (str) to label (int)
    to supercategory (str) to superlabel(int)
    '''
    cat_to_label = {}
    label_to_cat = {}
    label_to_superlabel = {}
    cat_to_superlabel = {}
    supercategories = sorted(list(set(v["supercategory"] for v in coco.cats.values())))
    supercat_to_superlabel = {supercategories[i]:i for i in range(len(supercategories))}
    superlabel_to_supercat = {i:supercategories[i] for i in range(len(supercategories))}
    l_to_top10 = {1: 0, 3: 1, 10: 2, 31: 3, 44: 4, 47: 5, 51: 6, 62: 7, 67: 8, 84: 9}
    top10_to_l = {0: 1, 1: 3, 2: 10, 3: 31, 4: 44, 5: 47, 6: 51, 7: 62, 8: 67, 9: 84}

    
    for k, v in coco.cats.items():
        cat_to_label[v["name"]] = k
        label_to_cat[k] = v["name"]
        cat_to_superlabel[v["name"]] = supercat_to_superlabel[v["supercategory"]]
        label_to_superlabel[k] = supercat_to_superlabel[v["supercategory"]]
        
    dicts = {"cat_to_l": cat_to_label, 
             "cat_to_sl": cat_to_superlabel, 
             "l_to_cat": label_to_cat,
             "l_to_sl": label_to_superlabel,
             "supercat_to_sl": supercat_to_superlabel,
             "sl_to_supercat": superlabel_to_supercat,
             "l_to_top10": l_to_top10, 
             "top10_to_l": top10_to_l
            }
    
    return dicts


class CustomCocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        coco,
        ids,
        anns,
        strength = 0.8,
        maxobjects = None,
        use_supercategory = False,
        use_top10 = False,
        use_masks = False, 
        use_locations = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root)
        self.coco = coco
        self.ids = ids
        self.anns = anns
        self.strength = strength
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.maxobjects = maxobjects
        self.use_supercategory = use_supercategory
        self.use_top10 = use_top10
        self.use_masks = use_masks
        self.use_locations = use_locations
        self.dicts = get_dicts()

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = np.array(self._load_image(id))
        image_size = image.shape[0] * image.shape[1]
        image_height = image.shape[0]
        image_width = image.shape[1]   
        
        anns = self.anns[index]
        
        masked_images = []
        targets = []  
        
        #change to anns if you want all images; here
        #we just select the largest 2
        for ann in anns:
            #get object segmentation mask
            mask = coco.annToMask(ann)
            
            #resize for RGB
            mask = np.stack(np.array([mask]*3), 2)
            
            #multiply
            masked_image = image * mask
            
            if self.use_locations:
                x, y, w, h = ann["bbox"]

                centerx = x + w/2
                centery = y + h/2

                centerx = centerx / image_width
                centery = centery / image_height

                target = np.array([centerx, centery])

                if self.target_transform is not None: 
                    target = self.target_transform(target)
            
            else:
                target = ann["category_id"]

                if self.use_supercategory: 
                    target = self.dicts["l_to_sl"][target]
                elif self.use_top10: 
                    target = self.dicts["l_to_top10"][target]

                if self.target_transform is not None:
                    target = self.target_transform(target)

                if self.transforms is not None: 
                    target = self.transforms(target)
                
            targets.append(target)
            masked_images.append(masked_image)
            
            
        
        new_masked_images = []
        for m in masked_images: 
            new_im = ((1 - self.strength) * center_pad(image, fill = 127) + self.strength * center_pad(m)).astype(np.uint8)
            new_im = Image.fromarray(new_im)
            
            if self.transform is not None:
                new_im = self.transform(new_im)
                
            if self.transforms is not None: 
                new_im = self.transforms(new_im)
                
            new_masked_images.append(new_im)
        
        image = center_pad(image, fill = 127).astype(np.uint8)
        image = Image.fromarray(image)
        
        if self.transform is not None: 
            image = self.transform(image)
        
        if self.transforms is not None: 
            image = self.transforms(image)
            
#         new_masked_images = [n/255.0 for n in new_masked_images]
#         image = image/255.0

        return image, new_masked_images, targets

    def __len__(self) -> int:
        return len(self.ids)
    
    
def filter_objects_by_annotation(annotation, img_size, threshold = 0.1):
    '''
    Given a COCO annotation and image size, 
    Return True iff the area of the annotated object is > the threshold
    '''
    ratio = annotation["area"] / img_size
    return (ratio > threshold)

def get_filtered_deprecated(): 
    '''
    Return a list of images which meet the criteria determined by
    filter_objects_by_annotation
    '''
    #get all image IDs
    img_ids = coco.getImgIds()

    #get a list of annotations
    list_of_anns = [coco.loadAnns(coco.getAnnIds(imgIds = i, iscrowd = False)) for i in img_ids]
    list_of_anns = [a for a in list_of_anns if len(a) > 0]

    filtered_images = []
    filtered_annotations = []
    for anns in list_of_anns:
        img_metadata = coco.loadImgs(anns[0]["image_id"])[0]
        img_size = img_metadata["height"] * img_metadata["width"]
        filtered_anns = [ann for ann in anns if filter_objects_by_annotation(ann, img_size)]

        if len(filtered_anns) >= 2:
            filtered_images.append(anns[0]["image_id"])
            filtered_annotations.append(filtered_anns)
    
    return filtered_images, filtered_annotations

def get_data(root, annfile, 
             metadatafile, 
             size = (100, 100), 
             strength = 0.5, 
             use_supercategory= False, 
             use_top10 = False, 
             use_masks = False,
             use_locations = False,
            ):
    global coco
    
    coco = COCO(annfile)

    if not os.path.exists(metadatafile):
        print(metadatafile, " not found. Generating Metadata File Now.")
        print("Please be patient, this may take a few minutes.")
        get_filtered_metadata(metadatafile)

    metadata = pickle.load(open(metadatafile, "rb"))
    filtered_ids = metadata["images"]
    filtered_annotations = metadata["annotations"]

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size), torchvision.transforms.ToTensor()])
    coco_dataset = CustomCocoDetection(root, annfile, coco, filtered_ids, filtered_annotations, strength = strength, transform = transform, use_supercategory = use_supercategory, use_top10 = use_top10, use_masks = use_masks, use_locations = use_locations)
    return coco_dataset, metadata


def get_train_val_split(coco_dataset, split = 0.7):
    len_dataset = len(coco_dataset)
    len_train = int(split*len_dataset)
    len_val = len_dataset - len_train
    train, val = torch.utils.data.random_split(coco_dataset, [len_train, len_val])
    return train, val

