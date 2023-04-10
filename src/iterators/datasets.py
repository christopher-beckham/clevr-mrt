import glob
import os
from typing import List, Dict
import numpy as np
import torch
import pickle

# import h5py
import json
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import h5py

from .utils import load_vocab

from .. import setup_logger
logger = setup_logger.get_logger()


class BlankDataset(Dataset):
    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return 1


CAM_NAMES = [
    "cam1",
    "cam5",
    "cam19",
    "cam7",
    "cam3",
    "cam16",
    "cam18",
    "cam6",
    "cam14",
    "cam17",
    "cam13",
    "cam8",
    "cc",
    "cam2",
    "cam15",
    "cam10",
    "cam12",
    "cam0",
    "cam4",
    "cam9",
    "cam11",
]


def construct_id_to_scene(subfolders):
    id_to_scene = {}
    n_questions = 0
    for subfolder in subfolders:
        q_file = "%s/questions.json" % subfolder
        s_file = "%s/scenes.json" % subfolder
        if not os.path.exists(q_file) or not os.path.exists(s_file):
            print("ERROR: skip:", subfolder)
            continue
        q_json = json.loads(open(q_file).read())
        s_json = json.loads(open(s_file).read())
        n_questions += len(q_json["questions"])
        # Collect scenes first.
        for idx, scene in enumerate(s_json["scenes"]):
            # Add subfolder to scene dict
            for key in scene:
                scene[key]["subfolder"] = os.path.basename(subfolder)
            this_scene_cc = scene["cc"]
            # e.g. 's002400'
            this_basename = this_scene_cc["image_filename"].split("_")[-2]
            # Map the basename e.g. s002400
            # to its dictionary of camera views.
            id_to_scene[this_basename] = scene
    return id_to_scene


class ClevrKiwiDataset(Dataset):
    def __init__(
        self,
        root_images: str,
        root_meta: str,
        transforms_: List,
        mode: str = "train",
        canonical_mode: str = "train_only",
        id_to_scene: Dict = None,
    ):
        """Dataset for CLEVR-MRT.

        When the dataset is first constructed, a cache file is generated in the
        same directory as `root_meta`, either `cache.pkl` for train/valid set or
        `cache_test.pkl` for the test set. In subsequent constructions of this
        dataset the pkl file will be loaded instead from disk. Therefore, if
        the contents of the dataset have changed in any way, this cache file
        should be deleted.

        Args:
            root_images: directory to the images of the dataset.
            root_meta:  directory to the metadata of the dataset.
            transforms_: a list of torchvision transforms.
              Defaults to None.
            mode: either 'train', 'val', or 'test'.
              Defaults to 'train'.
            canonical_mode: Accepts one of three options: 'b0' for only canonical views
              (like the original clevr dataset), 'none' for no canonical views, and
              'train_only' only provides canonical views in the training set.
              Defaults to 'train_only'.
            id_to_scene: To avoid generating a cache file or loading one in  from disk,
              you can just specify the id_to_scene mapping here if it is already in memory.
              Defaults to None.
        """

        CANONICAL_MODE_SUPPORTED = ["b0", "none", "train_only"]
        if canonical_mode not in CANONICAL_MODE_SUPPORTED:
            # b0 = canonical view only (like vanilla clevr)
            # none = no canonical views at all
            # train_only = canonical views included in train, eval no
            raise Exception(
                "Canonical mode not recognised. Must choose "
                + "from one of the following: {}".format(CANONICAL_MODE_SUPPORTED)
            )

        subfolders = glob.glob("%s/*" % root_images)

        self.root_images = root_images
        self.root_meta = root_meta
        self.transform = transforms.Compose(transforms_)
        self.canonical_mode = canonical_mode

        self.vocab = load_vocab("%s/vocab.json" % root_meta)

        if mode not in ["train", "val", "test"]:
            raise Exception("mode must be either train or val or test (got %s)" % mode)
        if mode in ['train', 'val']:
            self.mode_str = "train-val"
        else:
            self.mode_str = "test"
        self.mode = mode

        # This holds every question and for all intents
        # and purposes is the _length_ of this dataset.
        # In order to map a question to its scene we
        # must parse its filename and use id_to_scene
        # in order to go from question to camera views.
        if mode == "train":
            h5 = h5py.File("%s/train_questions.h5" % root_meta, "r")
        elif mode == "val":
            h5 = h5py.File("%s/valid_questions.h5" % root_meta, "r")
        else:
            h5 = h5py.File("%s/test_questions.h5" % root_meta, "r")

        self.answers = h5["answers"][:]
        self.image_filenames = [x.decode("utf-8") for x in h5["image_filenames"][:]]
        # self.template_filenames = [x.decode('utf-8') for x in h5['template_filenames'][:] ]
        self.questions = h5["questions"][:]
        self.question_strs = h5["question_strs"][:]

        assert len(self.answers) == len(self.image_filenames) == len(self.questions)

        if id_to_scene is None:
            if mode in ["train", "val"]:
                cache_file = "%s/cache.pkl" % root_meta
            else:
                cache_file = "%s/cache_test.pkl" % root_meta

            if not os.path.exists(cache_file):
                logger.debug("Cannot find %s, so generating it..." % cache_file)
                self.id_to_scene = construct_id_to_scene(subfolders)
                logger.debug("Writing cache to: %s" % root_meta)
                logger.debug("(NOTE: if you change the dataset, delete the cache file")
                with open(cache_file, "wb") as f_write:
                    pickle.dump(self.id_to_scene, f_write)
            else:
                with open(cache_file, "rb") as f_read:
                    logger.debug("Loading cache file: {}".format(cache_file))
                    self.id_to_scene = pickle.load(f_read)
        else:
            logger.debug("Cache object specified from `id_to_scene` argument...")
            self.id_to_scene = id_to_scene

        self.mode = mode

        logger.info(
            "root_images={}, root_meta={}, mode={}".format(root_images, root_meta, mode)
        )
        logger.info("  # of questions: {}".format(len(self.questions)))
        # self.image_filename is all cc's, but let it denote the 'scene'
        logger.info("  # of unique scenes: {}".format(len(set(self.image_filenames))))

    def __getitem__(self, index):
        # Ok, grab the metadata
        this_q = torch.from_numpy(self.questions[index]).long()
        this_answer = torch.LongTensor([self.answers[index]])
        this_filename_cc = self.image_filenames[index]
        this_id = this_filename_cc.split("_")[-2]
        # this_q_family = self.question_family[index]

        # this_template_filename = self.template_filenames[index]
        this_template_filename = "na"

        # A dictionary of keys consisting of camera
        # views.
        scene_from_id = self.id_to_scene[this_id]

        subfolder = scene_from_id["cc"]["subfolder"]

        cam_names = CAM_NAMES
        # If validation set, don't use a canonical
        # pose image (to be more difficult)
        if self.canonical_mode == "train_only":
            # Canonical view is only meant to be
            # in train, so if this is valid, remove
            # 'cc' from the array.
            if self.mode == "val":
                cam_names = [x for x in cam_names if "cc" not in x]
        elif self.canonical_mode == "none":
            # Remove canonical view altogether from both
            # train and valid.
            cam_names = [x for x in cam_names if "cc" not in x]
        elif self.canonical_mode == "b0":
            cam_names = ["cc"]
        else:
            raise Exception("")

        rnd_cam_name = cam_names[np.random.randint(0, len(cam_names))]
        img_filename = this_filename_cc.replace("_cc", "_" + rnd_cam_name).replace(
            ".png", ".jpg"
        )
        this_img_path = "%s/%s/%s/images/%s" % (self.root_images, self.mode_str, subfolder, img_filename)
        img = Image.open(this_img_path).convert("RGB")
        img = self.transform(img)

        ##########
        rnd_cam_name2 = cam_names[np.random.randint(0, len(cam_names))]
        img_filename2 = this_filename_cc.replace("_cc", "_" + rnd_cam_name2).replace(
            ".png", ".jpg"
        )
        this_img_path2 = "%s/%s/%s/images/%s" % (
            self.root_images,
            self.mode_str,
            subfolder,
            img_filename2,
        )
        img2 = Image.open(this_img_path2).convert("RGB")
        img2 = self.transform(img2)

        #########

        this_cam = torch.FloatTensor(scene_from_id[rnd_cam_name]["cam_params"])

        this_cam2 = torch.FloatTensor(scene_from_id[rnd_cam_name2]["cam_params"])

        cc_cam = torch.FloatTensor(scene_from_id["cc"]["cam_params"])

        # Compute interesting attributes about the scene
        colors = [elem["color"] for elem in scene_from_id["cc"]["objects"]]
        shapes = [elem["shape"] for elem in scene_from_id["cc"]["objects"]]
        mats = [elem["material"] for elem in scene_from_id["cc"]["objects"]]
        n_objects = len(scene_from_id["cc"]["objects"])
        meta = {
            "template_filename": this_template_filename,
            "n_color_unique": len(Counter(colors)),
            "n_shape_unique": len(Counter(shapes)),
            "n_mat_unique": len(Counter(mats)),
            "n_objects": n_objects,
        }

        if self.canonical_mode == "b0":
            # Emulating vanilla clevr, so null
            # out the camera.
            this_cam = this_cam * 0.0 + 1.0

        # X2_batch is None because there are no
        # alternate views per scene.
        return img, img2, this_q, this_cam, this_cam2, this_answer, cc_cam, meta

    def __len__(self):
        return len(self.questions)


#####################################
# DATASETS FOR CONTRASTIVE TRAINING #
#####################################


class ClevrKiwiAutoencoderDataset(ClevrKiwiDataset):
    def __init__(self, *args, **kwargs):
        transforms_blank = kwargs.pop("transforms_blank")
        super(ClevrKiwiAutoencoderDataset, self).__init__(*args, **kwargs)
        self.transform_blank = transforms.Compose(transforms_blank)

        # The `id_to_scene` dict is a mapping for ALL class
        # splits, i.e. train+val+test. For this class, we
        # use `image_filenames` to filter out and create a
        # new version of `id_to_scene` which only considers
        # the split we are interested in.
        new_id_to_scene = dict()
        for filename in self.image_filenames:
            this_id = filename.split("_")[-2]
            if this_id not in new_id_to_scene:
                new_id_to_scene[this_id] = self.id_to_scene[this_id]
        self.id_to_scene = new_id_to_scene

        self.scene_keys = list(new_id_to_scene.keys())

    def __len__(self):
        return len(self.scene_keys)

    def __getitem__(self, index):
        # Randomly select a scene, and also get its subfolder
        scene_from_id = self.id_to_scene[self.scene_keys[index]]
        subfolder = scene_from_id["cc"]["subfolder"]

        fname_template = "CLEVR_train-clevr-kiwi-spatial_{scene}_{cam}.jpg"

        # Select a random camera, and create its filename
        rnd_cam_name = CAM_NAMES[np.random.randint(0, len(CAM_NAMES))]
        img_filename = fname_template.format(
            scene=self.scene_keys[index], cam=rnd_cam_name
        )
        this_img_path = "%s/%s/images/%s" % (self.root_images, subfolder, img_filename)
        img_raw = Image.open(this_img_path).convert("RGB")
        img_da = self.transform(img_raw)
        this_cam = torch.FloatTensor(scene_from_id[rnd_cam_name]["cam_params"])

        # Select another random camera, and create its filename
        rnd_cam_name2 = CAM_NAMES[np.random.randint(0, len(CAM_NAMES))]
        img_filename2 = fname_template.format(
            scene=self.scene_keys[index], cam=rnd_cam_name2
        )
        this_img_path2 = "%s/%s/images/%s" % (
            self.root_images,
            subfolder,
            img_filename2,
        )
        img2_raw = Image.open(this_img_path2).convert("RGB")
        img2_da = self.transform(img2_raw)
        this_cam2 = torch.FloatTensor(scene_from_id[rnd_cam_name2]["cam_params"])

        null_ = torch.zeros((1,)).float()
        null_cam = torch.zeros((6,)).float()
        # this_answer = torch.LongTensor(obj_idxs)
        meta = {}

        # img, img2, question, cam1, cam2, answer, cc_cam, meta
        return img_da, img2_da, null_, this_cam, this_cam2, null_, null_cam, meta


class ClevrDataset(Dataset):
    def __init__(
        self, root_images, root_meta, how_many_objs=5, transforms_=None, mode="train"
    ):
        self.transform = transforms.Compose(transforms_)
        self.root_images = root_images
        self.root_meta = root_meta
        self.mode = mode

        if mode == "train":
            f = h5py.File("%s/train_questions.h5" % root_meta, "r")
        else:
            f = h5py.File("%s/val_questions.h5" % root_meta, "r")

        self.vocab = load_vocab(os.environ["DATASET_CLEVR_META"] + "/vocab.json")

        self.dat = {
            "questions": f["questions"][:],
            "image_idxs": f["image_idxs"][:],
            "answers": f["answers"][:],
        }
        f.close()

    def __getitem__(self, index):
        # Ok, grab the metadata
        this_q = torch.from_numpy(self.dat["questions"][index]).long()
        this_idx = self.dat["image_idxs"][index]
        this_answer = torch.LongTensor([self.dat["answers"][index]])
        this_img_path = "%s/%s/CLEVR_%s_%s.png" % (
            self.root_images,
            self.mode,
            self.mode,
            str(this_idx).zfill(6),
        )
        img = Image.open(this_img_path).convert("RGB")
        img = self.transform(img)

        # X2_batch is None because there are no
        # alternate views per scene.
        this_cam = torch.ones((6,)).float()

        return img, img, this_q, this_cam, this_cam, this_answer, this_cam, {}

    def __len__(self):
        return len(self.dat["image_idxs"])


if __name__ == "__main__":
    transforms_ = [transforms.Resize(224), transforms.ToTensor()]
    ds = ClevrDataset(
        root_images="/clevr/CLEVR_v1.0/images/",
        root_meta="/clevr_preprocessed/",
        transforms_=transforms_,
    )

    from torch.utils.data import DataLoader

    loader = DataLoader(ds, num_workers=0, batch_size=16)

    for x_batch, q_batch, y_batch in loader:
        break

    import pdb

    pdb.set_trace()
