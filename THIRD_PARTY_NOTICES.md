# Third-party materials and attribution

## Oxford-IIIT Pet images and annotations

Source: [The Oxford-IIIT Pet Dataset, Visual Geometry Group, University of Oxford](https://www.robots.ox.ac.uk/~vgg/data/pets/).
Dataset creators: Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and
C. V. Jawahar. Citation: *Cats and Dogs*, IEEE Conference on Computer Vision
and Pattern Recognition, 2012.

The dataset page states that the data is distributed under
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/),
with copyright retained by the original image owners. These terms apply to
dataset material, independently of any license for this repository's code.
Original photographer identities are not provided by the project manifests;
no ownership of their photographs is claimed here.

- `reports/errors/**/*.jpg` and the three top-level case JPEGs are copies of
  source photographs without re-encoding. Original image IDs are preserved;
  `reports/errors/index.csv` and `cases.csv` record their source paths and hashes.
- Grad-CAM panels in `reports/errors/` and `reports/figures/`, image examples in
  the EDA notebook, and rendered confusion-review pages incorporate dataset
  imagery. Their image content retains the dataset attribution and CC BY-SA
  4.0 terms. Modifications include resizing/cropping, visual overlays, labels,
  and juxtaposition; these do not imply endorsement by the dataset creators.
- Raw archives and extracted data remain local, ignored files in `data/`.
  Downloading them does not grant rights beyond the source's terms. Retain
  this notice and the source/license links when redistributing image material.

## Runtime dependencies and pretrained weights

Dependencies are installed separately, with their own licenses and notices;
this repository does not replace those terms. The pinned environment is in
`requirements.txt`. Principal direct dependency licenses are:

| Dependency | Upstream license information |
| --- | --- |
| PyTorch | [BSD-style license](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| torchvision | [BSD 3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) |
| NumPy | [BSD 3-Clause and bundled notices](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| pandas | [BSD 3-Clause](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| scikit-learn | [BSD 3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) |
| Pillow | [MIT-CMU and component notices](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| PyYAML | [MIT](https://github.com/yaml/pyyaml/blob/main/LICENSE) |
| Matplotlib | [PSF-based license and bundled notices](https://matplotlib.org/stable/project/license.html) |
| seaborn | [BSD 3-Clause](https://github.com/mwaskom/seaborn/blob/master/LICENSE.md) |
| Jupyter | [BSD 3-Clause](https://github.com/jupyter/jupyter/blob/master/LICENSE) |

The pretrained ResNet50 initialization is
`ResNet50_Weights.IMAGENET1K_V2`, downloaded by torchvision from the official
PyTorch model host. The torchvision software license is not a blanket license
for ImageNet source images or all uses of derived weights. See the
[torchvision pretrained-model terms](https://docs.pytorch.org/vision/stable/models.html)
and the applicable dataset terms. Fine-tuned checkpoint hashes identify
artifacts; they do not grant additional rights in third-party materials.

## Project code

The authors' requested policy for this cleanup is **all rights reserved**;
no open-source license is granted for original source code. See [LICENSE](LICENSE).
Third-party image/dependency terms above remain applicable and are not
overridden by the code notice.
