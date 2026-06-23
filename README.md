<div align="center">
<h2> ReCal3R: Reliability-Calibrated Learning Rates for Streaming 3D Reconstruction </h2>
<p align="center">
  <a href="https://arxiv.org/abs/2603.17625"><img src="https://img.shields.io/badge/arXiv-S_VGGT-red?logo=arxiv" alt="Paper PDF (Coming Soon)"></a>
  <a href="https://github.com/Powertony102/S-VGGT"><img src="https://img.shields.io/badge/Project_Page-SVGGT-yellow" alt="Project Page"></a>
</p>
<p align="center">
  <a href="https://xinzelicv.github.io/">Xinze Li</a><sup>1</sup>,
  Yiyuwan Wang<sup>2,1</sup>,
  Pengxu Chen<sup>3</sup>,
  <a href="https://wtchengcv.github.io/">Wentao Cheng</a><sup>1,*</sup>,
  Weifeng Su<sup>1,4</sup>,
  Wentao Fan<sup>1,4</sup>,
  Weisi Lin<sup>5</sup>
</p>
<p align="center">
  <sup>1</sup>Beijing Normal-Hong Kong Baptist University &nbsp;&nbsp;
  <sup>2</sup>Hong Kong Baptist University &nbsp;&nbsp;
  <sup>3</sup>Jilin University &nbsp;&nbsp;
  <sup>4</sup>Guangdong Provincial Key Laboratory of Interdisciplinary Research and Application for Data Science &nbsp;&nbsp;
  <sup>5</sup>Nanyang Technological University
</p>
<p align="center">
  <sup>*</sup>Corresponding Author
</p>

<p align="center">
  Contact: t330026083@mail.bnbu.edu.cn
</p>
<p align="center">
  <a href="https://bnbu.edu.cn/" style="margin: 0 25px;"><img height="50" src="assets/logo_bnbu.svg"></a>
  <a href="https://www.hkbu.edu.hk/en.html" style="margin: 0 25px;"><img height="50" src="assets/logo_hkbu.svg"></a>
  <a href="https://www.jlu.edu.cn/" style="margin: 0 25px;"><img height="50" src="assets/logo_jlu.webp"></a>
  <a href="https://www.ntu.edu.sg/" style="margin: 0 25px;"><img height="50" src="assets/ntu_logo.webp"></a>
</p>
</div>

## Getting Started

### Installation

1. Clone ReCal3R.
```bash
git clone https://github.com/Powertony102/ReCal3R.git
cd ReCal3R
```

2. Create the environment.
```bash
conda create -n recal3r python=3.11 cmake=3.14.0
conda activate recal3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
# issues with pytorch dataloader, see https://github.com/pytorch/pytorch/issues/99625
conda install 'llvm-openmp<16'
# for evaluation
pip install evo
pip install open3d
```

3. Compile the cuda kernels for RoPE (as in CroCo v2).
```bash
cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```

### Download Checkpoints

CUT3R provide checkpoints trained on 4-64 views: [`cut3r_512_dpt_4_64.pth`](https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link).

To download the weights, run the following commands:
```bash
cd src
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
cd ..
```



## 🍺 Acknowledgements

- Thanks to these great repositories:[CUT3R](https://github.com/CUT3R/CUT3R), [TTT3R](https://github.com/Inception3D/TTT3R), [TTSA3R](https://github.com/anonus2357/ttsa3r), [MeMix](https://github.com/dongjiacheng06/MeMix), [Easi3R](https://github.com/Inception3D/Easi3R), [DUSt3R](https://github.com/naver/dust3r), [MonST3R](https://github.com/Junyi42/monst3r), [Spann3R](https://github.com/HengyiWang/spann3r), [Viser](https://github.com/nerfstudio-project/viser) and many other inspiring works in the community.

- Special thanks to our supervisor [Dr. Wentao Cheng](https://wtchengcv.github.io/) for consistent suggestions and efforts to this work.
