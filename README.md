# 2D or not 2D ?
This is the official repository for our paper: _2D or not 2D: How does the dimensionality of gestures representation affect 3D co-speech gesture generation?_ Téo Guichoux, Laure Soulier, Nicolas Obin, Catherine Pelachaud.

[Arxiv](https://www.arxiv.org/abs/2409.10357)

# Access data
To access the database, please follow the instructions at [Gesture-Generation-from-Trimodal-Context](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context/tree/master).
Change the configuration files according to your data directory.

# Getting started
Clone the current repository.
Once you have modified the data paths in the configuration files, to train VideoPose3D simply run
```
python VP3D_train_TED.py
```

To train Trimodal 3D run:
```
python trimodal_train_TED.py --config=config/trimodal_3D.yml --input_context=both --name=trimodal_3D
```
For the 2D version run:
```
python trimodal_train_TED.py --config=config/trimodal_2D.yml --input_context=both --name=trimodal_2D
```

Similarly, for DiffGesture run:
```
python DiffGesture_train_TED.py --config=config/diffgesture_3D.yml --name=diffgesture_3D
```
or
```
python DiffGesture_train_TED.py --config=config/diffgesture_2D.yml --name=diffgesture_2D
```

The logs and checkpoints will be saved in the ```output ``` directory.
# Aknowledgement
This code is based on the code of [Gesture-Generation-from-Trimodal-Context](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context/tree/master), [DiffGesture](https://github.com/Advocate99/DiffGesture/tree/main) and [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).

If you want to use this code, please cite their work:

```
@article{Yoon2020Speech,
  title={Speech Gesture Generation from the Trimodal Context of Text, Audio, and Speaker Identity},
  author={Youngwoo Yoon and Bok Cha and Joo-Haeng Lee and Minsu Jang and Jaeyeon Lee and Jaehong Kim and Geehyuk Lee},
  journal={ACM Transactions on Graphics},
  year={2020},
  volume={39},
  number={6},
}

@inproceedings{zhu2023taming,
  title={Taming Diffusion Models for Audio-Driven Co-Speech Gesture Generation},
  author={Zhu, Lingting and Liu, Xian and Liu, Xuanyu and Qian, Rui and Liu, Ziwei and Yu, Lequan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10544--10553},
  year={2023}
}

@inproceedings{pavllo:videopose3d:2019,
  title={3D human pose estimation in video with temporal convolutions and semi-supervised training},
  author={Pavllo, Dario and Feichtenhofer, Christoph and Grangier, David and Auli, Michael},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}

```

