# CVPR2021-CEFNet
Encoder Fusion Network with Co-Attention Embedding for Referring Image Segmentation, CVPR2021


If you find this work useful in your research, please consider citing:
```
@inproceedings{feng2021encoder,
  title={Encoder Fusion Network with Co-Attention Embedding for Referring Image Segmentation},
  author={Feng, Guang and Hu, Zhiwei and Zhang, Lihe and Lu, Huchuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15506--15515},
  year={2021}
}
```

## Generate data
Code for data preparation：https://pan.baidu.com/s/1uWTBoVSeWRSgYMjJnnEvsQ (fetch code：data)

Partial coda are borrowed from [TF-phrasecut-public](https://github.com/chenxi116/TF-phrasecut-public). Please follow their instructions to make your setup ready.
···
python build_batches.py -d Gref -t train
python build_batches.py -d Gref -t val
python build_batches.py -d unc -t train
python build_batches.py -d unc -t val
python build_batches.py -d unc -t testA
python build_batches.py -d unc -t testB
python build_batches.py -d unc+ -t train
python build_batches.py -d unc+ -t val
python build_batches.py -d unc+ -t testA
python build_batches.py -d unc+ -t testB
python build_batches.py -d referit -t trainval
python build_batches.py -d referit -t test
···
