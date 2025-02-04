# RADAI

## Using Class Activation Maps (CAM) to localize Liver in CT scans

As given in this [paper](https://arxiv.org/pdf/1512.04150.pdf) I use class activation maps extracted from densetnet's last layer.
I downloaded a pre trained (on CIFAR dataset for instance segmentation) densenet froze the entire network except the last and trained to classify liver and no liver images.
The features in the last layer implicitly give us the location of the Liver organ. However, the model I trained requires further hyper parameter tuning and thresholding to get a much defined masks. Further use of CRF (conditional random fields) also would help. Similar idea to grow from seed mask (initial mask) is implemented in this [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Weakly-Supervised_Semantic_Segmentation_CVPR_2018_paper.pdf). Another interesting work in the similar area was explained in this [paper](https://arxiv.org/pdf/1803.10464.pdf). However, these ideas require powerful GPUs and a lot more time to implement.

To train the model run
```
python train.py data generatecsv
```
data - dataset path 
geenratecsv - it is used to generate and structure dataset files. Once the csv files are created this argument is not required. 

*.pth file is generated after the training is complete (it does not track the best performing model, however train.py can be modified to do so)

To generate masks run
```
python generatemasks.py data generatecsv
```
The file generated are stored in test256/out (both mask and image merged) and test256/outmasks (just the masks). 
