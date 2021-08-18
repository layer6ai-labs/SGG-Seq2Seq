<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

## ICCV'21 Context-aware Scene Graph Generation with Seq2Seq Transformers

Authors: [Yichao Lu*](https://www.linkedin.com/in/yichaolu/), [Himanshu Rai*](https://www.linkedin.com/in/himanshu-rai-521b1318/), [Cheng Chang*](https://scholar.google.com/citations?user=X7oyRLIAAAAJ&hl=en), Boris Knyazev, [Guangwei Yu](http://www.cs.toronto.edu/~guangweiyu), , Shashank Shekhar, Graham W. Taylor,  [Maksims Volkovs](http://www.cs.toronto.edu/~mvolkovs)  


## Prerequisites and Environment

* pytorch-gpu 1.13.1
* numpy 1.16.0


All experiments were conducted on a 20-core Intel(R) Xeon(R) CPU E5-2630 v4 @2.20GHz and NVIDIA V100 GPU with 32GB GPU memory.

**Note:** INSTRE dataset contains almost 30k images so the training phase requires ~19GB GPU memory. If your GPU doesn't have sufficient memory, please remove the `--gpu-id` argument and the model will train on the CPU. Training on CPU is approximately 10x slower but converges to the same result.

## Dataset

### Visual Genome
Download it [here](https://drive.google.com/open?id=1VDuba95vIPVhg5DiriPtwuVA6mleYGad). Unzip it under the data folder. You should see a `vg` folder unzipped there. It contains .json annotations that suit the dataloader used in this repo.

### Visual Relation Detection

See [Images:VRD](#visual-relation-detection-1)
## Citation

If you find this code useful in your research, please cite the following paper:

  

