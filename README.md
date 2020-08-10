# GAN-BERT Test for SST-2

* modified 
  - bert.py, ganbert.py, data/*, run_experiments.sh

* how to
```
* copy SST-2 train.txt, test.txt to data/
$ cd data
$ python split.py --input_path train.txt --labeled_path labeled.txt --unlabeled_path unlabeled.txt --label_rate=0.02
$ cp labeled.txt data/sst-2
$ cp unabeled.txt data/sst-2
$ cp test.txt data/sst-2
* pretrained bert : cased_L-12_H-768_A-12
$ ./run_experiments.sh 0.02

* 2%
** ganbert.py
I0810 14:15:19.033630 139935412299584 ganbert.py:622] ***** Eval results *****
I0810 14:15:19.033749 139935412299584 ganbert.py:624]   eval_accuracy = 0.82866555
I0810 14:15:19.035303 139935412299584 ganbert.py:624]   eval_f1_macro = 0.5513543
I0810 14:15:19.035398 139935412299584 ganbert.py:624]   eval_f1_micro = 0.8286655
I0810 14:15:19.035490 139935412299584 ganbert.py:624]   eval_loss = 2.251661
I0810 14:15:19.035563 139935412299584 ganbert.py:624]   eval_precision = 0.82866555
I0810 14:15:19.035632 139935412299584 ganbert.py:624]   eval_recall = 0.82866555
I0810 14:15:19.035705 139935412299584 ganbert.py:624]   global_step = 3409
I0810 14:15:19.035777 139935412299584 ganbert.py:624]   loss = 2.5218523
** bert.py
I0810 14:44:08.997009 140240206485312 bert.py:516] ***** Eval results *****
I0810 14:44:08.997140 140240206485312 bert.py:518]   eval_accuracy = 0.86545855
I0810 14:44:08.999604 140240206485312 bert.py:518]   eval_f1_macro = 0.576948
I0810 14:44:08.999707 140240206485312 bert.py:518]   eval_f1_micro = 0.86545855
I0810 14:44:08.999782 140240206485312 bert.py:518]   eval_loss = 0.37505102
I0810 14:44:08.999852 140240206485312 bert.py:518]   eval_precision = 0.86545855
I0810 14:44:08.999922 140240206485312 bert.py:518]   eval_recall = 0.86545855
I0810 14:44:08.999995 140240206485312 bert.py:518]   global_step = 63
I0810 14:44:09.000066 140240206485312 bert.py:518]   loss = 0.37449872

* 5%
** ganbert.py
I0810 15:12:13.429661 140258020972352 ganbert.py:622] ***** Eval results *****
I0810 15:12:13.429783 140258020972352 ganbert.py:624]   eval_accuracy = 0.88138384
I0810 15:12:13.431143 140258020972352 ganbert.py:624]   eval_f1_macro = 0.5875451
I0810 15:12:13.431236 140258020972352 ganbert.py:624]   eval_f1_micro = 0.88138384
I0810 15:12:13.431310 140258020972352 ganbert.py:624]   eval_loss = 1.484564
I0810 15:12:13.431380 140258020972352 ganbert.py:624]   eval_precision = 0.88138384
I0810 15:12:13.431483 140258020972352 ganbert.py:624]   eval_recall = 0.88138384
I0810 15:12:13.431556 140258020972352 ganbert.py:624]   global_step = 3787
I0810 15:12:13.431626 140258020972352 ganbert.py:624]   loss = 1.8002213
** bert.py
I0810 15:19:42.471866 140096035723072 bert.py:516] ***** Eval results *****
I0810 15:19:42.472009 140096035723072 bert.py:518]   eval_accuracy = 0.87644154
I0810 15:19:42.473665 140096035723072 bert.py:518]   eval_f1_macro = 0.58419245
I0810 15:19:42.473797 140096035723072 bert.py:518]   eval_f1_micro = 0.87644154
I0810 15:19:42.473877 140096035723072 bert.py:518]   eval_loss = 0.37862903
I0810 15:19:42.473955 140096035723072 bert.py:518]   eval_precision = 0.87644154
I0810 15:19:42.474031 140096035723072 bert.py:518]   eval_recall = 0.87644154
I0810 15:19:42.474112 140096035723072 bert.py:518]   global_step = 157
I0810 15:19:42.474190 140096035723072 bert.py:518]   loss = 0.37823224

* 10%
** ganbert.py
I0810 15:48:52.336787 140417664419648 ganbert.py:622] ***** Eval results *****
I0810 15:48:52.336913 140417664419648 ganbert.py:624]   eval_accuracy = 0.90170234
I0810 15:48:52.339158 140417664419648 ganbert.py:624]   eval_f1_macro = 0.6011285
I0810 15:48:52.339258 140417664419648 ganbert.py:624]   eval_f1_micro = 0.90170234
I0810 15:48:52.339338 140417664419648 ganbert.py:624]   eval_loss = 1.0568779
I0810 15:48:52.339435 140417664419648 ganbert.py:624]   eval_precision = 0.90170234
I0810 15:48:52.339513 140417664419648 ganbert.py:624]   eval_recall = 0.90170234
I0810 15:48:52.339592 140417664419648 ganbert.py:624]   global_step = 4418
I0810 15:48:52.339669 140417664419648 ganbert.py:624]   loss = 1.3593299
** bert.py
I0810 16:21:16.933243 139837606885184 bert.py:516] ***** Eval results *****
I0810 16:21:16.933361 139837606885184 bert.py:518]   eval_accuracy = 0.88358045
I0810 16:21:16.935383 139837606885184 bert.py:518]   eval_f1_macro = 0.5889775
I0810 16:21:16.935482 139837606885184 bert.py:518]   eval_f1_micro = 0.88358045
I0810 16:21:16.935555 139837606885184 bert.py:518]   eval_loss = 0.38831636
I0810 16:21:16.935626 139837606885184 bert.py:518]   eval_precision = 0.88358045
I0810 16:21:16.935694 139837606885184 bert.py:518]   eval_recall = 0.88358045
I0810 16:21:16.935766 139837606885184 bert.py:518]   global_step = 315
I0810 16:21:16.935837 139837606885184 bert.py:518]   loss = 0.38768956

* 20%


```


-----

# GAN-BERT

Code for the paper **GAN-BERT: Generative Adversarial Learning for Robust Text Classification with a Bunch of Labeled Examples** accepted for publication at **ACL 2020 - short papers** by *Danilo Croce* (Tor Vergata, University of Rome), *Giuseppe Castellucci* (Amazon) and *Roberto Basili* (Tor Vergata, University of Rome). The paper can be found [here](https://www.aclweb.org/anthology/2020.acl-main.191.pdf).

GAN-BERT is an extension of BERT which uses a Generative Adversial setting to implement an effective semi-supervised learning schema. It allows training BERT with datasets composed of a limited amount of labeled examples and larger subsets of unlabeled material. 
GAN-BERT can be used in sequence classification tasks (also involings text pairs). 

This code runs the GAN-BERT experiment over the TREC dataset for the fine-grained Question Classification task. We provide in this package the code as well as the data for running an experiment by using 2% of the labeled material (109 examples) and 5343 unlabeled examples.
The test set is composed of 500 annotated examples.

As a result, BERT trained over 109 examples (in a classification task involving 50 classes) achieves an accuracy of ~13% while GAN-BERT achieves an accuracy of ~42%.

## The GAN-BERT Model

GAN-BERT is an extension of the BERT model within the Generative Adversarial Network (GAN) framework (Goodfellow et al, 2014). In particular, the Semi-Supervised GAN (Salimans et al, 2016) is used to make the BERT fine-tuning robust in such training scenarios where obtaining annotated material is problematic. In fact, when fine-tuned with very few labeled examples the BERT model is not able to provide sufficient performances. With GAN-BERT we extend the fine-tuning stage by introducing a Discriminator-Generator setting, where:

- the Generator G is devoted to produce "fake" vector representations of sentences;
- the Discrimator D is a BERT-based classifier over k+1 categories.

![GAN-BERT model](https://github.com/crux82/ganbert/raw/master/ganbert.jpg)

D has the role of classifying an example with respect to the k categories of the task of interest, and it should recognize the examples that are generated by G (the k+1 category). 
G, instead, must produce representations as much similar as possible to the ones produced by the model for the "real" examples. G is penalized when D correctly classify an example as fake.

In this context, the model is trained on both labeled and unlabeled examples. The labeled examples contributes in the computation of the loss function with respect to the task k categories. The unlabeled examples contributes in the computation of the loss functions as they should not be incorrectly classified as beloning to k+1 category (i.e., the fake category).

The resulting model is demonstrated to learn text classification tasks starting from very few labeled examples (50-60 examples) and to outperform the classifcal BERT fine-tuned models by large margin in this setting.

In the following plots, the performances of GAN-BERT are reported for different tasks at different percentage of labeled examples. We measured the accuracy (or F1) of the model for the following tasks: Topic Classification on the 20News (20N) dataset; Question Classification (QC) on the TREC dataset; Sentiment Analysis on the SST dataset (SST-5); Natural Language Inference over the MNLI dataset (MNLI).

![Performances](https://github.com/crux82/ganbert/raw/master/ganbert_performances.png)

## Requirements

The code is a modification of the original Tensorflow code for BERT (https://github.com/google-research/bert).
It has been tested with Tensorflow 1.14 over a single Nvidia V100 GPU. The code should be compatible with TPUs, but it has not been tested on such architecture or on multiple GPUs.
Moreover, it uses tf_metrics (https://github.com/guillaumegenthial/tf_metrics) to compute some performance measure.

## Installation Instructions
It is suggested to use a python 3.6 environment to run the experiment.
If you're using conda, create a new environment with:

```
conda create --name ganbert python=3.6
```

Activate the newly create environment with:

```
conda activate ganbert
```

And install the required packages by:

```
pip install -r requirements.txt
```

This should install both Tensorflow and tf_metrics.

## How to run an experiment

The run_experiment.sh script contains the necessary steps to run an experiment with both BERT and GANBERT.

The script can be launched with:

```
sh run_experiment.sh
```

The script will first download the BERT-base model, and then it will run the experiments both with GANBERT and with BERT.

After some time (on a Nvidia Tesla V100 it takes about 5 minutes) there will be two files in output: *qc-fine_statistics_BERT0.02.txt* and *qc-fine_statistics_GANBERT0.02.txt*. These two contain the performance measures of BERT and GANBERT, respectively. 

After training a traditional BERT and GAN-BERT on only 109 labeled examples in a classification task involving 50 classes, the following results are obtained:

**BERT**
<pre>
eval_accuracy = 0.136 
eval_f1_macro = 0.010410878 
eval_f1_micro = 0.136 
eval_loss = 3.7638452 
eval_precision = 0.136 
eval_recall = 0.136 
</pre>

**GAN-BERT**
<pre>
eval_accuracy = 0.418 
eval_f1_macro = 0.056867357 
eval_f1_micro = 0.418
eval_loss = 2.744603 
eval_precision = 0.418
eval_recall = 0.418
</pre>


## Out-of-memory issues

As the code is based on the original BERT Tensorflow code and that it starts from the BERT-base model, the same batch size and sequence length restrictions apply here based on the GPU that is used to run an experiment.

Please, refer to the BERT github page (https://github.com/google-research/bert#out-of-memory-issues) to find the suggested batch size and sequence length given the amount of GPU memory available.

## Citation

To cite the paper, please use the following:

<pre>
@inproceedings{croce-etal-2020-gan,
    title = "{GAN}-{BERT}: Generative Adversarial Learning for Robust Text Classification with a Bunch of Labeled Examples",
    author = "Croce, Danilo  and
      Castellucci, Giuseppe  and
      Basili, Roberto",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.191",
    pages = "2114--2119"
}
</pre>


## References
- Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza,Bing Xu, David Warde-Farley, Sherjil Ozair, AaronCourville and Yoshua Bengio. 2014. Generative Adversarial Nets. In Z. Ghahramani, M.  Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems 27, pages 2672–2680. Curran Associates, Inc.
- Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen, and Xi Chen. 2016. Improved techniques for training gans. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems 29, pages 2234–2242. Curran Associates, Inc.
