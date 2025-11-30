# SharedStack
This repository includes an implementation of SharedStack, a training acceleration method.
Note that the implementation is based on NVIDIA Nemo toolkit and its variation.

SharedStack accelerates training based on parameter sharing, and detailed training algorithm will be updated on here.
We tested our method on a Conformer-CTC model, and the 20-layer 512-dim model with SharedStack achieves 2.9% WER on the LibriSpeech test-clean dataset and 6.9% WER on the LibriSpeech test-other dataset with x1.21 speedup.
With x1.30 speedup using SharedStack, the model achieves 3.0% and 7.0% WER on the same datasets.
Without the proposed method, the original model achieves 2.9% WER and 6.8% WER on the same datasets, respectively.

Also, the 24-layer 1024-dim model with SharedStack achieves 2.7% WER and 6.3% WER on the same datasets with x1.21 speedup.

## Training details
The models are trained with the 128 unigram tokenizer.

## Models
You can download the trained models in the following links.

- [20-layer Conven](https://drive.google.com/file/d/11qfKFH-WlZQDfNBJOmePXzFx4ZEn-lBh/view?usp=sharing)
- [20-layer SharedStack](https://drive.google.com/file/d/1hqeAdq_vMxQf3mDXF3OMp6SdjoMOjgOH/view?usp=sharing)
- [24-layer SharedStack]()
