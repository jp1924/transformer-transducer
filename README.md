# [Transformer Transducer](https://arxiv.org/abs/2002.02562)
```
Model works but not validated.   
I don't know how to verify it because the paper author,   
"train batch_size to 1024 in an environment where 64 TPU cores were set."

I've been working on it, but it can training.
```

This repo is an implementation of Transformer-Transducer as a Hugging face.    
I referred to the repo below to make this.    
- [oshindow/Transformer-Transducer](https://github.com/oshindow/Transformer-Transducer)
- [zzpDapeng/Transformer-Transducer](https://github.com/zzpDapeng/Transformer-Transducer)
- [huggingface/transformers](https://github.com/huggingface/transformers)  
</b>  
  
## Environment
OS: Ubuntu 18.04.6 LTS     
python: 3.8.13      
pytorch
> pip install torchaudio==0.10.1+cu111 torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html    
</b>

## Abstract
- Transformer-Transducer is a an End2End-based ASR streaming model that converts spoken speech into text in real time.
- Transformer-Transducer is a model that implements RNN-T as Transformer and train using RNN-T loss.
- It consists of Label Encoder in charge of text, Audio Encoder in charge of voice, and Joint Network that combines the calculations of each Encoder   
- And in order not to exceed the max_length of the Transformer, the audio is converted into log-Mel Spectrogram, and then each Mel is stacked to match the voice within the max_length
- The authors of the paper trained the model by 200K steps with a batch-size of 1024 at 8x8 TPU.

## Transformer Transducer
Transformer-Transducer consists of Label Encoder in charge of text, Audio Encoder in charge of voice, and Joint Network that combines the calculations of each Encoder     

In this paper, it is shown as Audio Encoder and Label Encoder, but compared to Seq2Seq, Audio Encoder corresponds to Seq2Seq's Encoder and Label Encoder corresponds to Seq2Seq's Decoder. However, the difference from Seq2Seq is that each encoder performs Self-Attention separately and then combines the calculations in the Joint-Network      

$Joint = Linear(AudioEncoder(x)) + Linear(LabelEncoder(Labels(z_1:(i-1))))$     
$Softmax(Linear(tanh(Joint)))$     

Joint-Network combines the results calculated by Audio Encoder and Label Encoder into one. The process of extracting feature vector for audio and text from each encoder and then linking the extracted values to each other. 

If you look at the formula (or [code]()), you can see that the calculation result of Audio, Label Encoder passes through Lienar and then goes through tanh. I guess the reason why Tanh was added is to filter the silence of the voice.    
    
## Evaluation step
![](/img/rnn_t.png)    

RNN-T loss calculates loss using logits whose shape is 4. 
However, for evaluation to be possible on the huggingface, the shape of the logit from the model must not exceed 3 at most
Therefore, to reduce the shape, text is generated using the generate of the model during evaluation_loop

Unlike CTC, RNN-T should not be predicted using argmax at interference or validation steps.

The rule for RNN-T to generate text is as follows. U + 1 when text comes out, and T + 1 when blank comes out.   
The above method should be applied to the entire process of generating text.
If there are multiple text in the same voice frame,   
U₂ + 1 > U₃ + 1 > U₀ + 1 > U₁ + 1 > ... may proceed until blank appears.
