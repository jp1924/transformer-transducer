# torch_BERT
pytorch.ver BERT     
    
## TODO
처음부터 모든걸 만들기 힘들기 때문에 일부 기능을 Transformers로 부터 빌린 뒤 하나하나 대체해 나가는 방식으로 제작합니다.    
이 레포의 목적은 huggingface transformers를 최대한 사용하지 않고 순수 pytorch만으로    
MachineLearning에 있는 요소들을 스스로 구현할 수 있는지를 보는 repo입니다.   
    
각 체크박스에 있는 요소를 구현할 때는 별도의 brach를 파서 구현하도록 합니다.

- [ ] Tokenizer    
    - [ ] BPE    

- [ ] Model    
    - [ ] Self-Attention    
    - [ ] Feed Forward Network    
    - [ ] Classification Header    
    
- [ ] Trainer    
    - [ ] train_step    
        - [ ] DDP or DeepSpeed    
        - [ ] Gradient Accumulation    
        - [ ] Hyperparameter Search    
        - [ ] DataLoader    
            - [ ] Collator    
            - [ ] Sampler    
        - [ ] Scheduler    
            - [ ] warm_up step    
        - [ ] Logger(like wandb)    
    - [ ] valid_step    
        - [ ] Logger    
        - [ ] DataLoader    
        - [ ] metrics    


## pytorch
> pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html    
