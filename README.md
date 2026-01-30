# Empirical Evaluation of Robustness for Deep Learning-Based Software Defect Prediction Models

This repository contains the source code and datasets for the paper **"Empirical Evaluation of Robustness for Deep Learning-Based Software Defect Prediction Models"**.

We provide all scripts and processed datasets required to reproduce the experiments involving CodeBERT, GraphCodeBERT, UnixCoder, and VulBERTa.

##  Getting Started

### 1. Environment Setup

Clone the repository and set up the Python environment using Conda:

```bash
# Clone the repository
git clone https://github.com/cszaiti/SDP.git
cd SDP

# Create and activate conda environment
conda create -n sdp_env python=3.9
conda activate sdp_env

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pre-trained Models

You need to download the pre-trained model files (`pytorch_model.bin`) from Hugging Face and place them in the corresponding configuration directories.

**Download Links:**
*   **CodeBERT:** [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)
*   **GraphCodeBERT:** [microsoft/graphcodebert-base](https://huggingface.co/microsoft/graphcodebert-base)
*   **UnixCoder:** [microsoft/unixcoder-base-nine](https://huggingface.co/microsoft/unixcoder-base-nine)
*   **VulBERTa:** [yakov2210/vulberta_devign](https://huggingface.co/yakov2210/vulberta_devign)

**Placement Instructions:**
After downloading the `pytorch_model.bin` for each model, move it to the folder containing its `config.json`. For example:

*   **CodeBERT:** `codebert_graphcodebert_unixcoder/code/codebert/pytorch_model.bin`
*   **GraphCodeBERT:** `codebert_graphcodebert_unixcoder/code/graphcodebert/pytorch_model.bin`
*   **UnixCoder:** `codebert_graphcodebert_unixcoder/code/unixcoder-nine/pytorch_model.bin`
*   **vulberta:** `vulberta/models/VulBERTa/pytorch_model.bin`

### 3. Datasets

We use the Defect Detection dataset from [CodeXGLUE](https://github.com/LovelyBuggies/Defect-Detection). The processed datasets are already included in the `dataset` folders within this repository.

---

##  Experiment Reproduction

### A. CodeBERT, GraphCodeBERT, and UnixCoder

Below are the commands to reproduce the experiments. We use **GraphCodeBERT** as an example.

> **Note:** For UnixCoder, please change `--block_size` to `1000` and `--train_batch_size` to `10` (adjust batch size based on your GPU memory).

First, navigate to the source code directory:

```bash
cd codebert_graphcodebert_unixcoder/code
```


#### 1. Baseline Performance Evaluation
Train and evaluate the model on the original dataset.

```bash
python run.py \
    --output_dir=../saved_graphcodebert_models/ \
    --tokenizer_name=./graphcodebert/ \
    --model_name_or_path=./graphcodebert/pytorch_model.bin \
    --config_name=./graphcodebert/config.json \
    --model_type=roberta \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 8 \
    --block_size 500 \
    --train_batch_size 36 \
    --eval_batch_size 36 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
```

#### 2. Intrinsic Robustness Metric
Perform attacks on the trained model.

> **Important:** Ensure the best model checkpoint is placed in `../saved_graphcodebert_models/checkpoint-best-acc/` before running this step. Otherwise, the script will use the last saved checkpoint.

```bash
python run.py \
    --output_dir=../saved_graphcodebert_models/ \
    --tokenizer_name=./graphcodebert/ \
    --model_name_or_path=./graphcodebert/pytorch_model.bin \
    --config_name=./graphcodebert/config.json \
    --model_type=roberta \
    --do_attack \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 8 \
    --block_size 500 \
    --train_batch_size 36 \
    --eval_batch_size 36 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
```

#### 3. Representation Learning Impact Assessment

```bash
python run.py \
    --output_dir=../saved_graphcodebert_models/ \
    --tokenizer_name=./graphcodebert/ \
    --model_name_or_path=./graphcodebert/pytorch_model.bin \
    --config_name=./graphcodebert/config.json \
    --model_type=roberta \
    --do_train \
    --do_eval \
    --do_test \
    --do_attacktrain \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 8 \
    --block_size 500 \
    --train_batch_size 36 \
    --eval_batch_size 36 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
```

#### 4. Adversarial Training Effectiveness Verification

```bash
python run.py \
    --output_dir=../saved_graphcodebert_models/ \
    --tokenizer_name=./graphcodebert/ \
    --model_name_or_path=./graphcodebert/pytorch_model.bin \
    --config_name=./graphcodebert/config.json \
    --model_type=roberta \
    --do_attack \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 8 \
    --block_size 500 \
    --train_batch_size 36 \
    --eval_batch_size 36 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
```

### B. VulBERTa

For VulBERTa, navigate to the `vulberta` directory and follow similar steps, but ensure you use the appropriate batch sizes.

> **Note:** For VulBERTa, we recommend `--train_batch_size 128` (depending on VRAM) and `--block_size 1000`.

```bash
cd vulberta
# Run commands similar to the above, ensuring paths to config/models are correct relative to the 'vulberta' directory.
```

##  Project Structure

Below is an overview of the file structure to help you understand the organization of the repository:

```text
.
├─ codebert_graphcodebert_unixcoder  
│  ├─ bast_model                     
│  ├─ code                           
│  │  ├─ bast_model                  
│  │  ├─ codebert                    
│  │  │  ├─ config.json              
│  │  │  ├─ merges.txt               
│  │  │  ├─ tokenizer_config.json    
│  │  │  └─ vocab.json               
│  │  ├─ evaluator                   
│  │  │  ├─ evaluator.py             
│  │  │  └─ test.jsonl               
│  │  ├─ graphcodebert               
│  │  │  ├─ config.json              
│  │  │  ├─ merges.txt               
│  │  │  ├─ tokenizer_config.json    
│  │  │  └─ vocab.json               
│  │  ├─ libs                        
│  │  │  └─ c.so                     
│  │  ├─ src                         
│  │  │  ├─ models                   
│  │  │  │  └─ base_model.py         
│  │  │  └─ utils                    
│  │  │     ├─ config.py             
│  │  │     ├─ helper.py             
│  │  │     └─ ideas.txt             
│  │  ├─ unixcoder-nine              
│  │  │  ├─ config.json              
│  │  │  ├─ merges.txt               
│  │  │  ├─ README.md                
│  │  │  ├─ special_tokens_map.json  
│  │  │  ├─ tokenizer_config.json    
│  │  │  ├─ vocab.json               
│  │  │  └─ vocab.txt                
│  │  ├─ attacker.py                 
│  │  ├─ build_c_ku.py               
│  │  ├─ create_uid.py               
│  │  ├─ log_acc.txt                 
│  │  ├─ model.py                    
│  │  ├─ modifier.py                 
│  │  ├─ pattern.py                  
│  │  ├─ qz3.py                      
│  │  ├─ run.py                      
│  │  ├─ run.sh                      
│  │  ├─ trie.pkl                    
│  │  └─ trie.py                     
│  ├─ dataset                        
│  │  ├─ attack.jsonl                
│  │  ├─ forbidden_uid.jsonl         
│  │  ├─ preprocess.py               
│  │  ├─ test.jsonl                  
│  │  ├─ test.txt                    
│  │  ├─ train.jsonl                 
│  │  ├─ train.txt                   
│  │  ├─ tt.jsonl                    
│  │  ├─ uid_all.jsonl               
│  │  ├─ uid_test.jsonl              
│  │  ├─ uid_train.jsonl             
│  │  ├─ valid.jsonl                 
│  │  └─ valid.txt                   
│  ├─ evaluator                      
│  │  ├─ evaluator.py                
│  │  └─ test.jsonl                  
│  ├─ libs                           
│  │  └─ c.so                        
│  ├─ saved_models                   
│  ├─ src                            
│  │  ├─ models                      
│  │  │  └─ base_model.py            
│  │  └─ utils                       
│  │     ├─ config.py                
│  │     ├─ helper.py                
│  │     └─ ideas.txt                
│  ├─ attacker.py                    
│  ├─ build_c_ku.py                  
│  ├─ create_uid.py                  
│  ├─ log_acc.txt                    
│  ├─ model.py                       
│  ├─ modifier.py                    
│  ├─ pattern.py                     
│  ├─ qz3.py                         
│  ├─ run.py                         
│  ├─ run.sh                         
│  ├─ trie.pkl                       
│  └─ trie.py                        
├─ vulberta                          
│  ├─ dataset                        
│  │  ├─ attack.jsonl                
│  │  ├─ forbidden_uid.jsonl         
│  │  ├─ preprocess.py               
│  │  ├─ test.jsonl                  
│  │  ├─ test.txt                    
│  │  ├─ train.jsonl                 
│  │  ├─ train.txt                   
│  │  ├─ tt.jsonl                    
│  │  ├─ uid_all.jsonl               
│  │  ├─ uid_test.jsonl              
│  │  ├─ uid_train.jsonl             
│  │  ├─ valid.jsonl                 
│  │  └─ valid.txt                   
│  ├─ evaluator                      
│  │  ├─ evaluator.py                
│  │  └─ test.jsonl                  
│  ├─ libs                           
│  │  └─ c.so                        
│  ├─ models                         
│  │  ├─ bast_model                  
│  │  └─ VulBERTa                    
│  │     ├─ checkpoint-best-acc      
│  │     ├─ config.json              
│  │     ├─ predictions.txt          
│  │     ├─ rng_state.pth            
│  │     ├─ scheduler.pt             
│  │     ├─ trainer_state.json       
│  │     └─ training_args.bin        
│  ├─ tokenizer                      
│  │  ├─ drapgh-merges.txt           
│  │  └─ drapgh-vocab.json           
│  ├─ attacker.py                    
│  ├─ custom.py                      
│  ├─ model.py                       
│  ├─ models.py                      
│  ├─ modifier.py                    
│  ├─ pattern.py                     
│  ├─ qz3.py                         
│  └─ run.py                         
└─ README.md 
```
