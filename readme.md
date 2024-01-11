使用13亿中文大模型对罪犯减刑时长进行预测。
赛题地址：[https://www.datafountain.cn/competitions/611/ranking?isRedance=0&sch=2086](https://www.datafountain.cn/competitions/611/ranking?isRedance=0&sch=2086)

代码地址：[https://github.com/ganzhiruyi/ustc_ml2023](https://github.com/ganzhiruyi/ustc_ml2023)
## 分类任务
采用封神榜开源的中文最大Bert类模型[Erlangshen-MegatronBert-1.3B](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B) ，作为中文底座大模型，将时长转化成34分类问题，交叉验证的方法进行模型训练。
### [Erlangshen-MegatronBert-1.3B](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B)模型介绍：
Encoder结构为主的双向语言模型，专注于解决各种自然语言理解任务。 采用了[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)的工作，使用了32张A100，总共耗时14天在悟道语料库（180 GB版本）上训练了十亿级别参数量的BERT。同时，鉴于中文语法和大规模训练的难度，使用四种预训练策略来改进BERT：1) 整词掩码, 2) 知识动态遮掩, 3) 句子顺序预测, 4) 层前归一化.
### 训练流程：

1. **初始化**：开始时，所有分类的被初始化。
2. **数据切分**：1%的数据作为验证集，99%的数据作为训练集
3. **迭代多个Epoch**：
   - 对于每一个step：
      - 学习率采用先线性warmup，然后再cosine decay的schedule
      - 计算总的训练分类loss，即当前模型预测与真实标签之间的差异。
      - 采用adam进行参数更新。
      - 过程中在验证集进行loss, accuracy, precise,recall,f1等多个指标验证。
4. **预测**：所有迭代完成后，进行测试集的预测。
### 实验中的参数设置
调参后主要参数设置如下
```shell
--model_max_length 512 \
--num_train_epochs 4 \
--per_device_train_batch_size 32 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-5 \
--lr_scheduler_type cosine \
--adam_beta1 0.9 \
--adam_beta2 0.98 \
--adam_epsilon 1e-8 \
--max_grad_norm 1.0 \
--weight_decay 1e-1 \
--warmup_ratio 0.01 \
--logging_steps 1 \
--log_level "debug" \
--bf16 True \
--deepspeed $CONFIG_JSON \
--do_train \
--do_eval \
--evaluation_strategy "steps" \
--save_steps 5000 \
--eval_steps 200 \
--run_name $run_name \
--gradient_checkpointing False \

```
### 实验结果
训练集loss
![image.png](https://cdn.nlark.com/yuque/0/2024/png/22769537/1705004127358-934f6d49-57e1-4c10-bddb-5afe028b17ef.png#averageHue=%23fefefe&clientId=u34a6bf12-b02e-4&from=paste&height=307&id=uc8866095&originHeight=614&originWidth=2426&originalType=binary&ratio=2&rotation=0&showTitle=false&size=94890&status=done&style=none&taskId=u79d6d76d-93f1-407c-9302-2dd4f23a45a&title=&width=1213)

验证指标
![image.png](https://cdn.nlark.com/yuque/0/2024/png/22769537/1705004153239-ea679bc9-66fe-4821-9412-adf51a85f319.png#averageHue=%23fdfdfd&clientId=u34a6bf12-b02e-4&from=paste&height=626&id=u9b3a813c&originHeight=1252&originWidth=2430&originalType=binary&ratio=2&rotation=0&showTitle=false&size=171297&status=done&style=none&taskId=ubc14eed8-ad9f-4ddc-a8de-4ef9c7ccfea&title=&width=1215)

### 提交结果
![image.png](https://cdn.nlark.com/yuque/0/2024/png/22769537/1705003797049-a7a177ef-6480-4553-9ad8-83692517a897.png#averageHue=%23fcfcfc&clientId=u9982fbd2-482d-4&from=paste&height=342&id=u5630b4f8&originHeight=684&originWidth=1854&originalType=binary&ratio=2&rotation=0&showTitle=false&size=90132&status=done&style=none&taskId=ud20425b7-e053-424f-93da-ac6030f47cf&title=&width=927)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/22769537/1705003840698-5da33301-4f7e-498c-b8aa-b4b3365cc60e.png#averageHue=%23f5f4f4&clientId=u9982fbd2-482d-4&from=paste&height=809&id=u5aaa9cd5&originHeight=1618&originWidth=2878&originalType=binary&ratio=2&rotation=0&showTitle=false&size=425437&status=done&style=none&taskId=u33844907-451c-479d-9968-3984ac715de&title=&width=1439)
