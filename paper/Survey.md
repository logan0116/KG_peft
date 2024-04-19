# A Survey of Graph Meets Large Language Model: Progress and Future Directions

(整理来源于[知乎博客](https://zhuanlan.zhihu.com/p/668827243))

本文关注GNN与LLM结合问题，根据LLM在Graph相关任务中所扮演的角色，即enhancer, predictor, alignment component，并将现有的方法分为三类并进行讨论。

GNNs擅长捕获结构信息，但主要依赖于语义约束的嵌入作为节点特征，限制了表达节点语义的能力。而LLMs擅长于文本编码，但往往难以捕获图数据中存在的结构信息。两者相互补充，更具建模能力。


## LLM as Enhancer

该方法目的是通过LLM提高节点嵌入的质量。生成的嵌入应用到图结构上，以供任何GNN使用；或者直接输入到下游分类器中，用于各种任务。

是否使用LLM来生成额外的文本信息，可以更细粒度地划分为两个分支： Explanation-based Enhancement 和 Embedding-based Enhancement。

![2024-04-18_23-19.png](image%2F2024-04-18_23-19.png)

## LLM as Predictor

该方法的核心思想是，利用LLM对各种与图相关的任务进行预测，例如分类和推理。然而，将LLM应用于Graph会遇到挑战，因为图数据通常无法与顺序文本的直接转换。本节将模型依据是否使用GNN来提取LLM的结构特征将模型大致划分为 Flatten-based Prediction 和 GNN-based Prediction 。

![2024-04-18_23-19_1.png](image%2F2024-04-18_23-19_1.png)

### Flatten-based Prediction

Flatten-based Prediction通常包括两个步骤：(1)将图结构转换为节点序列或标记(2)然后应用解析函数从LLM生成的输出中检索预测的标签。

### GNN-based Prediction

与Flatten-based Prediction相比，GNN-based Prediction将固有的结构特征和图数据中的依赖与LLM相结合，允许LLM进行结构感知。

## GNN-LLM Alignment

对GNN和LLM的嵌入空间进行对齐是将图模态与文本模态相结合的有效方法。GNN-LLM Alignment对齐确保保留每个编码器的独特功能，同时在特定阶段协调它们的嵌入空间。本节总结了对齐GNN和LLM的技术，们可以分为对称或不对称，这取决于是否同时强调GNN和LLM，或者一种模式优先于另一种模式。

![2024-04-18_23-20.png](image%2F2024-04-18_23-20.png)

## 总结

目前的三种结构可以基于一个更清晰的框架分析

LLM as Enhancer:
    
    input -> LLM -> GNN -> output

LLM as Predictor:

    input -> GNN -> LLM -> output

GNN-LLM Alignment:

    input -> [GNN & LLM] -> output

这三种方法的结合，可以更好地利用LLM和GNN的优势，提高模型的性能。

我的需求更贴近于GNN-LLM Alignment

在LLM微调的过程中想办法嵌入知识图谱，以提高在QA任务中的表现（相较于一般的微调，相较于一般的RAG）。



