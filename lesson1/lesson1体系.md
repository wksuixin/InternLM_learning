# 书生·浦语大模型全链路开源体系

## 1. InternLM2体系
**有7B和20B模型:**
InternLM2-Base: 模型基座
InternLM2: 在base的基础上，多个方向进行能力强化，是推荐的大部分应用中考虑的基座。
InternLM2-Chat：在base基础上，经过SFT和RLHF，面向对话交互进行了优化，有很好的指令遵循和共情聊天等能力。 

## 2. 介绍
### 2.1 新一代的数据清洗过滤技术。
* 多维度数据价值评估
* 高质量预料驱动的数据富集
* 有针对性的数据补齐

### 2.2 亮点
* 超长上下文：20万token实现大海捞针
* 综合性能全面提升： 推理 数学 代码提升显著
* 优秀的对话和创作体验：精准指令跟随，丰富的结构化创作。AlpacaEval2.
* 工具调用能力升级： 
* 突出的数理能力和实用的数据分析功能： GSM8K和MATH上与GPT-4相仿。
 
 ### 2.3 体系
 * 数据：2TB数据，涵盖多种模态与任务(书生万卷1.0，书生万卷CC)。
 * 预训练：InternLM-Train
 * 微调：XTuner（增量续训，有监督微调）
 * 部署：LMDeploy
 * 评测： OpenCompass(100套评测集 50万道题目)；（CompassRank, CompassKit, CompassHub）
 * 应用：Lagent AgentLego