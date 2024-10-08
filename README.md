# LLMisinfo-MDP
Repository created for the Large Language Models for fact-checking online misinformation (LLMisinfo) Multidisciplinary project

## References
### The implementation follows 2 different styles of prompt structure to check the veracity of a claim based on these papers:
- D. Quelle and A. Bovet, “The perils and promises of fact-checking with large language models” Frontiers in Artificial Intelligence, vol. 7, Feb. 2024, doi: 10.3389/frai.2024.1341697. https://arxiv.org/abs/2310.13549
- X. Zhang and W. Gao, “Towards LLM-based Fact Verification on News Claims with a Hierarchical Step-by-Step Prompting Method,” arXiv.org, Sep. 30, 2023. https://arxiv.org/abs/2310.00305

### Initial papers
- D. Quelle and A. Bovet, “The perils and promises of fact-checking with large language models” Frontiers in Artificial Intelligence, vol. 7, Feb. 2024, doi: 10.3389/frai.2024.1341697. https://arxiv.org/abs/2310.13549
- E. C. Choi and E. Ferrara, “FACT-GPT: Fact-Checking Augmentation via Claim Matching with LLMs,” arXiv.org, Feb. 08, 2024. https://arxiv.org/abs/2402.05904
- E. C. Choi and E. Ferrara, “Automated Claim Matching with Large Language Models: Empowering Fact-Checkers in the Fight Against Misinformation,” arXiv.org, Oct. 13, 2023. https://arxiv.org/abs/2310.09223

### Related papers - referenced in “The perils and promises of fact-checking with large language models”
- S. Yao et al., “REACT: Synergizing reasoning and acting in language models,” arXiv.org, Oct. 06, 2022. https://arxiv.org/abs/2210.03629
- Kotonya, N., and Toni, F. (2020). “Explainable automated fact-checking: a survey,” in Proceedings of the 28th International Conference on Computational Linguistics (Barcelona: International Committee on Computational Linguistics), 5430–5443. https://arxiv.org/abs/2011.03870
- Shaar, S., Babulkov, N., Da San Martino, G., and Nakov, P. (2020). “That is a known lie: detecting previously fact-checked claims,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, eds. D. Jurafsky, J. Chai, N. Schluter and J. Tetreault (Association for Computational Linguistics) (Stroudsburg, PA), 3607–3618. https://arxiv.org/abs/2005.06058
  
### Related papers - referenced in FACT-GPT papers
- B. He, M. Ahamad, and S. Kumar, “Reinforcement Learning-based Counter-Misinformation Response Generation: A case study of COVID-19 vaccine misinformation,” arXiv.org, Mar. 11, 2023. https://arxiv.org/abs/2303.06433
- H. Dai et al., “AUGGPT: Leveraging ChatGPT for text Data augmentation,” arXiv.org, Feb. 25, 2023. https://arxiv.org/abs/2302.13007

### Related Papers - foud on Google Scholar
- **(FOLLOW THE PROMPT STRUCTURE)** X. Zhang and W. Gao, “Towards LLM-based Fact Verification on News Claims with a Hierarchical Step-by-Step Prompting Method,” arXiv.org, Sep. 30, 2023. https://arxiv.org/abs/2310.00305
- Z. Li, H. Zhang, and J. Zhang, “A Revisit of Fake News Dataset with Augmented Fact-checking by ChatGPT,” arXiv.org, Dec. 19, 2023. https://arxiv.org/abs/2312.
- N. Lee, Y. Bang, A. Madotto, and P. Fung, “Misinformation Has High Perplexity,” arXiv.org, Jun. 08, 2020. https://arxiv.org/abs/2006.04666

### Papers to Datasets
- RAWFC: Zhiwei Yang, Jing Ma, Hechang Chen, Hongzhan Lin, Ziyang Luo, and Yi Chang. 2022. A coarse-to-fine cascaded evidence-distillation neural network for ex-plainable fake news detection. In COLING, pages 2608–2621. https://aclanthology.org/2022.coling-1.230.pdf
- LIAR: William Yang Wang. 2017. “liar, liar pants on fire”: A new benchmark dataset for fake news detection. In ACL (short), pages 422–426. https://aclanthology.org/P17-2067/

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/alessandrom10/LLMisinfo-MDP
    cd LLMisinfo-MDP
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv venv-name
    source venv-name/bin/activate   # On Windows, use: venv-name\Scripts\activate
    ```

3. **Install required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

**Before running the script, you can configure the prompting method and other variables in the `my_config.yaml` file. An example of such configuration has been provided in `example_my_config.yaml`**

    prompting: "standard" # nosearch - standard - hiss - react

Save the changes before running the script.

## Running the Script - Fact-check a claim

1. **Make sure you are in the main project folder (where the script is located):**

    ```bash
    cd path/to/main/folder
    ```

2. **Run the Python script:**

    ```bash
    python main_factchecker.py
    ```
