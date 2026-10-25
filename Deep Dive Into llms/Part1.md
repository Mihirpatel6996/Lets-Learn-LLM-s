# Deep Dive into LLMs (like ChatGPT): Pre-training, Tokenization, Neural Networks, Inference and Base Models

## Overview

This video is a full, general-audience introduction to large language models (LLMs) like ChatGPT. The goal is to give you **mental models** for what this tool is, where it is strong, and where it has "sharp edges". The speaker walks through the first major stage of building an LLM, called **pre-training**: collecting internet text, converting it to tokens, training a neural network to predict the next token, and then generating text from the trained model (inference). He shows real examples with GPT-2 and Llama 3.1, and shows that a **base model** is only an "internet text simulator", not yet an assistant. The transcript provided ends just as the video begins to move from pre-training to post-training.

---

## 1. Introduction: What Are We Talking To?

The speaker explains why he made the video and what he wants viewers to learn.

- The video is a **comprehensive but general-audience** introduction to LLMs like ChatGPT.
- The main aim is to give you **mental models** for thinking about what this tool is.
- The tool is described as:
  - "Magical and amazing" in some respects.
  - Really good at some things and not very good at other things.
  - Full of **sharp edges** that users should be aware of.
- Key questions the video wants to answer:
  - What is behind the text box where you type and press Enter?
  - What should we put into that box?
  - What are the words that come back, and how are they generated?
  - What exactly are you talking to?
- Plan of the video:
  - Go through the **entire pipeline** of how these systems are built.
  - Keep everything accessible to a general audience.
  - Along the way, discuss some **cognitive and psychological implications** of the tools.
- The building process happens in **multiple stages arranged one after another**. The first stage is **pre-training**.

> **Keyword: LLM (Large Language Model)**
> **What it is:** A very large neural network that is trained on huge amounts of text so it can predict and generate text. ChatGPT is built on one.
> **Why it was said in this video:** The whole video is about explaining what an LLM is and how one is built.
> **Why it is important:** LLMs are now used by many people every day. Knowing how they work helps you trust them the right amount and avoid their weak points.

---

## 2. Pre-training Stage, Step 1: Downloading and Processing the Internet

The first step of pre-training is to collect a very large amount of text from the internet and clean it.

### 2.1 What are we trying to achieve?

- We want a **huge quantity** of text from publicly available sources.
- The documents should be **high quality**.
- We also want **large diversity**, because we want the model to hold a lot of knowledge.
- So the target is: **many, many documents that are high quality and very diverse**.
- Doing this well is quite complicated and takes multiple stages.
- All major LLM providers (OpenAI, Anthropic, Google and others) have some **internal equivalent** of a dataset like this.

> **Keyword: Pre-training**
> **What it is:** The first big stage of building an LLM, where the model learns from a very large amount of internet text by predicting the next token.
> **Why it was said in this video:** The speaker says this is the first stage of building ChatGPT, and the first step in it is to download and process the internet.
> **Why it is important:** Almost all of the model's knowledge about the world comes from this stage.

> **Keyword: FineWeb**
> **What it is:** A dataset of internet text collected, created and curated by the company Hugging Face. It has a blog post that explains in detail how it was built.
> **Why it was said in this video:** It is used as a **representative example** of what a production-grade pre-training dataset looks like.
> **Why it is important:** It shows that the data behind LLMs is carefully built and filtered, not just a random copy of the web.

### 2.2 How big is the data?

- FineWeb ends up being only about **44 terabytes** of disk space.
- You can buy a 1 terabyte USB stick easily, and this could almost fit on a **single hard drive** today.
- So it is **not a huge amount of data** at the end of the day, because:
  - The internet is very large, but we only work with **text**.
  - We **filter aggressively**.
- Later he says this equals about **15 trillion tokens**.

### 2.3 Starting point: Common Crawl

- Most of the data comes from **Common Crawl**.
- Common Crawl is an organization that has been scouring the internet since **2007**.
- As of 2024, it had indexed about **2.7 billion web pages**.
- How crawling works:
  - Start with a few **seed web pages**.
  - Follow all the links on them, and keep following links.
  - Keep indexing all the information you find.
  - Over time you get a huge amount of internet data.
- This raw data is **quite raw** and is filtered in many different ways.

> **Keyword: Common Crawl**
> **What it is:** A non-profit style crawler project that has been copying web pages from the internet since 2007 and makes the data available.
> **Why it was said in this video:** It is the starting point and the source of most of the data in datasets like FineWeb.
> **Why it is important:** It shows where LLM training text really comes from: publicly available web pages collected at scale.

### 2.4 Filtering stages (as described in the FineWeb diagram)

**a) URL filtering**
- There are **block lists** of URLs or domains that you do not want to take data from.
- Usually these include:
  - Malware websites
  - Spam websites
  - Marketing websites
  - Racist websites
  - Adult sites
- Many types of sites are removed here because we do not want them in the dataset.

> **Keyword: URL Filtering**
> **What it is:** Removing pages from the dataset based on their web address, using block lists of unwanted domains.
> **Why it was said in this video:** It is the first filtering stage in the FineWeb pipeline.
> **Why it is important:** It is a cheap and early way to keep harmful or low-value content out of the training data.

**b) Text extraction**
- The crawlers save the **raw HTML** of web pages.
- When you inspect a page, you see markup, lists, CSS and other code, which is "computer code almost".
- What we want is **only the text** of the page, not navigation menus and similar parts.
- This needs a lot of **filtering, processing and heuristics** to keep just the good content.

> **Keyword: Text Extraction**
> **What it is:** Turning raw HTML pages into clean readable text by removing markup, CSS and navigation.
> **Why it was said in this video:** Crawlers store raw HTML, but the model only needs the text.
> **Why it is important:** If code and menus stay in the data, the model wastes training effort on content that is not useful.

**c) Language filtering**
- FineWeb uses a **language classifier** that guesses the language of each page.
- It keeps only pages that are **more than 65% English** (as an example).
- This is a **design decision** that each company can make for itself.
- Effect on the model:
  - If you remove all Spanish, the model will later **not be very good at Spanish**, because it has never seen much of it.
  - Different companies focus on **multilingual performance** to different degrees.
  - FineWeb is focused on English, so a model trained on it would be very good at English but maybe not at other languages.

> **Keyword: Language Filtering**
> **What it is:** Using a classifier to guess each page's language and keeping only the languages you want.
> **Why it was said in this video:** To show that the data mix directly affects what the final model is good at.
> **Why it is important:** The languages in the training data decide the languages the model can handle well.

**d) Other filters**
- After language filtering there are a few more filtering steps.
- **Deduplication** is one of them.
- The last one mentioned is **PII removal**.

> **Keyword: Deduplication**
> **What it is:** Removing repeated or near-identical documents from the dataset.
> **Why it was said in this video:** It is listed as one of the extra filtering steps after language filtering.
> **Why it is important:** The speaker only names the step in this part of the video. In general, it avoids spending training effort on the same content many times.

> **Keyword: PII (Personally Identifiable Information) Removal**
> **What it is:** Detecting information such as addresses and Social Security numbers and filtering out those pages.
> **Why it was said in this video:** It is the final cleaning step he mentions in the pipeline.
> **Why it is important:** It helps protect people's private data from ending up in a model.

- He says he will not go into full detail, but this is a **fairly extensive part of preprocessing**.

### 2.5 What the final text looks like

- FineWeb can be downloaded by anyone from the Hugging Face page.
- Examples of final text shown:
  - An article about **tornadoes in 2012**.
  - An odd medical article about the **adrenal glands**.
- Think of these as **web pages filtered down to just their text**.
- The result is about **44 terabytes** of text.
- To give an intuition, the speaker took the **first 200 web pages**, joined (concatenated) all the text together and showed it:
  - It is just **raw internet text**.
  - Zooming out, it is a "massive **tapestry** of text data".
  - This text has many **patterns**.
- The next goal: **train neural networks** so they internalize and model **how this text flows**. In other words, build networks that **mimic** this giant texture of text.

---

## 3. Tokenization: Turning Text into Symbols

Before text can go into a neural network, we must decide how to represent it.

### 3.1 Why we need symbols

- Neural networks (with the current technology) expect a **one-dimensional sequence of symbols**.
- They also need a **finite set of possible symbols**.
- So we must:
  1. Decide what the symbols are.
  2. Represent all our data as a one-dimensional sequence of those symbols.
- Text is already a one-dimensional sequence (left to right, top to bottom), even if it looks two-dimensional on a screen.

### 3.2 Step 1: Raw bits

- Computers already store text in some form. With **UTF-8 encoding** you get the raw **bits** for the text.
- With bits you have exactly **two symbols: 0 and 1**, but a **very long sequence**.
- The key trade-off:
  - **Sequence length is a very finite and precious resource** in a neural network.
  - So we do not want extremely long sequences of just two symbols.
  - We want to **trade off vocabulary size against sequence length**: more symbols and shorter sequences.

> **Keyword: Vocabulary (Symbol Set)**
> **What it is:** The full list of different symbols the model can read and write.
> **Why it was said in this video:** The speaker explains that we choose how many symbols to use, and that this choice changes how long our sequences are.
> **Why it is important:** A bigger vocabulary gives shorter sequences, which saves the precious resource of sequence length.

### 3.3 Step 2: Bytes

- A simple way to shorten the sequence: group **8 bits into one byte**.
- 8 bits can be on or off in only **256 combinations**.
- So the sequence becomes **8 times shorter**, but now has **256 possible symbols** (values 0 to 255).
- He encourages you not to think of these as numbers, but as **unique IDs or unique symbols**. For example, imagine each one is replaced by a unique emoji: you get a sequence of emojis, with 256 possible emojis.

### 3.4 Step 3: Byte Pair Encoding (BPE)

- Production state-of-the-art models go **beyond bytes** to shrink the sequence even more, in return for more symbols in the vocabulary.
- This is done with the **Byte Pair Encoding algorithm**:
  - Look for **consecutive bytes or symbols that are very common**.
  - Example: the sequence **116 followed by 32** is very common.
  - **Mint a new symbol** (with ID 256) and rewrite every occurrence of that pair with the new symbol.
  - **Repeat** as many times as you wish.
  - Each time you add a new symbol, the sequence gets **shorter** and the vocabulary gets **larger**.
- In practice a good vocabulary size is about **100,000 symbols**.
- **GPT-4 uses 100,277 symbols.**

> **Keyword: Byte Pair Encoding (BPE)**
> **What it is:** An algorithm that finds common pairs of symbols and replaces them with a single new symbol, repeating this many times.
> **Why it was said in this video:** It is the method used to go from bytes (256 symbols) to a vocabulary of about 100,000 symbols.
> **Why it is important:** It makes sequences much shorter, so the model can handle more text within its limited sequence length.

### 3.5 What are tokens?

- The process of converting raw text into these symbols is called **tokenization**.
- The symbols are called **tokens**.

> **Keyword: Token / Tokenization**
> **What it is:** A token is one symbol from the vocabulary, usually a small chunk of text. Tokenization is the conversion of text into tokens (and back).
> **Why it was said in this video:** Tokens are the "atoms" that the neural network actually reads and writes, not letters or words.
> **Why it is important:** Many strange behaviors of LLMs relate to tokens. The speaker says he will come back to tokenization later in the video.

### 3.6 Exploring GPT-4's tokenizer (Tiktokenizer website)

- The speaker uses a website called **Tiktokenizer**.
- He selects **cl100k_base**, which is the **GPT-4 base model tokenizer**.
- Examples he shows:
  - `hello world` becomes **exactly two tokens**: "hello" (ID 15339) and " world" (ID 1917). Note the space is part of the second token.
  - Changing it slightly (for example, joining the words, or adding **two spaces**) gives a **different tokenization**, with a new token (220) appearing for the extra space.
  - It is **case sensitive**: a capital "H" is different. "Hello world" with a capital H ends up as **three tokens** instead of two.
- A single line of text he shows becomes a sequence of **length 62** tokens.
- The token IDs are only **unique labels**; the numbers themselves have no meaning.

---

## 4. Neural Network Input and Output (What We Actually Train)

Now the whole dataset is represented as tokens, and the speaker explains what the network sees and predicts.

### 4.1 The dataset in tokens

- FineWeb is 44 TB of disk space, which is about a **15 trillion token** sequence.
- Every token is a little text chunk, an "atom" of the sequence. The numbers are just unique IDs.

### 4.2 Taking windows of tokens

- We want to model the **statistical relationships of how tokens follow each other**.
- Method:
  - Take a **window of tokens** from the data, fairly randomly.
  - The window length can range from **zero tokens up to a maximum size** we choose.
  - In practice you might see windows of about **8,000 tokens** (or 4,000, or 16,000).
  - In principle, windows could be any length, but very long windows are **computationally expensive**, so we crop at some number.
- In his example, he takes a window of **four tokens** so that it fits on the screen.

> **Keyword: Context / Context Window (Context Length)**
> **What it is:** The tokens you give the model as input before it predicts the next token. The maximum number it can take is the maximum context length.
> **Why it was said in this video:** The window of tokens is called the "context", and it is the input to the neural network.
> **Why it is important:** The model can only use as much history as fits in its context length. Longer context gives better predictions but costs more compute.

### 4.3 Predicting the next token

- The four tokens (the **context**) go into the neural network as **input**.
- The task: **predict the token that comes next** in the sequence. In his example, the correct next token is **3962**.
- The **output** of the network:
  - Because the vocabulary has **100,277** tokens, the network outputs **exactly that many numbers**.
  - Each number is the **probability** that the matching token comes next.
- At the start, the network is **randomly initialized**, so its probabilities are random too.
  - Example: one token gets 4%, another 2%, and the correct token 3962 gets only 3%.

### 4.4 Learning from the correct answer

- Because the window came from our dataset, **we know the correct answer** (the label).
- We want the probability of the correct token (3% here) to go **up** and the probabilities of all other tokens to go **down**.
- There is a **mathematical process** for working out how to adjust the network so the correct token gets a slightly higher probability.
- After one update, the same input would give, for example, 4% for the right token and lower values for others.
- Important points:
  - This happens **at every token position** in the sampled windows, not only for one example.
  - It happens **in parallel, in large batches** of tokens.
  - **Training** is simply a long sequence of such updates.
  - The goal is that the network's predictions become **consistent with the statistical patterns** of how tokens follow each other in the training data.

> **Keyword: Next-Token Prediction**
> **What it is:** The training task of guessing which token comes next, given the tokens before it.
> **Why it was said in this video:** It is the single task that the network is trained on during pre-training.
> **Why it is important:** From this simple task, the model ends up learning a lot about language and the world.

---

## 5. Neural Network Internals

A short look inside the network, without going deep into the maths.

- **Inputs** are token sequences of length zero up to some maximum (such as 8,000).
- In principle the input could be infinitely long, but that is too expensive, so we crop it. That crop becomes the **maximum context length** of the model.
- The inputs are mixed in **one giant mathematical expression** together with the **parameters (weights)** of the network.
- Modern networks have **billions of parameters**. At the start, they are **set randomly**, so the predictions are random.
- **Training** means iteratively updating the parameters so the outputs match the patterns in the training set.
- Analogy: the parameters are like **knobs on a DJ set**.
  - Twiddling the knobs gives different predictions for every possible input.
  - Training means finding a setting of the knobs that is consistent with the statistics of the training set.

> **Keyword: Parameters (Weights)**
> **What it is:** The numbers inside a neural network that are adjusted during training. They decide how inputs turn into outputs.
> **Why it was said in this video:** The speaker explains that training is just finding good values for these numbers.
> **Why it is important:** All the "knowledge" of the model is stored in its parameters. Model size is usually described by how many parameters it has.

> **Keyword: Neural Network**
> **What it is:** A mathematical function that takes inputs, mixes them with many adjustable numbers, and gives outputs.
> **Why it was said in this video:** It is the model that learns to predict the next token.
> **Why it is important:** The speaker wants to show it is "not very scary": it is a function made of simple operations.

### 5.1 The mathematical expression

- Modern networks are expressions with **trillions of terms** (as stated in the video), but the idea is simple.
- Inputs (like x1, x2) are mixed with weights (w0, w1, w2 and so on) using simple operations: **multiplication, addition, exponentiation, division** and so on.
- **Neural network architecture research** is about designing expressions that are:
  - **Expressive**
  - **Optimizable**
  - **Parallelizable**
- But at the end of the day, "these are not complex expressions".

### 5.2 The Transformer

- The speaker shows a website with a visualization of a production-grade network.
- This network is called the **Transformer**. The example shown has roughly **85,000 parameters** (as stated in the video).
- Flow of information:
  - Tokens come in at the top.
  - Each token is **embedded** into a "distributed representation", meaning each token has a **vector** that represents it inside the network.
  - Values flow through layers: **layer norms, matrix multiplications, softmax**, an **attention block**, then a **multi-layer perceptron block**, and so on.
  - At the end, the output gives the **predictions for the next token**.
- The intermediate values can be loosely thought of as the "**firing rates of synthetic neurons**".

> **Keyword: Transformer**
> **What it is:** The special kind of neural network design used in modern LLMs.
> **Why it was said in this video:** It is the specific structure of the production-grade network shown on the visualization website.
> **Why it is important:** GPT-2, GPT-4 and Llama are all Transformers, so this is the core design behind today's LLMs.

> **Keyword: Embedding**
> **What it is:** Turning each token into a vector (a list of numbers) that the network can work with.
> **Why it was said in this video:** The first step inside the Transformer is to embed the tokens into a distributed representation.
> **Why it is important:** A token ID by itself is just a label. The vector gives the network something it can do maths on.

- **Caution about the neuron analogy**:
  - These are **extremely simple neurons** compared to biological neurons.
  - Biological neurons are complex dynamical processes that have **memory**.
  - This expression has **no memory**; it is a fixed mathematical expression from input to output, **stateless**.
  - Still, you can loosely think of it as a "synthetic piece of brain tissue".
- The main point the speaker wants you to take away:
  - It is a **mathematical function**, controlled by a fixed set of parameters, that turns inputs into outputs.
  - As we twiddle the parameters we get different predictions.
  - We need to find a **good setting** so predictions match the patterns in the training set.

---

## 6. Inference: Generating New Text from the Model

After training, the next major stage is **inference**.

- In inference, we **generate new data from the model** to see what patterns it has internalized in its parameters.
- Step-by-step process:
  1. Start with some tokens as a **prefix** (his example: token 91).
  2. Feed them into the network. It gives a **probability vector** over all tokens.
  3. **Sample** one token from this distribution, like **flipping a biased coin**. Tokens with higher probability are more likely to be picked.
  4. **Append** the sampled token to the sequence (his example: token 860 came next).
  5. Feed the longer sequence back in, and sample again. Repeat.
- In his example, the first sampled tokens matched the training data (860 follows 91, then 287), but the fourth token sampled was **13659 ("article")** instead of **3962** from the original data.
- This shows that:
  - The system is **stochastic**: we sample and flip coins.
  - Sometimes we get lucky and **reproduce a small chunk** of training text.
  - Sometimes we get a token that was **not verbatim part of any training document**.
  - Once a different token gets in, the next tokens depend on it, so the stream quickly becomes **very different** from the training documents.
- So outputs are **remixes** of the training data. They have **similar statistical properties** but are **not identical** to the training data. They are "inspired by" it.
- "Article" is a plausible token in that context, so it may well have followed that context somewhere in the training documents.

> **Keyword: Inference**
> **What it is:** Using a trained model to generate new tokens, one at a time, by sampling from its predicted probabilities.
> **Why it was said in this video:** It is the second major stage of working with these networks, after training.
> **Why it is important:** Every time you chat with an LLM, you are only doing inference.

> **Keyword: Stochastic (Sampling)**
> **What it is:** Involving randomness. Here the next token is chosen randomly according to the model's probabilities, not always the top one.
> **Why it was said in this video:** The speaker shows that the same prefix can lead to different outputs each time.
> **Why it is important:** It explains why LLMs can give different answers to the same question.

### 6.1 Training vs inference in practice

- Downloading and tokenizing the internet is a **pre-processing step done once**.
- Then you can train **many different networks** with different settings, arrangements and sizes.
- Once you have a trained network with a set of parameters you like, you can use it for inference.
- When you use ChatGPT:
  - The model was **trained by OpenAI many months ago, probably**.
  - They have a specific set of weights that work well.
  - Everything you do in the chat is **inference only**. **No more training happens**. The parameters are **held fixed**.
  - You give it tokens, and it **completes the token sequence**.

---

## 7. GPT-2: A Concrete Example of Training and Inference

The speaker uses OpenAI's **GPT-2** to make training concrete.

### 7.1 Why GPT-2?

- **GPT** stands for **Generatively Pre-trained Transformer**.
- GPT-2 is the second version of the GPT series from OpenAI, published in **2019** in a paper.
- ChatGPT today is based on **GPT-4**, the fourth version.
- He likes GPT-2 because it is the **first time a recognizably modern stack came together**. All the pieces are recognizable today; "everything has just gotten bigger".

> **Keyword: GPT (Generatively Pre-trained Transformer)**
> **What it is:** The name of OpenAI's model series. It is a Transformer that is pre-trained to generate text.
> **Why it was said in this video:** The speaker explains the name and uses GPT-2 as his main teaching example.
> **Why it is important:** The name describes the three ideas behind the model: it generates, it is pre-trained, and it is a Transformer.

### 7.2 GPT-2 compared to today

| Property | GPT-2 (2019) | Modern models (as said in the video) |
|---|---|---|
| Architecture | Transformer | Transformer |
| Parameters | 1.6 billion (he also says ~1.5 billion when talking about the released model) | Closer to a trillion, or several hundred billion |
| Maximum context length | 1,024 tokens | A couple hundred thousand, maybe even a million |
| Training data | About 100 billion tokens | e.g. FineWeb has 15 trillion tokens |

- A bigger context means the model has more history to use, so it can make **better predictions** about the next token.

### 7.3 The cost of training

- In 2019, training GPT-2 was estimated to cost about **$40,000**.
- The speaker reproduced GPT-2 in his project **llm.c**. It took about **one day** and about **$600**, and he says it could probably come down to about **$100**.
- Why costs fell:
  1. **Datasets are better**: filtering, extraction and preparation are more refined, so quality is higher.
  2. **Hardware is much faster** (biggest difference).
  3. **Software** is much better at squeezing speed out of the hardware.

### 7.4 What a training run looks like

- Each **line** in the log is **one update** to the model, where the parameters change slightly so it is better at predicting the next token.
- Each update improves predictions on **1 million tokens** at once.
- The number to watch is the **loss**:
  - It is a **single number** that tells you how well the network is doing.
  - **Low loss is good.**
  - The loss decreases as updates continue. A researcher watches it and makes sure it keeps improving ("twiddling your thumbs and drinking coffee").
- Run details given:
  - About **1 million tokens per update**, each update takes about **7 seconds**.
  - Total of **32,000 steps**, which is about **33 billion tokens**.
  - He is at about step 420, so a bit more than 1% done.

> **Keyword: Loss**
> **What it is:** A single number that measures how wrong the network's predictions currently are. Lower is better.
> **Why it was said in this video:** It is the number a researcher watches during training to know whether the model is improving.
> **Why it is important:** It is the main signal of whether training is working.

- Every 20 steps, he runs **inference** to see samples:
  - At **step 1** the output is **completely random text**, because the network is random.
  - At about **step 420 (1% of training)**, the text is still gibberish but has a bit of **local coherence**.
  - If he waited for all 32,000 steps, the model would produce **fairly coherent English**.
- The run needs about **a day or two more**, and he just checks that loss is decreasing.

### 7.5 The compute behind it

- He cannot run this on a laptop. It runs on a **cloud computer**.
- The machine is an **8x H100 node**: eight H100 GPUs in a single computer.
- He rents it from **Lambda** (other companies offer this service too). The price shown is about **$3 per GPU per hour**.
- **Why GPUs?**
  - Training is very computationally expensive.
  - But it shows a lot of **parallelism**: many independent workers can do **matrix multiplications** at the same time.
- Scale-up: one GPU → eight GPUs in a node → many nodes → an entire **data center**.
- Big tech companies want these GPUs so they can train LLMs. The speaker links this to Nvidia's stock price reaching about **$3.4 trillion** (as he said at the time). He calls it "the **Gold Rush**".
- All these GPUs are collaborating on **one job: predicting the next token** on a dataset like FineWeb.
  - More GPUs → more tokens processed → faster iteration → bigger networks can be trained.
- He gives the example of Elon Musk getting **100,000 GPUs in a single data center**, which take a lot of power.

> **Keyword: GPU (Graphics Processing Unit)**
> **What it is:** A chip that can do many calculations at the same time. It is very good at the matrix multiplication used in neural networks.
> **Why it was said in this video:** The speaker explains what hardware is used to train models and why it costs so much.
> **Why it is important:** Access to GPUs is a main limit on who can train large models, and it drives the cost.

---

## 8. Base Models and Llama 3.1 Inference

Since the speaker cannot spend tens or hundreds of millions of dollars on training, he looks at models released by big companies.

### 8.1 What is a base model?

- Many companies train these models, but **few release the base model**.
- The model that comes out at the end of pre-training is a **base model**:
  - It is an **"internet text token simulator"**.
  - It is **not an assistant**. You cannot ask a question and expect an answer.
  - It only creates **remixes** of internet text. It "dreams internet pages".
- Base models are rarely released because they are only **step one** of several steps needed to build an assistant.

> **Keyword: Base Model**
> **What it is:** The model that results from pre-training. It continues text in the style of the internet but does not answer questions like an assistant.
> **Why it was said in this video:** The speaker explains what you get at the end of pre-training and why it is not yet ChatGPT.
> **Why it is important:** It is the foundation that later stages build on to make a helpful assistant.

### 8.2 What does a model release include?

You need **two things**:

1. **The code** (often Python), usually a couple hundred lines, that describes the exact sequence of operations of the network. This is the **forward pass**. It is fairly standard and understandable.
2. **The parameters**: this is **where the actual value is**. For GPT-2 this is roughly 1.5 billion numbers: one long list of numbers that is a very good setting of all the "knobs".

- GPT-2 was released in 2019 (the GitHub repository is shown). It is an older model.

### 8.3 Llama 3

- **Llama 3** is a much bigger and more modern model, trained and released by **Meta**.
- The biggest base model is **Llama 3.1 405 billion parameters**, trained on **15 trillion tokens**, in the same way as GPT-2 but much bigger. (The transcript text sometimes says "45 billion", which looks like a transcription error for 405 billion.)
- Meta also released an **Instruct model**:
  - "Instruct" means it is an **assistant**: you ask questions and it answers.
  - This is covered later in the video.
- For now, the speaker plays with the **base model**.

> **Keyword: Instruct Model**
> **What it is:** A version of the model that has been further trained to behave like an assistant, answering questions and following instructions.
> **Why it was said in this video:** It is foreshadowing for later sections: Meta released both the base and the instruct versions of Llama 3.1.
> **Why it is important:** The instruct model is the one that feels like ChatGPT. The base model does not.

### 8.4 Playing with the base model

- He uses a company called **Hyperbolic** that serves the Llama 3.1 405B **base** model. He stresses you must choose the **base** model.
- He sets max tokens to **128** to save compute.
- What happens is the **same as inference**: it continues the token sequence of whatever prefix you give.

**Experiment 1: "What is 2 plus 2?"**
- The model does **not** reply "four, what else can I help with?".
- The prompt is tokenized and acts as a **prefix**; the model is a "glorified autocomplete", a **very expensive autocomplete** based on statistics of web pages.
- In one run it answered and then drifted into philosophical territory. In others it just went elsewhere.
- Each run gives a **different** output, because the system is **stochastic**: for the same prefix we get a different continuation each time.
- Conclusions:
  1. It is **not an assistant yet**: it is a token autocomplete.
  2. It is a **stochastic system**.

### 8.5 The model as a "compression of the internet"

- Even though it is not directly useful as an assistant, the base model has **learned a lot about the world** by predicting next tokens and **stored that knowledge in its parameters**.
- Analogy: the 405 billion parameters are like a **zip file of the internet**, but it is **lossy** compression, not lossless. We are left with a "gist" of the internet that we can generate from.
- We can **elicit** knowledge by prompting well.

**Experiment 2: "Top 10 landmarks to see in Paris"**
- He writes a prompt that primes the model to continue a list.
- It produces a list of landmarks.
- Cautions:
  - You **cannot fully trust** the information. It is a **vague recollection** of internet documents.
  - Things that occur **frequently** in the data are more likely to be remembered correctly than rare things.
  - The knowledge is **vague, probabilistic and statistical**, not stored explicitly or exactly.
  - He cannot personally verify the content, but it seems approximately correct.

**Experiment 3: Zebra Wikipedia entry**
- He pastes the first sentence of the Wikipedia page for "zebra".
- The model produces what looks like an **exact recitation** of the Wikipedia article, from memory.
- Over time (for example, within 512 tokens), it will eventually **deviate**, because it cannot remember everything exactly.

> **Keyword: Regurgitation (Memorization)**
> **What it is:** When a model repeats training text almost word-for-word from memory.
> **Why it was said in this video:** The zebra example shows the model reciting Wikipedia from its parameters.
> **Why it is important:** It is usually **undesirable** in a final model, to cite directly the things you were trained on. It also shows how strongly a model can memorize.

- **Why it happens:**
  - High-quality sources such as Wikipedia are **preferentially sampled** during training.
  - The model has probably seen such a page about **10 times** (several **epochs**).
  - Like a person who reads a text 100 times and can recite it. Models can be **more efficient per exposure** than humans: about 10 views is enough to memorize it.

> **Keyword: Epoch**
> **What it is:** One pass over the data. Doing "a few epochs" on a source means the model saw it several times.
> **Why it was said in this video:** To explain why Wikipedia pages get memorized: they were seen about 10 times.
> **Why it is important:** The more often the model sees the same text, the more likely it is to memorize it.

**Experiment 4: Something from the future (knowledge cutoff)**
- The Llama 3 paper says the pre-training data has a **knowledge cutoff at the end of 2023**.
- So the model has never seen anything about the **2024 US election**.
- He primes it with a sentence about the Republican Party and Trump being President from 2017, and lets it continue.
- In one sample it said the running mate was **Mike Pence**, against **Hillary Clinton and Tim Kaine**. In another sample the running mate was **Ron DeSantis** against **Joe Biden and Kamala Harris**.
- These are "**parallel universes**": the model makes **educated guesses** from its parameters.

> **Keyword: Knowledge Cutoff**
> **What it is:** The date after which the model's training data has no information.
> **Why it was said in this video:** To show what happens when you ask about events the model never saw.
> **Why it is important:** The model does not know about events after its cutoff, and may confidently guess.

> **Keyword: Hallucination**
> **What it is:** When a model produces content that sounds plausible but is made up, because it is making its best probabilistic guess.
> **Why it was said in this video:** The speaker says everything in the election examples is hallucination: the model takes its best guess.
> **Why it is important:** It is one of the main "sharp edges" of LLMs. You must not blindly trust the output.

**Experiment 5: Few-shot prompt for translation**
- Even a base model can be used in **practical applications** if the prompt is clever.
- He builds a prompt of **10 English → Korean word pairs**, followed by a new English word ("teacher") and a colon.
- The model completes with the right Korean translation.
- This works because of **in-context learning**: while reading the context, the model learns "in place" that there is a pattern and continues it. It takes on the role of a translator.

> **Keyword: In-Context Learning**
> **What it is:** The model's ability to pick up a pattern from examples in the prompt and continue it, without changing its parameters.
> **Why it was said in this video:** It is why the translation prompt works on a base model.
> **Why it is important:** It lets you build useful applications just by designing prompts.

> **Keyword: Few-Shot Prompt**
> **What it is:** A prompt that includes a few worked examples so the model can copy the pattern for a new input.
> **Why it was said in this video:** The 10-pair translation list is a few-shot prompt.
> **Why it is important:** It is a simple, powerful way to steer a base model to do a task.

**Experiment 6: Making an assistant using only a prompt**
- Trick: write a prompt that looks like a **web page of a conversation between a helpful AI assistant and a human**.
- The speaker asked ChatGPT itself to write the prompt (which he calls "kind of meta").
  - It describes an AI assistant that is knowledgeable, helpful and capable of answering many questions.
  - A description alone is not enough; it works **much better with a few-shot set of example turns** (human / assistant).
- He adds "Human: Why is the sky blue?" at the end, and the base model replies as the assistant (Rayleigh scattering).
- It is a bit subtle: the model continues the sequence, so after answering it **hallucinates the next human question** and goes on and on. But the task is accomplished.
- If you simply type "Why is the sky blue?" with no such framing, you don't expect it to work with a base model, and you just get more questions.
- So this is one way to create an assistant even if you **only have a base model**.

---

## 9. Transition: From Pre-training to Post-training

- The speaker zooms out to summarize what has been covered so far: **we wish to train LLM assistants like ChatGPT**, and so far we have discussed the **pre-training stage**.
- (The provided transcript ends here, at about the 59-minute mark.) The base model is only "step one"; the remaining steps to get a true assistant, including the **instruct** behavior he foreshadowed, are covered later in the video and are not part of this transcript.

---

## Key Takeaways

1. **LLMs start from the internet.** Pre-training uses a huge, filtered, high-quality and diverse set of web text (for example FineWeb, about 44 TB or about 15 trillion tokens).
2. **Data cleaning matters.** URL filtering, text extraction, language filtering, deduplication and PII removal shape what the model knows and which languages it is good at.
3. **Models read tokens, not words or letters.** Byte Pair Encoding builds a vocabulary of around 100,000 symbols (GPT-4 uses 100,277) to keep sequences short.
4. **The core training task is simple:** predict the next token from a context window. Parameters are nudged so that the correct next token gets a higher probability, in huge parallel batches.
5. **A neural network is a big mathematical function** with billions of adjustable parameters ("knobs"). The Transformer is the architecture used in production.
6. **Training and inference are different.** Training changes parameters. Inference only samples tokens using fixed parameters. When you chat with ChatGPT, it is all inference.
7. **Outputs are stochastic.** The same prompt can give different results, and generated text is a "remix" that is statistically similar to, but not identical to, the training data.
8. **GPT-2 shows how much has changed:** 1.6B parameters, 1,024-token context and ~100B training tokens in 2019, cost about $40,000 then, but can be reproduced for about $600 or less today thanks to better data, faster hardware and better software.
9. **Compute is the "gold rush".** GPUs (like the H100) are used because training is highly parallel matrix multiplication. More GPUs mean faster iteration and bigger models.
10. **A base model is not an assistant.** It is an internet text token simulator and a lossy "zip file" of the internet.
11. **The knowledge in a base model is vague and probabilistic.** Frequent facts are better remembered, rare facts are shaky, and the model can memorize (regurgitate) high-quality sources like Wikipedia.
12. **Models hallucinate.** For things after the knowledge cutoff (like the 2024 election), the model makes educated guesses rather than knowing.
13. **Prompting can unlock useful behavior** even from a base model, through few-shot prompts and in-context learning, including a prompt that makes it act like an assistant.
14. **This is only the first stage.** Turning a base model into a proper assistant requires further steps, which come after this part of the video.