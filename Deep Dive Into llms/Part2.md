# How LLM Assistants Are Built: Post-Training, Hallucinations, Tools, "Thinking in Tokens" and Reinforcement Learning (Part 2)

**Source video:** [Video link](https://www.youtube.com/watch?v=7xTGNNLPyMI&t=3563s) | **Covered range:** 00:59:23 → ~02:27:00+ (transcript ends mid-way through the RL summary)

---

## Overview

This part of the video starts after **pre-training** is finished. We have a **base model**, which is only an "internet document simulator". The speaker explains how we turn it into a useful **assistant** using **post-training** (supervised fine-tuning on conversations). He then explains the "psychology" of these models: why they hallucinate, how tools like web search fix this, why they do not know who they are, why they need many tokens to "think", and why they fail at spelling and counting. The last part introduces **reinforcement learning (RL)**, where the model practices problems and discovers for itself which token sequences lead to correct answers. The whole pipeline is compared to how children learn from a textbook: exposition, worked solutions, practice problems.

---

## Table of Contents

1. [From Pre-training to Post-training (00:59:23)](#1-from-pre-training-to-post-training)
2. [Post-training Data: Conversations (01:01:06)](#2-post-training-data-conversations)
3. [Hallucinations, Tool Use, Knowledge vs Working Memory (01:20:32)](#3-hallucinations-tool-use-knowledge-vs-working-memory)
4. [Knowledge of Self (01:41:46)](#4-knowledge-of-self)
5. [Models Need Tokens to Think (01:46:56)](#5-models-need-tokens-to-think)
6. [Tokenization Revisited: Spelling Problems (02:01:11)](#6-tokenization-revisited-models-struggle-with-spelling)
7. [Jagged Intelligence (02:04:53)](#7-jagged-intelligence)
8. [From Supervised Fine-Tuning to Reinforcement Learning (02:07:28)](#8-supervised-fine-tuning-to-reinforcement-learning)
9. [Reinforcement Learning (02:14:42)](#9-reinforcement-learning)
10. [Key Takeaways](#key-takeaways)

---

## 1. From Pre-training to Post-training

**What was discussed:** The speaker zooms out and recaps stage 1 (pre-training). Then he introduces stage 2 (post-training) and explains why we need it.

### Recap of pre-training
- We take **internet documents**, break them into **tokens** (small text chunks, "atoms" of text), and train a neural network to **predict token sequences**.
- The output of this stage is the **base model**. It is the set of parameters (the "settings") of the network.
- The base model is basically an **internet document simulator at the token level**. It generates token sequences that have the same statistics as internet documents.
- It can be used in some applications, but it is not what we want. We want an **assistant**: we ask a question and get an answer.

> **Keyword: Base Model**
> **What it is:** The model you get after pre-training. It continues text the way internet documents would continue, but it does not "answer questions" on purpose.
> **Why it was said in this video:** The speaker needed to show where post-training starts. The base model is the input to post-training.
> **Why it is important:** A base model is useful as raw knowledge, but it is not a helpful assistant. Everything we call "ChatGPT-like" behavior is added after this point.

> **Keyword: Pre-training**
> **What it is:** The first, huge training stage where the model reads a very large amount of internet text and learns to predict the next token.
> **Why it was said in this video:** To remind us what was already done before we move to the second stage.
> **Why it is important:** This is where most of the compute, the large data centers and "millions of dollars" are spent. It builds the model's general knowledge.

### What is post-training?
- Post-training is the **second stage**. We take the base model and "hand it off" to post-training.
- It is **computationally much cheaper** than pre-training, but still **extremely important**.
- Goal: turn the LLM from something that samples internet documents into an **assistant that gives answers to questions**.

> **Keyword: Post-training**
> **What it is:** All the extra training done after pre-training to shape the model's behavior (for example, into a helpful assistant).
> **Why it was said in this video:** It is the main topic of this part of the video.
> **Why it is important:** It is what turns a text-continuation engine into a product people can talk to.

---

## 2. Post-training Data: Conversations

**What was discussed:** What conversations look like as training data, who writes them, how they are turned into tokens, and what this means for "what you are really talking to" when you use ChatGPT.

### 2.1 The idea: program the assistant by example
- We now think in terms of **conversations**. They can be **multi-turn** (many back-and-forth messages). The simplest case is a human talking to an assistant.
- Examples given:
  - Human: "What is 2 plus 2?" → Assistant: "2 plus 2 is 4."
  - Human follows up: "What if it was star instead of plus?" → the assistant answers.
  - An example where the assistant has a **nice personality**.
  - An example where the human asks for something we do not want to help with → the assistant gives a **refusal** ("I can't help with that").
- We want to "program" how the assistant behaves. But because this is a neural network, we **cannot write explicit code rules**. Everything is done through **training on data sets**.
- So we **implicitly program** the assistant by **creating a data set of conversations**. The assistant is "**programmed by example**".
- A real data set can have **hundreds of thousands of conversations** (multi-turn, long, covering many topics). The speaker only shows three as an illustration.

### 2.2 Where does the data come from?
- Originally from **human labelers**. They get a conversation context and write the **ideal assistant response**.
- The model then trains on this and **imitates** those responses.

### 2.3 How the training works
- We take the base model and **throw out the internet-documents data set**. We **replace it with the conversations data set**.
- We **continue training** with the **exact same algorithm**. Only the data set changes.
- The model **adjusts very quickly** and learns the statistics of how an assistant answers human queries.
- At inference time, we "prime" the assistant with the conversation, and it imitates what human labelers would have done.
- **Time cost:** Pre-training can take roughly **3 months** on many thousands of computers. Post-training is much shorter, for example **about 3 hours**, because the conversation data set is **much smaller** than the internet text.

> **Keyword: Supervised Fine-Tuning (SFT)**
> **What it is:** Continuing to train the base model on a smaller, carefully made data set of example conversations (input + ideal answer written by humans), so the model learns to copy that behavior.
> **Why it was said in this video:** This is the first and main step of post-training. The speaker later calls it "supervised fine-tuning" when summarizing.
> **Why it is important:** It is how an assistant "personality" and answer style get into the model. Algorithmically it is the same as pre-training; only the data differs.

### 2.4 Turning conversations into tokens
- Everything in these models must become **tokens**. So we need a way to **encode conversations as token sequences**.
- The speaker compares this to **TCP/IP**: there are precise **rules and protocols** for how data is laid out so everyone agrees. LLMs need similar data structures and rules to encode and decode conversations.
- Using the **Tiktokenizer** web tool, he shows a 2-turn user/assistant conversation. It ends up as a **one-dimensional sequence of 49 tokens**.
- Different LLMs use **slightly different formats**. It is "a bit of a wild west".
- **GPT-4o example (as described in the video):**
  - A special token `<|im_start|>` begins a turn.
  - Then the token that says **whose turn** it is (for example `user`).
  - Then a separator token.
  - Then the **tokens of the actual text** (for example "What is 2 plus 2?").
  - Then `<|im_end|>` closes the turn.
  - The speaker says "im" may stand for "imaginary monologue", but admits he does not actually know why it is called that.
- Important point: **`im_start` is not normal text**. It is a **new special token** that was never seen in pre-training. We **introduce it in post-training**.
- These special tokens are mixed in with the text so the model learns: "this is the start of a turn", "this turn belongs to the user", "now it is the assistant's turn", and so on.
- The exact details are **not important**. The important point: a structured conversation becomes a **1D token sequence**. After that, **everything we did before still applies**: we train the model to predict the next token.

> **Keyword: Special Tokens**
> **What it is:** Extra tokens added to the vocabulary that are not normal words. They act as markers, such as "start of turn", "end of turn", or later "search start/search end".
> **Why it was said in this video:** To show how a conversation (which has structure) is represented as a flat token sequence the model can train on.
> **Why it is important:** They give the model a protocol for the conversation. They are the base for chat formatting and for tool use later in the video.

> **Keyword: Tokenization**
> **What it is:** Breaking text into small chunks (tokens), each with a number ID, which is the only thing the model actually sees.
> **Why it was said in this video:** The speaker repeats that "everything is just a one-dimensional token sequence", even conversations.
> **Why it is important:** This explains many model behaviors, including later problems with spelling and counting.

### 2.5 What happens at inference (when you chat)
- Example: you have already had "What is 2 plus 2?" → "2 plus 2 is 4", and you send "What if it was times?".
- On the server (for example OpenAI's), the system builds the context: all previous turns + `<|im_start|>assistant<|im_sep|>` and **stops there**.
- Then it **starts sampling** from the model: "what is a good first token, a good second token…". The LLM takes over and generates the reply.
- The reply does not need to be identical to the training data, but it has the **flavor** of what was in the data set.

### 2.6 InstructGPT (OpenAI, 2022)
- The first major paper on this: **InstructGPT**. It was the first time OpenAI talked about **fine-tuning language models on conversations**.
- **Section 3.4** talks about **human contractors** hired (for example through **Upwork** or **Scale AI**) to create conversations.
- Labelers write **prompts** and also the **ideal assistant responses**.
- Example prompts: "List five ideas for how to regain enthusiasm for my career", "What are the top 10 science fiction books I should read next?", translation tasks into Spanish, etc.
- Labelers follow **labeling instructions** written by the company. At a high level: be **helpful, truthful and harmless**.
- In real life these instructions are not short. They can be **hundreds of pages**, and labelers study them professionally.
- The InstructGPT data set was **never released** by OpenAI.

> **Keyword: Labeling Instructions**
> **What it is:** A (very long) guide written by the company that tells human labelers how to write ideal assistant answers.
> **Why it was said in this video:** To show where the assistant's "personality" really comes from.
> **Why it is important:** The model's behavior is a statistical copy of what labelers wrote, so these instructions indirectly decide how the AI behaves.

### 2.7 Open-source example: Open Assistant
- A community effort (**OpenAssistant**) tried to repeat this. People on the internet were asked to write conversations.
- Example entry: a prompt asking for a short introduction to the relevance of a term in economics (with examples), then the ideal answer, then a follow-up like "now explain it to a dog" with a simpler answer.
- These become the labels the model trains on.

### 2.8 What the model learns
- We cannot cover every possible future question. But with enough examples (for example **~100,000** conversations), the model picks up the **statistical pattern** and adopts the **persona of a helpful, truthful, harmless assistant**.
- At test time, it might repeat a training answer exactly, but more likely it produces something with a **similar vibe** that fits what you want.

### 2.9 Modern state of the art: LLMs help create the data
- In the last 2–3 years, humans no longer do all the heavy lifting. **LLMs help create the conversation data sets**.
- It is rare for people to write a response fully from scratch. More often an LLM drafts an answer and a human **edits** it.
- Example: **UltraChat** is a modern data set, largely **synthetic**, probably with some human involvement (the speaker says he may be wrong on this detail).
- Modern SFT data sets have **millions of conversations**, mostly synthetic, probably edited a bit by humans, spanning many topics.
- They are often called **SFT mixtures** (a mix of different types and sources, partly synthetic, partly human).

> **Keyword: Synthetic Data**
> **What it is:** Training data that was generated by another AI model instead of being written fully by a human.
> **Why it was said in this video:** To explain how post-training data sets grew from thousands to millions of conversations.
> **Why it is important:** It makes building large data sets cheaper and faster, but the root of the data is still human curation.

### 2.10 "Dispelling the magic": what are you really talking to?
- When you ask ChatGPT something, the answer is **statistically aligned with the training set**, which has its seed in **humans following labeling instructions**.
- So you are **not talking to a magical AI**. You are talking to a **statistical simulation of a human labeler** (an "instantaneous simulation" of the kind of person hired).
- These labelers are **not random people**. Companies hire **experts** (for example for code questions), so the labeler is probably fairly highly skilled.
- **Example: "Recommend the top five landmarks to see in Paris."**
  - If this exact question is in the post-training data, you will very likely see an answer very similar to what the human labeler wrote. The labeler probably spent ~20 minutes researching and made a list.
  - If it is **not** in the data, the answer is more **emergent**: the model combines its **pre-training knowledge** (what landmarks are talked about on the internet) with the **post-training style** (how a labeler would answer).

---

## 3. Hallucinations, Tool Use, Knowledge vs Working Memory

**What was discussed:** The speaker starts "LLM psychology": the emergent cognitive effects of this training pipeline. First topic: hallucinations, where they come from, and two mitigations (teaching "I don't know" and using tools like web search).

### 3.1 What is a hallucination and why does it happen?

> **Keyword: Hallucination**
> **What it is:** When an LLM makes up information that is false but sounds confident.
> **Why it was said in this video:** It is the first "psychological quirk" of the training pipeline that the speaker explains.
> **Why it is important:** It is a major problem for LLM assistants. If you do not know it exists, you may trust wrong answers.

- Hallucinations were a big problem with early models from many years ago. They have improved, thanks to mitigations explained below.
- **Why it happens:** Imagine training conversations like "Who is Tom Cruise?" (famous actor), "Who is John Baraso?" (a US senator), "Who is Genghis Khan?" The human labeler either knew the answer or researched it, and wrote it in a **confident tone**.
- At test time, you ask "**Who is Orson Kovats?**" (a name the speaker invented randomly; he believes this person does not exist).
- The model does **not** say "I don't know". Even if some part of the network "knows" internally that it is not familiar with this person, the model **statistically imitates its training set**. In the training set, "Who is X?" questions are **always answered confidently**.
- So it gives its **statistically most likely guess** and makes things up. Remember: the model has **no internet access** and does **no research**. It is a "**statistical token tumbler**" that samples the next token.

### 3.2 Live demo with Falcon 7B
- He uses the **Hugging Face inference playground** and an **old model, Falcon 7B Instruct**, on purpose.
- Asking "Who is Orson Kovats?" multiple times gives different, wrong answers: "American author and science fiction writer", "fictional character from a 1950s TV show", "former minor league baseball player".
- This shows the model does not know. It samples from probabilities, in the **style** of answers it saw in training. To us it looks like made-up factual knowledge.
- A modern ChatGPT model **does not make up** an answer. It says it does not appear to know such a person. (The speaker told it not to use tools, so this came from the model's own knowledge.)

### 3.3 Mitigation 1: Teach the model to say "I don't know"
- We need training examples where the correct answer is "I don't know", but **only where the model really does not know**.
- So: how do we find the **boundary of the model's knowledge**? By **probing the model empirically**.
- The speaker explains how **Meta** did this for **Llama 3** (the paper calls hallucinations "**factuality**"). The procedure:
  1. Take a **random document** from the training set (his example: a Wikipedia featured article about a hockey player, Dominik Hašek) and a **paragraph** from it.
  2. Use an **LLM to generate factual questions and answers** from that paragraph. This works well because the text is in the LLM's **context window**, so it does not need to rely on memory.
  3. **Ask the model being tested** (he uses Mistral 7B as the example) the question, **several times** (for example 3–5 times).
  4. Use **another LLM as a judge** to compare the model's answers against the correct answer. No humans are needed.
  5. If the model gets it right every time → it **knows** this fact (example: "Which team did he play for?" → Buffalo Sabres, correct 3 times).
  6. If the answers are wrong or inconsistent → it **does not know** (example: "How many Stanley Cups did he win?" Correct answer: 2. The model said 4, then something else, then "none").
  7. For the questions the model does not know, **add a new training conversation** where the answer is "I'm sorry, I don't know" (or "I don't remember").
- Do this for many questions and many documents. Even a **few examples** are enough for the model to learn the **link between its internal uncertainty and the words "I don't know"**.
- **Why this works:** The speaker's idea is that the network probably has an **internal "uncertainty" neuron/feature** that lights up when the model is unsure. But before this training, that internal signal is **not wired to the model's words**. The model shows confidence just like in its training set. SFT with "I don't know" examples creates that connection. (He says that empirically this "turns out to be probably the case".)
- This is a **large mitigation** and is part of why ChatGPT can say it does not know.

> **Keyword: LLM as a Judge**
> **What it is:** Using one language model to check (grade) the answers of another model against a known correct answer.
> **Why it was said in this video:** It lets the "probe the model" procedure run automatically with no humans.
> **Why it is important:** It makes large-scale checking of model knowledge cheap, and it is the base for finding what a model does not know.

### 3.4 Mitigation 2: Let the model use tools (web search)
- Better than saying "I don't know": give the model a chance to **look it up**.
- **Analogy:** The knowledge in the model's parameters is like a **vague recollection** of something you read a long time ago (like a month ago). If something is rare, your memory is weak. What you do is **go and look it up**. That refreshes your **working memory**, and then you can talk about it accurately.
- **How it works technically:**
  - Introduce **new special tokens**, for example `search_start` and `search_end`, plus a **format/protocol** for using them.
  - Instead of answering, the model can emit `search_start`, write a **search query**, then `search_end`.
  - The **program that runs inference** sees `search_end` and **pauses sampling** from the model. It opens a session with a search engine (for example Bing), pastes the query, retrieves the text from the results, and **copy-pastes that text into the context window** (possibly wrapped in special tokens).
  - Now the model continues generating, with that web text **directly in its context window**, so it can reference it easily.
- **How to teach the model to use the tool:** Again through **training data**. A few thousand examples of conversations showing when to search and how to write queries are enough. Because the model already understands the world from pre-training, it already has a good native idea of what a good search query is.
- **Demo in ChatGPT:** For "Who is Orson Kovats?" the model briefly shows "searching the web", then produces an answer **with citations** (sources and URLs), because the web text was placed in its context.
- The user can also say "**don't use any tools**" and ChatGPT will then answer from memory only.
- **Another example:** "How many Stanley Cups did Dominik Hašek win?" ChatGPT answers directly "twice" from memory, as it is confident. But you can **force a web search** to double check. It then searches, finds sources (like Wikipedia) and cites them.

> **Keyword: Tool Use**
> **What it is:** The model can emit special tokens to call outside programs (web search, code interpreter) and then use their results.
> **Why it was said in this video:** It is the second mitigation for hallucinations, and a general way to make models more reliable.
> **Why it is important:** It lets the model get fresh and exact information (or exact computation) instead of guessing from vague memory.

> **Keyword: Context Window**
> **What it is:** The text (tokens) that the model can see right now in the current conversation. It is fed directly into the neural network.
> **Why it was said in this video:** The speaker calls it the model's **working memory**, and web search results are pasted into it.
> **Why it is important:** Information in the context window is directly available and much more reliable than information stored in parameters.

### 3.5 Key mental model: parameters vs context window

> **Keyword: Knowledge in Parameters vs Working Memory**
> **What it is:** Two kinds of "memory" in an LLM. **Parameters** (the network weights) hold a vague, compressed recollection from training. The **context window** holds exact tokens that the model can see now.
> **Why it was said in this video:** The speaker says this is a "very important psychology point" and compares it to the human brain: what we remember long-term vs. what we experienced a few seconds or minutes ago.
> **Why it is important:** It changes how you should use LLMs. For specific facts or text, give the data in the prompt instead of relying on the model's memory.

- **Practical example (Pride and Prejudice):**
  - Prompt A: "Can you summarize chapter one of Jane Austen's Pride and Prejudice?" This works reasonably, because the model remembers a famous book well.
  - Prompt B (better): "Can you summarize chapter one… I am attaching it below for your reference", then paste the chapter text with a delimiter.
  - Prompt B is expected to give a **higher-quality summary**, because the model has **direct access** to the text and does not have to recall it. It is like you writing a better summary after re-reading the chapter.
- **Takeaway:** When you want an LLM to work with specific content, **put that content in the context**.

---

## 4. Knowledge of Self

**What was discussed:** Why asking an LLM "What model are you? Who built you?" is a bit nonsensical, and how developers can set a model's identity.

- People on the internet often ask LLMs "what model are you and who built you?".
- The speaker says the question is **nonsensical**, because the model is **not a person**:
  - It has **no persistent existence**. It "boots up, processes tokens, and shuts off".
  - It builds a context window for each conversation, then everything is deleted. It is **restarted from scratch** each conversation.
  - It has **no persistent self** and no real sense of self. It is a "token tumbler" that follows statistical regularities of its training data.
- **By default**, you get fairly random answers. Demo with **Falcon**: it says "I was built by OpenAI based on the GPT-3 model", which is made up.
- Many people take this as evidence the model was trained on OpenAI data. The speaker thinks **that is not necessarily true**. His explanation:
  - If nobody programs the model to answer identity questions, you get its **statistical best guess**.
  - During SFT the model takes on an "helpful assistant" persona, but it is **not told what label to give itself**.
  - Pre-training data (the whole internet) has a lot of text about **ChatGPT and OpenAI**. So the likely guess for "what am I?" is "ChatGPT by OpenAI".
- **Two ways developers can set the identity:**
  1. **Put it in the SFT data.** Example: the **OLMo** model from **Allen AI** (fully open source). Its SFT mixture has about **1 million conversations**, and **240 of them are hard-coded** identity conversations ("Tell me about yourself" → "I'm OLMo, an open language model developed by the Allen Institute for AI…", "What is your name?" → "OLMo project"). Training on ~240 of these makes the model repeat them later.
  2. **Use a system message.** A hidden message at the very start of the conversation that says things like "You are a model developed by OpenAI, your name is GPT-4o, you were trained on this date, your knowledge cutoff is…". When you open ChatGPT you see a blank page, but these tokens are **hidden in the context window**.
- The speaker says it is all "**cooked up and bolted on**". It is not deeply there like it would be for a human.

> **Keyword: System Message**
> **What it is:** A hidden instruction at the start of a conversation that tells the model who it is and how to behave. The user does not see it, but the model reads it.
> **Why it was said in this video:** It is one of the two ways to give a model a stable identity.
> **Why it is important:** It is a simple way for developers to control model behavior without retraining.

---

## 5. Models Need Tokens to Think

**What was discussed:** Why the way an answer is written in training data matters a lot for math/reasoning. Core idea: each token gets only a **limited, roughly fixed amount of computation**, so reasoning must be **spread across many tokens**.

### 5.1 The puzzle: two "correct" answers, one is much better
- Prompt: "Emily buys three apples and two oranges. Each orange costs $2. The total cost is $13. What is the cost of each apple?"
- Two possible assistant answers. **Both give 3**, which is correct. But:
  - **Answer A (bad):** Immediately says "The answer is $3", and then explains afterwards.
  - **Answer B (good):** Works step by step: the oranges cost $4, so 13 − 4 = 9, 9 ÷ 3 = 3, so the answer is $3.
- If you train with answers like A, the model can become **really bad at math**.

### 5.2 Why? A fixed amount of compute per token
- Models work on a **1D sequence of tokens, left to right**. For each next token, the whole sequence goes through the network, which outputs **probabilities for the next token**.
- There is only a **finite number of layers** (the demo network shows ~3 layers; modern state-of-the-art networks might have ~100). So the compute per token is **small and roughly fixed**.
- Therefore the model cannot do "arbitrary computation" in one forward pass. We must **distribute reasoning across many tokens**.
- In **Answer A**, the model must produce "3" in a single token right after "The answer is $". That means **cramming all the computation into one token**. Everything after that is only a **post-hoc justification**, since the answer is already in the context.
- Training on such answers teaches the model to **guess the answer in one token**, which will not work.
- In **Answer B**, we get **intermediate results** (each easy by itself). By the time the final answer appears, earlier results are in the "working memory" (context), so the last step is easy.

> **Keyword: Forward Pass**
> **What it is:** One run of the input tokens through all layers of the neural network to produce the probabilities for the next token.
> **Why it was said in this video:** To explain that the work per token is limited by the number of layers.
> **Why it is important:** It explains why a model cannot "think deeply" in a single token, and why showing steps helps.

> **Keyword: Intermediate Results / Distributing Computation**
> **What it is:** Writing out the small steps of a solution so that each token only needs to do a small amount of work.
> **Why it was said in this video:** This is the main reason "models need tokens to think".
> **Why it is important:** It makes answers more correct. The intermediate tokens are "not for you, they are for the model".

- The speaker notes ChatGPT already does the right thing: it defines variables, sets up the equation, and produces intermediate results. OpenAI has labelers who care about this. These steps are for the model, not just for the human.

### 5.3 Demo: forcing a single-token answer
- He asks ChatGPT the same problem but says: "**answer in a single token**, just immediately give me the answer".
  - For the simple numbers it worked (the answer "$3" is actually two tokens, but it was still done in one forward pass).
  - He made the numbers bigger ("23 apples and 177 oranges…") and again asked for an immediate answer. The model said **5, which is wrong**. It could not do the calculation in one forward pass.
  - When he allowed the normal step-by-step approach, the model went through the intermediate steps and arrived at the right answer, **7**.

### 5.4 Use tools instead of mental arithmetic
- Even with steps, the speaker does not fully trust the model's "**mental arithmetic**". Any intermediate step might fail, especially with large numbers. It is "kind of amazing" it can do it at all.
- So he suggests: tell the model "**use code**".
  - The model writes a program, which is sent to a **Python interpreter** (a different part of the computer). The program runs and the result is returned to the model.
  - He trusts it more because Python has more **correctness guarantees** than neural mental arithmetic. You can also inspect the code.
  - Like web search, the model uses **special tokens** for this. The code is not "computed" by the language model.

> **Keyword: Code Interpreter**
> **What it is:** A tool that lets the model write code and run it (for example in Python) and see the result.
> **Why it was said in this video:** It is a way to avoid unreliable mental arithmetic.
> **Why it is important:** It gives exact calculation instead of "neural-network guessing".

### 5.5 Counting also fails
- Models are bad at **counting** for the same reason: counting needs too much work in one token.
- Example: "How many dots are below?" with a long string of dots. The model tries to answer in one token. Looking at the tokenizer, groups of dots become single tokens (for example one token for ~20 dots). The model has to count from token IDs. It answers 161, but the true count is **177**.
- With "**use code**", it works (177). Why? The speaker broke the problem into easier parts. The model is **good at copy-pasting** (copying the tokens into a Python string), and the **Python interpreter does the counting** (`.count`).
- Lesson: **models need tokens to think. Do not rely on mental arithmetic. For counting tasks, ask them to use a tool.**

---

## 6. Tokenization Revisited: Models Struggle with Spelling

**What was discussed:** The speaker returns to tokenization to explain why models fail at character-level tasks.

- Models **do not see characters**. They see **tokens** (text chunks). Their whole world is tokens, unlike human eyes that see letters.
- **Example:** The word "ubiquitous". Task: print every third character, starting with the first. Expected: "u", then "q", etc. The model got it wrong.
  - Reason 1: a bit of mental-arithmetic failure.
  - Reason 2 (more important): in the tokenizer "ubiquitous" is **three tokens**. The model does not directly access individual letters. It had to learn, from scratch, which letters are inside which tokens.
- With "**use code**", it works (the result "uqs" looked correct to the speaker), because copy-pasting the word into Python is easy and Python handles characters.
- **Why use tokens at all?** Mostly **efficiency**. Many people would like to remove tokenization (character-level or byte-level models) but that creates **very long sequences** and nobody knows yet how to handle that well.
- **Famous example: "How many R's are in 'strawberry'?"**
  - It went viral because models that solve math-olympiad problems insisted there were only two R's.
  - Cause: combining **(1) not seeing characters** and **(2) being bad at counting**.
  - Now models answer correctly (3), and the speaker suspects OpenAI may have hardcoded the answer, but he is not sure.

---

## 7. Jagged Intelligence

**What was discussed:** Even strong models have strange failures on simple tasks. The speaker calls these "jagged edges".

- Models can solve complex math problems and answer PhD-level physics, chemistry and biology questions, but they sometimes fail at **super simple questions**.
- **Example: "Which is bigger, 9.11 or 9.9?"** The model sometimes says 9.11 is bigger. It may justify it, sometimes flip its answer later, and sometimes get it right. It is **not very reproducible**.
- **A possible explanation (second-hand):** The speaker says he has not read the paper himself, but he was told by the research team that when they examined the activations, a group of neurons associated with **Bible verses** lit up. In Bible-verse numbering, 9.11 comes **after** 9.9, so the model may find it "cognitively distracting". Even when it tries to justify with math it can still end with the wrong answer. He says this is **not fully understood**.
- **Conclusion:** Treat the model as a **stochastic system** that is "really magical" but **cannot be fully trusted**. Use it **as a tool**, not as something you let loose on a problem and then copy-paste the results.

> **Keyword: Jagged Intelligence**
> **What it is:** The uneven ability of LLMs: super-human on some hard tasks, but failing on some very simple ones, in ways that are hard to predict.
> **Why it was said in this video:** The speaker wants us to be aware of the "sharp edges" of the technology, even if we know how the models work.
> **Why it is important:** It reminds us to verify outputs and not assume "smart in one area means smart everywhere".

---

## 8. Supervised Fine-Tuning to Reinforcement Learning

**What was discussed:** A recap of the two major stages so far, and the introduction of the third stage, reinforcement learning, with an analogy to school.

### 8.1 Recap
- **Stage 1 – Pre-training:** Train on internet documents → **base model** (an internet document simulator). Takes many months on thousands of computers. It is a "lossy compression of the internet". Interesting, but not directly useful.
- **Stage 2 – Supervised fine-tuning (SFT):** **Algorithmically identical** to pre-training. Only the data set changes: to millions of **conversations** between a human and an assistant. Humans write the prompts and the ideal responses based on labeling documentation. Modern pipelines use LLM help, but it is fundamentally **human curation** at the end. The result is an **assistant**.
- **Cognitive effects discussed:**
  - Hallucinations are common unless you mitigate them.
  - Models can do a lot "in their head" but become better when they lean on **tools**: web search (less hallucination, fresher information) and code interpreter (write code, run it, see the result).

### 8.2 Reinforcement learning as the third stage
- RL is still considered part of **post-training**, but it is the **last, third major stage**, and it is a **different way** of training language models.
- Inside companies like OpenAI these are separate teams: one for pre-training data, one for pre-training itself, one for conversation data / SFT, and one for RL. It is a hand-off: base model → SFT assistant → RL.

### 8.3 The school/textbook analogy
A textbook has **three classes of information**:

| Textbook part | What it is | Equivalent LLM stage |
|---|---|---|
| **Exposition** (the bulk of the text, background knowledge) | Reading builds a knowledge base | **Pre-training** |
| **Worked problems with solutions** | An expert shows the full ideal solution; we imitate the expert | **SFT** (imitating expert human solutions) |
| **Practice problems** (problem + final answer in the answer key, but **no solution**) | You try many things yourself and discover what gets you to the answer | **Reinforcement learning** |

- Practice problems are **critical for learning**, because you must discover the solution yourself.
- In practice, you use the **background knowledge** (pre-training) and a bit of **imitation of experts** (SFT) to guide your attempts.
- In RL, the model gets **prompts and final answers**, but **no expert solutions**. It must practice and try things.

---

## 9. Reinforcement Learning

**What was discussed:** Why human-written solutions are not always the best training signal, and how RL lets the model discover its own best token sequences by trial and error.

### 9.1 The problem with human-written solutions
- Using the Emily apples/oranges problem again, the speaker shows **four possible candidate solutions** (set up equations, talk through it in English, skip right to the answer, etc.). All reach the answer 3.
- A solution has **two purposes**:
  1. **Reach the right answer** (the important one here).
  2. **Present it nicely to a human** (intermediate steps, clear explanation).
- If we only care about reaching the final answer (and doing so economically), **which solution is best for the LLM?** As a human labeler, **"I would not know"**.
- Why? **What is easy or hard for a human is different from what is easy or hard for an LLM.** Its cognition is different.
  - Example: in the shortest solution, the step "30 − 4 ÷ 3 =" asks for a lot of computation in **one token**. That might be too big a leap for the model, so it might make mistakes.
  - On the other hand, some steps that I write may be **trivial** for the LLM, and we waste tokens.
- This is a **pervasive issue**, not only in math:
  - The LLM may have **PhD-level knowledge** in math, physics and chemistry. In some ways it knows more than the labeler, and the labeler might not use that knowledge.
  - Or the labeler may **inject knowledge** that is not in the model's parameters. This creates **sudden leaps** that confuse the model.
- Conclusion: **We are not in a good position to write the best token sequences for the LLM.** Human-written solutions are useful by imitation to **initialize** the system, but the model needs to **discover for itself** which token sequences reliably lead to the answer. That happens through **reinforcement learning and trial and error**.

> **Keyword: Reinforcement Learning (RL)**
> **What it is:** A training method where the model tries many solutions on its own, gets checked on whether the final answer is correct, and is trained to do more of what worked.
> **Why it was said in this video:** It is the third and last major stage of training an LLM assistant.
> **Why it is important:** It lets the model find reasoning paths that work well **for its own "mind"**, instead of just copying human experts.

### 9.2 How RL works (simple version)
1. Take a **prompt** where we know the correct final answer (here: $3).
2. Let the model **generate a solution**. Check if it is correct.
3. Repeat many times. Every attempt is **different**, because at each token the model **samples** from a probability distribution, so it goes down slightly different paths.
4. The speaker demos this with **Gemma 2 (2 billion parameters)** in the Hugging Face playground. Three different attempts all reach $3. (This prompt is too easy: even a small model always gets it right. So he asks us to "use some imagination".)
5. In practice you might sample **thousands or even millions of solutions for a single prompt**. Some are correct, some are not.
6. **Cartoon example:** 15 solutions are generated. 4 reach the right answer (green), the rest fail (red).
7. We **encourage the solutions that led to correct answers**. We **train on those sequences**: the model becomes more likely to take similar paths in similar prompts in the future.
8. These training sequences are **not from human annotators**. They come from the **model itself** ("the model is practicing"). It is like a student who sees that a solution worked and thinks, "this is how I should solve this kind of problem".
- There are many ways to tweak the method. The simplest idea: pick the **single best solution** among the correct ones (maybe the shortest or the cleanest-looking), and train on it.
- This is run across **tens of thousands of diverse prompts** (math, physics, etc.) with **thousands of solutions per prompt**, all at the same time, repeated over many iterations.
- Over time the model discovers token sequences that **don't make mental leaps**, **work reliably and statistically**, and **fully use the model's own knowledge**.
- In short: **guess and check**. Guess many solutions, check them, and do more of what worked.

### 9.3 Role of SFT in the RL era
- The SFT model is still useful: it **initializes** the model near the area of correct solutions (for example it already writes out solutions, may set up equations or talk through steps).
- But **RL is where everything gets "dialed in"**: we discover what really works for the model and encourage it, so the model gets better over time.

### 9.4 The full analogy: training LLMs ≈ training children
- Children go through chapters, each with exposition, worked solutions and practice problems.
- For AI the stages are done **one by one**:
  1. **Pre-training** → read all the exposition across all textbooks → build a knowledge base.
  2. **SFT** → study all the worked solutions from human experts → an SFT model that can **imitate experts**, but "blindly", as a statistical mimic.
  3. **RL** → do all the practice problems across all textbooks → the RL model.
- The transcript ends just as the speaker starts to say "the next point I would like to make is that actually…". The rest is not included in this part.

---

## Key Takeaways

- **Base model ≠ assistant.** The base model just simulates internet documents. **SFT on conversations** turns it into an assistant, using the same algorithm with a different data set.
- **Assistants are "programmed by example".** Their behavior is a **statistical imitation of human labelers** following company **labeling instructions**. You are not talking to a magical AI; you are talking to something like a simulation of a skilled labeler.
- **Conversations are just token sequences.** Special tokens mark turns, and the model learns by next-token prediction as before.
- **Modern data sets are mostly synthetic** (LLM-generated), with humans editing and curating.
- **Hallucinations happen** because the model imitates confident answers in the training data. Fixes:
  1. Probe what the model knows, and add **"I don't know"** examples for what it does not know.
  2. Give the model **tools** (web search) to look up facts.
- **Parameters = vague recollection; context window = working memory.** For best results, **paste the relevant text into the prompt** instead of relying on memory.
- **The model has no persistent self.** Identity answers come from training data statistics, SFT identity examples, or a hidden **system message**.
- **Models need tokens to think.** Each token has only a small, fixed amount of compute. Good answers **spread reasoning across many tokens** with intermediate steps. Do not force one-token answers for hard problems.
- **Do not trust mental arithmetic, counting or spelling.** Use tools (code interpreter) for calculation, counting and character manipulation. Tokenization (models see tokens, not letters) explains spelling failures like "strawberry".
- **Intelligence is jagged.** Models can excel at hard tasks and fail at simple ones (like 9.11 vs 9.9). Use them as tools and verify the results.
- **Human solutions are not always optimal for an LLM**, because human and LLM cognition differ. **Reinforcement learning** lets the model practice, try many solutions, and reinforce the ones that reach correct answers. It finds token sequences that work for the model itself.
- **Three-stage analogy:** Pre-training = reading exposition; SFT = studying worked solutions; RL = doing practice problems.
