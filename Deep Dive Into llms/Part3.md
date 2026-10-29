# Deep Dive into LLMs – Part 3: Reinforcement Learning, Thinking Models, RLHF and the Road Ahead

## Overview

This part of the video covers the **third and last training stage of an LLM: reinforcement learning (RL)**. It starts with why humans cannot write the "best" solutions for a model, and then shows how a model can practice by itself using trial and error. It explains DeepSeek-R1 and "thinking" models, compares RL with AlphaGo, and then covers RLHF, which is RL in areas with no clear right answer. The video ends with future trends, places to track and use LLMs, and a grand summary of what happens when you type a question into ChatGPT.

**Timestamps in this part:** DeepSeek-R1 (02:27:47) · AlphaGo (02:42:07) · RLHF (02:48:26) · Preview of things to come (03:09:39) · Keeping track of LLMs (03:15:15) · Where to find LLMs (03:18:34) · Grand summary (03:21:46)

---

## 1. Why Humans Cannot Write the "Best" Solution for an LLM

### What was discussed
The speaker uses a simple word problem in a tokenizer view to show that human labelers do not know which solution is best to train on. This idea is the reason RL is needed.

### Detailed points
- The example problem: *Emily buys 3 apples and 2 oranges. Each orange costs $2. The total cost is $13. What is the cost of each apple?* The answer is **$3**.
- The speaker shows **four possible candidate solutions**. All of them reach the answer 3, but in different ways:
  - Some set up a system of equations.
  - Some talk through it in plain English.
  - Some skip straight to the answer.
- If you are a **human data labeler**, you do not really know which of these to add to the training set.
- The speaker likes the tokenizer view because it is the **native view of the LLM**. The model sees a **one-dimensional sequence of token IDs**, not words in the way humans read them.

**Keyword: Tokenizer / Token Sequence**
**What it is:** A tokenizer cuts text into small pieces called tokens. Each token has a number (ID). The model reads and writes only these numbers in a line.
**Why it was said in this video:** The speaker wants to remind us that the LLM always works with a one-dimensional token sequence, so we must think at the token level when we judge a solution.
**Why it is important:** Everything the model "thinks" must fit into these tokens. This is why the way a solution is spread across tokens matters.

- There are **two separate goals** in any solution:
  1. **Reach the right answer** (the main purpose).
  2. **Present the answer nicely to a human** (a secondary purpose: show steps, make it readable).
- The speaker says: for now, **focus only on reaching the final answer**, and ignore presentation.
- Which solution is best for the model? We do not know. Reasons:
  - Each token gets only a **finite, fairly small amount of compute**. So the model cannot make a very big "leap" in a single token.
  - In one candidate, the step "30 − 4 ÷ 3 =" asks for a lot of mental arithmetic **in one token**. This is a bad example to train on. It pushes the model to skip steps quickly, and it will make mistakes in mental arithmetic.
  - Spreading the solution out (an equation, or talking through it) may work better, but we cannot be sure.
  - Another problem: too many tokens may be **wasted** on steps that are trivial for the model.
- **Human thinking is different from LLM thinking.** What is easy for a human may be a huge leap for the model. What is hard for a human may be trivial for the model.
- This applies to **knowledge** too:
  - The LLM has a lot of knowledge (the speaker mentions PhD-level math, physics, chemistry). In some ways it knows more than the labeler. The labeler may not use that knowledge.
  - The labeler may also **inject knowledge the model does not have** in its parameters. This causes sudden leaps that confuse the model.
- **Conclusion:** We are not in a good position to write the best token sequences for the LLM. Human-written examples are useful to **initialize** the model by imitation, but the model should **discover for itself** which token sequences reliably lead to the right answer. This discovery happens through **reinforcement learning and trial and error**.

---

## 2. How Reinforcement Learning Works (Guess and Check)

### What was discussed
The speaker shows RL in action using the Hugging Face inference playground and a small model, then explains the idea with a cartoon diagram.

### Detailed points
- The model used in the demo is **Gemma 2 (2 billion parameters)**. This is a very small model, chosen just to show the idea.
- The basic process:
  1. Take the prompt.
  2. Let the model generate a solution.
  3. Check the solution against the known correct answer ($3).
  4. Repeat many times.
- Every attempt is **different** because models are **stochastic**: at every token there is a probability distribution, and the model samples from it. So the model goes down slightly different paths each time. In the demo, three attempts gave three slightly different solutions, all correct.
- In practice, you might sample **thousands or even millions of independent solutions for just one prompt**. Some will be correct and some will not.

**Keyword: Stochastic Sampling**
**What it is:** The model does not always pick the single most likely token. It randomly picks from a probability distribution, so the output changes each time.
**Why it was said in this video:** To explain why every attempt at the same prompt gives a different solution. This variety is what RL uses.
**Why it is important:** Without this randomness, the model would try the same solution every time and there would be nothing to compare or learn from.

- **The cartoon diagram:**
  - One prompt → many solutions in parallel.
  - **Green** = reached the right answer. **Red** = did not.
  - Example: **15 solutions generated, only 4 are correct**. (The speaker says this example problem is too easy for this to happen, but asks us to imagine it.)
- **Core idea:** Encourage the kind of solutions that lead to right answers, by **training on those sequences**.
  - The training data does **not** come from human experts. It comes from **the model itself**.
  - This is like a student who tries many solutions, sees which ones worked, and says "this is how I should solve these problems".
- There are many ways to tweak the method. For the simple idea, you can think of it as picking the **single best solution** out of the correct ones (for example the shortest, or the one that looks nicest) and training on it. After the parameter update, the model is **slightly more likely** to take that path in the future.
- In real life, this is done at huge scale: **tens of thousands of prompts** (math, physics, and others), with **thousands of solutions per prompt**, all at the same time.
- Over many iterations, the model discovers which token sequences work for it. These sequences:
  - make no unreasonable mental leaps,
  - work reliably and statistically,
  - fully use the knowledge the model actually has.
- In short, RL is **guess and check**: guess many solutions, check them, and do more of what worked.

**Keyword: Reinforcement Learning (RL)**
**What it is:** A way of training where the model tries things, gets a score for the result, and then becomes more likely to repeat the things that scored well.
**Why it was said in this video:** It is the third main stage of training an LLM, and the speaker is explaining it step by step with a concrete example.
**Why it is important:** RL lets the model find strategies that humans did not write down. It is the stage where "everything gets dialed in".

**Keyword: Rollout**
**What it is:** One full attempt (one generated solution or one played game) that gets scored.
**Why it was said in this video:** The speaker uses "solutions" and "rollouts" for the many attempts per prompt that are checked in parallel.
**Why it is important:** The more rollouts you try, the better chance the model finds good paths, and the more data RL has to learn from.

### Role of SFT after RL is introduced
- The **SFT (supervised fine-tuning) model** is still helpful. It **initializes** the model into the *vicinity* of correct solutions. For example, it teaches the model to write out solutions, set up equations, or talk through a problem.
- But **RL is where everything gets dialed in**: the model discovers what works for itself and gets better over time.

---

## 3. The Three Stages = How We Teach Children

### What was discussed
A summary of the full training pipeline using the analogy of a textbook.

### Detailed points
- Children go through textbook chapters and do many types of exercises inside each chapter. LLM training is similar, but **stage by stage**, with each stage covering the whole "textbook" set:
  1. **Pre-training** = reading all the **expository material** (the explanations) in all textbooks at once. This builds a **knowledge base**.
  2. **SFT** = studying all the **worked solutions from human experts**. This gives a model that **imitates the experts**, but does so **blindly**: it just makes its best statistical guess at mimicking expert behavior.
  3. **RL** = doing all the **practice problems**. This produces the **RL model**.
- Pre-training and SFT have been around for years and are **standard**; every LLM provider does them.
- **RL is still early and not standard** in the field. The core idea is very simple (trial-and-error learning), but there are many **details and mathematical nuances**:
  - how to pick the best solutions,
  - how much to train on them,
  - what the prompt distribution should be,
  - how to set up the training run so it actually works.
- Companies like OpenAI had experimented with RL fine-tuning internally for a while, but **did not talk about it publicly**.

---

## 4. DeepSeek-R1: RL Made Public

### What was discussed
The DeepSeek-R1 paper and why it was such a big deal.

### Detailed points
- **DeepSeek** is a company from China. Their paper talked **very publicly** about RL fine-tuning for LLMs, how important it is, and how it brings out **reasoning capabilities**.
- The paper gave the **many details needed to reproduce** the results, so it **renewed public interest** in using RL for LLMs.

### Figure: accuracy goes up
- The paper shows the model's **accuracy on math problems** rising steadily as the number of training steps (thousands) goes up.
- The problems are fairly simple math problems. The model starts off not very good, but by doing trial and error on a big dataset of problems, it keeps improving.

### Figure: response length goes up
- The **average length per response also goes up** during training. The model uses **more tokens** to get higher accuracy.
- The long solutions contain things like: *"Wait, wait, wait. That's an aha moment I can flag here. Let's reevaluate this step by step…"*

**Keyword: Emergent Property / "Aha Moment"**
**What it is:** A behavior that nobody programmed. It appears by itself as the model is optimized. The "aha moment" is the model stopping and re-checking its own work.
**Why it was said in this video:** The speaker points to this as proof that RL finds new ways of problem solving without any human telling it how.
**Why it is important:** It shows that RL can create useful strategies (like backtracking) that would be very hard for a human to write into the "ideal answer".

- What the model learns to do:
  - re-evaluate steps,
  - try ideas from **different perspectives**,
  - **retrace, reframe, backtrack**,
  - use analogies,
  - check results in different ways.
- These are things humans do **in their heads** when solving a problem. They are not what we write on the page as the final solution. This is why no human could "hard-code" it into the ideal assistant response.
- The only thing given to the model is **the correct answers**. The thinking process comes out of trying to solve problems correctly.
- The speaker calls these **cognitive strategies**: how you manipulate a problem and look at it from different angles.
- The longer responses are the **chain of thought**. This is what makes the responses longer, but it is also what increases accuracy.

**Keyword: Chain of Thought**
**What it is:** The step-by-step thinking the model writes out before giving the final answer.
**Why it was said in this video:** It is what makes the RL model's output long, and it is the visible form of the "thinking" that RL discovered.
**Why it is important:** Because the model has only limited compute per token, spreading out its thinking over many tokens helps it reach correct answers more reliably.

### Demo: the same problem on a reasoning model
- The speaker pastes the Emily/apples problem into **chat.deepseek.com** with the **"DeepThink" button on** (this turns on R1).
- Compared with the earlier SFT-style answer (which mimics an expert solution), the RL model's output reads like thinking:
  - "Let me try to figure this out…"
  - It reaches $3, then says *"wait a second, let me check my math again"*.
  - It tries a **different approach** (setting up an equation) and confirms the same answer.
  - It says it is confident.
- **After thinking, it writes a nice solution for the human** and boxes the answer. So two aspects are separate:
  - the **thinking** (about correctness, from RL),
  - the **presentation** (written for the human).

---

## 5. Thinking Models in Practice

### What was discussed
Where to use reasoning models, privacy concerns, and how the major providers differ.

### Detailed points
- **Privacy concern:** Some people are nervous about putting sensitive data into chat.deepseek.com because it is run by a Chinese company.
- DeepSeek-R1 is **open weights**: anyone can download and use it.
  - You cannot run the full model in full precision on a MacBook; it is too large.
  - But **many companies host the full model**. The speaker likes **together.ai**. In its playground you can choose DeepSeek-R1 among many state-of-the-art models.
  - Since DeepSeek released the model, the output should be basically equivalent in power (the exact text differs due to randomness). The speaker notes this one is hosted by an **American company**.

**Keyword: Open Weights**
**What it is:** The trained model parameters are released so anyone can download them, run them, or host them.
**Why it was said in this video:** To explain that you can use DeepSeek-R1 from other hosts, not only from DeepSeek's own website.
**Why it is important:** It gives people choice and control over where their data goes, and over how the model is used.

- **OpenAI models:**
  - Models such as **o1, o3-mini, o3-mini-high** use "advanced reasoning". This refers to being trained by RL with techniques **very similar to DeepSeek-R1** (per public statements of OpenAI employees).
  - **GPT-4 / GPT-4o / GPT-4o mini** (free tier) should be thought of as **mostly SFT models**. They do not do the "thinking" seen in RL models, even though they have a little reinforcement learning involved (covered in RLHF below).
  - The thinking models may need a paid subscription ($20 or $200 per month for the top models).
  - OpenAI **does not show the exact chains of thought** in the web UI. It shows **summaries**. The speaker thinks this is partly because of **distillation risk**: someone could imitate the reasoning traces and recover much of the reasoning performance.
- **Performance:** DeepSeek and OpenAI thinking models are roughly **on par**, though it is hard to tell because of evaluations. The $200/month OpenAI models may still look better. R1 is still a very solid choice, and it is available on many sites because it is open weights.
- **Google:** Through **AI Studio** (aistudio.google.com), the speaker shows **Gemini 2.0 Flash Thinking Experimental**, an early experimental thinking model that also solves the problem. The speaker says the AI Studio UI looks busy and ugly.
- **Anthropic:** At the time of the video (early 2025), Anthropic **did not offer a thinking model**.
- All these thinking models are **experimental** as of very early 2025. RL is an exciting new stage, but getting the details right is difficult.

**Keyword: Distillation (risk)**
**What it is:** Training a model by imitating the outputs of another model.
**Why it was said in this video:** The speaker gives this as the likely reason OpenAI hides raw reasoning and only shows summaries.
**Why it is important:** The reasoning traces are valuable. If they are copied, others can get much of the performance without doing the expensive RL.

### When should you use a thinking model?
- Use a thinking model when the prompt needs **advanced reasoning** (hard math, code).
- For **simple knowledge questions** it may be overkill. There is no need to wait 30 seconds of thinking for a factual question.
- The speaker's own usage: about **80–90% of use is GPT-4o**. For very difficult math or code problems, the speaker reaches for a thinking model and waits longer.

---

## 6. AlphaGo: RL Beyond Human Imitation

### What was discussed
A connection to Google DeepMind's AlphaGo. It shows that RL being very powerful is **not new** to AI.

### Detailed points
- AlphaGo learned to play **Go** against top human players.
- The paper has a plot of **Elo rating** (strength) comparing:
  - a model trained by **supervised learning** (imitating human expert games),
  - a model trained by **reinforcement learning**.
- Results:
  - **Supervised learning** improves, but then **tops out**. You cannot go fundamentally beyond human players if you only imitate them. It never gets better than the very top players like **Lee Sedol**.
  - **Reinforcement learning is significantly more powerful**. The system **plays against itself**, tries many moves, and **reinforces the games that lead to a win**. It learns the sequences of actions that empirically and statistically win.
  - RL is **not constrained by human performance**, and it can beat even the top players.
- The diagram is the **same as the LLM one**, except there is **no prompt** (the game is fixed) and the reward is **a win** instead of a specific answer.
- The researchers could probably have run it longer, but chose to stop because it **costs money**.
- The speaker says we are only starting to see hints of this diagram in LLMs for reasoning problems. We need to go beyond just imitating experts, set up **game-like environments**, and let the system discover reasoning traces that are unique and work.

### Move 37
- RL does not stop the model from **leaving the human distribution**.
- **Move 37** in AlphaGo was a move **no human expert would play**. Its probability of being played by a human was estimated at about **1 in 10,000**. In hindsight it was **brilliant**: AlphaGo found a strategy unknown to humans.
- The speaker recommends the YouTube video *"Lee Sedol vs AlphaGo Move 37 reactions and analysis"*. In the clip, people react with surprise; one thought it was a mistake.

**Keyword: Move 37**
**What it is:** A famous move by AlphaGo that human experts found very strange but that turned out to be a great move.
**Why it was said in this video:** It is the example of RL discovering something beyond human ideas.
**Why it is important:** It hints that LLMs trained with RL might one day find reasoning strategies that humans have never thought of.

### What this could mean for LLMs
- Speculation from the speaker (not facts): how can a model reason better than a human? Maybe:
  - it finds **analogies** that humans could not create,
  - it finds a **new thinking strategy**,
  - it even invents **its own language** to think in, because the model is **not forced to stick to English**.
- In principle the system's behavior is **less defined**. It is free to do whatever works, and may **drift away from the distribution of its training data (English)**.
- This needs a **large and diverse set of problems** ("game environments" or practice problems) where thinking can be refined. Creating such **large, diverse prompt distributions** is a lot of the frontier LLM research right now. We need practice problems for all domains of knowledge.

---

## 7. Reinforcement Learning in Unverifiable Domains (RLHF)

### What was discussed
So far all RL examples have had a clear right answer. This section covers what to do when there is none.

### Verifiable vs unverifiable domains
- **Verifiable domain:** Any candidate solution can be scored easily against a concrete answer. Examples: the answer is 3.
  - Option 1: require the model to **box its answer** and check for equality with the correct one.
  - Option 2: use an **LLM judge** that compares the solution with the known answer. LLMs are good enough now to do this fairly reliably.
  - This all happens **automatically, with no humans in the loop**.
- **Unverifiable domain:** Examples are **creative writing, jokes, poems, summaries**. It is hard to score the different solutions.
- Example: ask ChatGPT for a **joke about pelicans**. It gives jokes such as pelicans stuffing things in their beaks "because they don't believe in backpacks", or "why don't pelicans ever pay for their drinks? Because they always bill it to someone else". The speaker finds the jokes not very good and says humor is "secretly very difficult".
- Problem: **how do we score the jokes?**

**Keyword: Verifiable Domain**
**What it is:** A task area where an automatic check can say if the answer is right (for example, math with a known answer, or code with tests).
**Why it was said in this video:** To separate the places where RL is easy (verifiable) from the places where it is hard (creative tasks).
**Why it is important:** Where answers can be checked, RL can run for a very long time and find great strategies. Where they cannot, we need a different method.

### The naive approach is too expensive
- A human could score everything, but RL needs a huge number of scores.
- The speaker's **cartoon numbers**: 1,000 updates × 1,000 prompts per update × 1,000 rollouts per prompt = **1 billion** human evaluations of jokes. This is unscalable.

### The RLHF trick: indirection with a reward model
- The idea (from an **OpenAI paper**; the speaker notes many of its authors later became Anthropic co-founders) is to **involve humans only a little**.
- Train a **separate neural network, the reward model**, to **imitate human scores**. It becomes a **simulator of human preferences**. Then do RL against the simulator instead of real humans.
- The simulator can be queried as many times as needed, and it is fully automatic. It is not a perfect human, but if it is statistically similar to human judgment, it works in practice.

**Keyword: RLHF (Reinforcement Learning from Human Feedback)**
**What it is:** RL where the reward does not come from a fixed correct answer. It comes from a model trained on human preferences.
**Why it was said in this video:** It is the method that lets us do RL in unverifiable domains such as jokes, poems and summaries.
**Why it is important:** It is how models like GPT-4o were improved in creative and open tasks, because human time is too limited to score everything.

**Keyword: Reward Model**
**What it is:** A separate neural network (probably a Transformer) that takes a prompt and a candidate response and outputs one number (for example between 0 and 1) as a score. It is **not** a language model that generates text.
**Why it was said in this video:** It is the "simulator of humans" that replaces real human scoring during RL.
**Why it is important:** It makes RL possible at scale, but it is also the weak point (see the downsides below).

### How the reward model is trained (the cartoon example)
1. Take a prompt, e.g. "write a joke about pelicans", and generate **five rollouts** (five jokes).
2. Ask a **human to order them from best to worst** (1 to 5). Humans order instead of giving scores because **ordering is an easier task**.
3. Ask the **reward model** to score each joke (inputs: the prompt and the candidate joke; output: one number, e.g. 0.1 = very low, 0.8 = high).
4. **Compare** the model's scores with the human ordering. There is a precise mathematical way (a **loss function**), but the intuition is:
   - If the human ranked a joke #1 and the model gave it 0.8, they roughly agree. The score should go a little higher (e.g. 0.81).
   - If the human ranked a joke #2 but the model gave it only 0.1, there is a big disagreement. The score should go up a lot (e.g. 0.15 or more).
   - If the human ranked a joke worst but the model gave it a fairly high score, the score should come down.
5. Update the network slightly, again and again. It becomes a better and better simulator of human orderings.
- Humans only need to look at something like **1,000 prompts × 5 rollouts = about 5,000 jokes**, not 1 billion.

### Upsides of RLHF
- It lets us run **RL in arbitrary domains**, including **unverifiable** ones: summarization, poem writing, joke writing, creative writing.
- Empirically, when done correctly, the models become **a little bit better**.
- The speaker's **best guess** (not well established) for why: the **discriminator–generator gap**.
  - It is much **easier to discriminate (judge) than to generate** for humans.
  - In SFT, humans must **write** the ideal response. For poems or jokes this needs creative writing and is very hard.
  - In RLHF, humans just **order** five outputs. This is easier and gives **higher-accuracy data**.
  - The system then discovers the kinds of responses humans would grade well.

**Keyword: Discriminator–Generator Gap**
**What it is:** The idea that judging which of several options is better is easier than creating a good option yourself.
**Why it was said in this video:** It is the speaker's best guess for why RLHF improves models.
**Why it is important:** It explains why human feedback by ranking is a cheaper and more accurate source of supervision than asking humans to write ideal responses.

### Downsides of RLHF
1. **It is a lossy simulation.** We do RL against a **simulation of humans**, not actual humans. The reward model is just a model outputting scores and may not match a real human's opinion in every case.
2. **RL is very good at gaming the reward model.** This is the subtle and bigger problem.
   - The reward model is a huge neural network (billions of parameters) that outputs one score. There are inputs that are **not in its training set** that get **very high scores in a fake way**.
   - What you see if you run RLHF for long: for example after 1,000 updates, you expect real bangers about pelicans. Instead, in the **first few hundred steps** the jokes improve a bit, and then they **fall off a cliff** into nonsense. The top "joke" becomes something like **"the the the the"**, and the reward model gives it a score of 1.0.
   - These are **adversarial examples**: small odd inputs that slip into the "nooks and crannies" of the model.
   - Fixing it by adding "the the the" to the dataset with a very low score works for that example, but there are **infinitely many** such adversarial examples. You can do many rounds and never win.

**Keyword: Adversarial Examples**
**What it is:** Strange inputs that trick a neural network into giving a wrong (often very confident) output.
**Why it was said in this video:** RL found nonsense text that the reward model scored as the best joke.
**Why it is important:** It is the main reason RLHF cannot be run for a long time.

- **Practical result:** You run RLHF for a **few hundred updates**, the model gets better, and then you **crop it and ship it**. You cannot run too long against the reward model.
- **The speaker's view: "RLHF is not RL"** (in the magical sense).
  - Technically it is RL, but not the kind you can run indefinitely.
  - In **verifiable domains** (math, code with a boxed answer) the scoring function is hard to game: you either got it right or not. So RL can run for **tens or hundreds of thousands of steps** and find very surprising strategies.
  - In **Go**, there is a perfect simulator of win or loss. It cannot be gamed, so RL can continue until it beats Lee Sedol.
  - With a **gameable reward model**, you cannot repeat the process forever. So the speaker sees RLHF as **a little fine-tune** that gives a small improvement, not something where "more compute and longer runs give magical results".
- GPT-4o has gone through RLHF because it works well, but it is not RL in the same sense as the thinking models.

---

## 8. Recap of the Three Stages and the Limits of LLMs

### Detailed points
- The three major stages (pre-training, SFT, RL) loosely match how we teach children:
  - **Pre-training** = basic knowledge from reading exposition.
  - **SFT** = looking at many worked examples and imitating experts.
  - **RL** = practice problems.
- The difference: we now have to **write textbooks for LLMs** across all disciplines (including code and math), improve the algorithms, and then execute training well at scale.
- The training runs are **extremely large, complicated, distributed jobs** over **tens of thousands or even hundreds of thousands of GPUs**. The engineering is at the state of the art of what is possible with computers at that scale. The speaker did not cover it much.
- **Hallucinations:** Even with mitigations, models still hallucinate. It has improved and will keep improving, but they are not perfect.

**Keyword: Swiss Cheese Model of LLM Capabilities**
**What it is:** A way to think about LLMs: they are very good in many areas, but have random "holes" where they fail.
**Why it was said in this video:** The speaker uses it again to warn that models can fail in unexpected places.
**Why it is important:** It reminds us to check the model's work. Example from the video: it may not know which is bigger, **9.11 or 9.9**, yet it can solve Olympiad questions.

- **Advice:** Do not treat models as infallible. Use them as **tools**, for **inspiration** and **first drafts**, check their work, and **be responsible for the final product** of your work.

---

## 9. Preview of Things to Come

### What was discussed
Expected future directions of LLMs.

### Detailed points
- **Multimodality (soon):** Models will handle **text, audio and images** natively. They can hear, speak, see and paint, inside one model, and this enables more natural conversation.
  - This is not a fundamental change. You can **tokenize audio and images** and use the same methods as for text.
  - **Audio:** take slices of the **spectrogram** and tokenize them.
  - **Images:** use **patches** and tokenize each patch. An image becomes just a sequence of tokens.
  - Then you have streams of tokens for audio, images and text, interleaved and handled together in a single model.
- **Agents (long-running tasks):**
  - Today we hand models **individual tasks "on a silver platter"**, and we still organize the coherent execution of many tasks into a job.
  - Models cannot yet string tasks together in a **coherent, error-correcting way over long periods**, but they are getting there.
  - Expect **agents** that work for many tens of seconds, minutes or hours, and come back from time to time to report progress.
  - They are not infallible, so **supervision is needed**. Like the **human-to-robot ratio** in factories, we may talk about **human-to-agent ratios**, where humans become supervisors of digital agent tasks.
- **Pervasive and invisible:** AI will be integrated into tools everywhere.
- **Computer use:** Models will take **actions on your behalf** (keyboard and mouse). The early example is **ChatGPT Operator**.
- **Research still needed – test-time training:**
  - Today there are two stages: **training** (tune the parameters) and **inference/deployment** (parameters are fixed and do not change).
  - The only learning the model does at test time is **in-context learning**, through the changing tokens in the context window.
  - Humans learn as they work (the speaker mentions how the brain may update "parameters" during sleep). Models have no equivalent yet.
  - This matters because the **context window is a finite and precious resource**. With very long, multimodal tasks (videos, etc.) the context will need to be far beyond hundreds of thousands of tokens.
  - Simply making the context longer will not scale. The speaker thinks **new ideas are needed**.

**Keyword: Context Window**
**What it is:** The working memory of the model: the set of tokens it can see at one time.
**Why it was said in this video:** To explain why long-running multimodal tasks are a problem: the window is limited and precious.
**Why it is important:** The model can only "learn" at test time through what is in this window, so it limits how long and how complex a task can be.

**Keyword: In-Context Learning**
**What it is:** The model adapts its behavior using the examples and information inside the current context, without changing its parameters.
**Why it was said in this video:** It is the only kind of learning the model has at test time.
**Why it is important:** It shows the gap between models and humans, who keep learning as they work.

---

## 10. Keeping Track of LLMs

The speaker names three resources for staying up to date:

1. **LM Arena (leaderboard)**
   - Ranks top models using **human comparisons**: people prompt two models without knowing which is which and judge the better answer. A ranking is computed from this.
   - In the video: **Google (Gemini) is on top**, **OpenAI right behind**, and **DeepSeek is #3**.
   - DeepSeek is **MIT licensed and open weights**, which is described as **unprecedented** for a model this strong.
   - Further down: xAI, Anthropic (Sonnet at **#14**), and Meta with Llama (also open weights, but lower).
   - **Caution:** The speaker thinks the leaderboard was really good for a long time, but in recent months it has become **a little bit gamed**, and the speaker does not trust it as much. Example: many people use Claude Sonnet and like it, but it ranks low, and Gemini ranks very high even though fewer people use it. Use it as a **first pass**, and try a few models for your own tasks.
2. **AI News newsletter** (by swyx and friends)
   - Very **comprehensive**, produced almost every other day. Partly human-curated and partly generated automatically by LLMs.
   - The summaries at the top are quite good and seem to have human oversight. You probably will not miss anything major, but it is very long.
3. **X (Twitter)**
   - A lot of AI news happens on X. Follow people you like and trust.

---

## 11. Where to Find and Use LLMs

- **Big proprietary models:** Go to the provider's website. For OpenAI, chatgpt.com. For Gemini, gemini.google.com or **AI Studio** (the speaker says there are two sites and nobody understands why).
- **Open-weights models (DeepSeek, Llama etc.):** Use an **inference provider**. The speaker's favorite is **together.ai** (playground with many models).
- **Base models:** They are not common even on inference providers, since providers target assistants and chat. The speaker goes to **Hyperbolic**, which serves **Llama 3.1 base**, and wishes more people hosted base models because they are useful and interesting.
- **Run locally:**
  - You cannot run the biggest DeepSeek model on a MacBook, but there are smaller **distilled** versions, and you can run them at **lower precision**.
  - Native precision examples: **FP8** for DeepSeek, **BF16** for Llama.
  - The speaker's favorite app is **LM Studio**. The speaker finds the UI ugly and confusing (too many models, distillations and precisions) and aimed at professionals, but once you understand it, it works.
  - Demo: a **Llama 3.2 Instruct 1B** model runs locally on the MacBook Pro GPU, answers pelican-joke requests, and nothing leaves your computer. You can eject the model afterward to free RAM.

**Keyword: Precision (FP8, BF16)**
**What it is:** How many bits are used to store each number (parameter) of the model. Lower precision uses less memory.
**Why it was said in this video:** To explain how big models can be shrunk to fit on a laptop.
**Why it is important:** It makes local use possible, with some loss in quality. The speaker says not to worry if you do not fully understand the details.

---

## 12. Grand Summary: What Happens When You Use ChatGPT

The speaker goes back to the original question: *when we go to chatgpt.com, enter a query and hit go, what are we talking to?*

### Step by step
- Your query is **chopped into tokens** and put into the **conversation protocol format** (the structure used to keep conversation objects). The whole thing becomes a **one-dimensional token sequence**.
- ChatGPT **continues appending tokens** to this sequence, like a **token autocomplete**.
- **Stage 1 – Pre-training:** knowledge acquisition from the internet into the parameters of the neural network.
- **Stage 2 – SFT:** this is where the **personality** comes in.
  - A company like OpenAI curates a dataset of perhaps **1 million conversations** on diverse topics, between a human and an assistant.
  - There is a lot of synthetic data and LLM help, but fundamentally it is a **human data curation task**. Human labelers are given labeling instructions and write ideal assistant responses.
  - **How to think about the answer:** it is a **neural network simulation of an OpenAI data labeler**. As if the labeler read the labeling instructions and spent **2 hours** writing the ideal response. We did not wait two hours, so we get a simulation.
  - It is a **lossy simulation**: a fixed mathematical expression mixes the token inputs with parameters, with a **finite amount of compute per token**. Neural networks do not work like human brains, so what is easy or hard for them is different.
- **Consequences:** hallucinations, and the Swiss cheese pattern. Reasons given:
  - not enough tokens to think,
  - mental arithmetic breaks,
  - they cannot count letters,
  - they may think **9.11 is bigger than 9.9** (the transcript says "9.11 is smaller than 9.9" as an example of the error and describes it as looking "kind of dumb").
- **Stage 3 – RL (thinking models):**
  - GPT-4o does **RLHF, not RL**. RLHF is a little fine-tuning, with "no time for magic".
  - Thinking models (like o3-mini) **do use RL**. They practice on a large collection of practice problems that companies create and curate. They discover thinking strategies that look like your **internal monologue**.
  - Their output is **not just an imitation of a human data labeler**. It is something new: thinking that **emerged** from RL.
  - OpenAI does not show the under-the-hood thinking, only a summary.

### Open questions and status
- **Does thinking learned in verifiable domains transfer to unverifiable domains** (like creative writing)? The extent is **unknown in the field**. The question given to the model in the demo was a creative writing question, so it did not really get to shine.
- RL for LLMs is still **new, primordial and nascent**. We see the **beginnings of hints of greatness**. In principle it is capable of something like **Move 37 in open-domain thinking**, new analogies no human had. For now, these models mostly shine in **verifiable domains** such as math and code.

### Closing advice
- It is an extremely exciting time. The speaker uses these models **tens or hundreds of times daily**, they dramatically speed up work, and the speaker expects a lot of wealth creation.
- Be aware of shortcomings: even RL models will **randomly do dumb things, hallucinate, skip mental arithmetic, and fail at counting**.
- **Use them as tools in a toolbox. Check their work and own the product of your work.** Use them for inspiration and first drafts, ask questions, but **always check and verify**.

---

## Key Takeaways

- **The labeler problem:** Humans cannot know which token sequence is best for an LLM, because human cognition is different from the model's. So the model should discover good sequences itself.
- **RL = guess and check:** Generate many solutions, check them against the right answer, and train more on the ones that worked. The training data comes from the model itself.
- **Three stages:** Pre-training (read exposition) → SFT (imitate worked solutions) → RL (practice problems). Pre-training and SFT are standard; RL is early and hard to get right.
- **SFT is still useful** as an initialization, but **RL dials everything in**.
- **DeepSeek-R1** made RL public. Accuracy and response length both rose during RL, and the model learned to re-check, backtrack, reframe and try other approaches ("aha moments"), without being told to.
- **Thinking models** (DeepSeek-R1, OpenAI o1/o3, Gemini thinking) are trained with RL. GPT-4o and similar models are mostly SFT (plus a little RLHF). Use thinking models for hard math and code, and simpler models for simple questions.
- **AlphaGo** shows that imitation hits a ceiling at human level, while RL can go beyond it (Move 37). LLMs may one day show something similar in reasoning, maybe even with new thinking languages.
- **Verifiable domains** allow long RL runs because the answer is hard to game. **Unverifiable domains** need RLHF.
- **RLHF** uses a reward model (a simulator of human preferences trained on human *rankings*). Humans find ranking easier than writing, which gives better data (the discriminator–generator gap).
- **RLHF limit:** RL games the reward model through adversarial examples, so you can only run a few hundred updates, then stop. The speaker sees RLHF as "not real RL", more like a small fine-tune.
- **Future:** multimodal models (tokenized audio and images), long-running agents with humans as supervisors, computer use, and new research like test-time training for long contexts.
- **Where to track and use LLMs:** LM Arena (use with caution), AI News newsletter, X; use provider websites, together.ai, Hyperbolic (base models), and LM Studio (local models).
- **How to use LLMs well:** They are powerful but have Swiss cheese holes and can hallucinate. Treat them as tools, verify their output, and stay responsible for your work.
