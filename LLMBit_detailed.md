**Document Type:** Expanded Workshop Transcript
**Event:** GAMBIT Collaboration Meeting (Oslo)
**Date:** December 2025
**Session:** "LLMBit" – AI/ML Tools Workshop
**Presenter:** Will Handley (Cambridge)
**Key Discussant:** Anders Kvellestad (Oslo)

---

### 1. Introduction and Context

The session began with Will Handley addressing the collaboration members regarding the rapid evolution of AI tools over the course of 2025. He opened by polling the room: "Who has used ChatGPT in the past hour?" followed by, "Who here uses *agentic* coding systems like Claude Code?"

While most attendees were familiar with standard chatbots, few had adopted agentic workflows. Handley positioned the talk not just as a technical demo, but as an argument that the scientific workflow has fundamentally shifted. He noted that there are still "ultra-conservative" members of academia who ignore these tools, but argued that the capabilities demonstrated in 2025 make adoption mandatory for staying competitive.

### 2. The AI Landscape in 2025

Handley outlined the timeline of the "AI explosion" witnessed over the last year:

*   **February 2025 (The Reasoning Shift):** Models like OpenAI’s **o3** achieved "PhD-student capabilities." Handley shared an anecdote from Cambridge:
    > "I got o3 to take the third-year Cambridge Astrophysics exams 'blind'—meaning no internet access, just the model. It achieved the highest mark we’ve seen in 25 years. This shows these things aren't just 'next token predictors' anymore; they are capable of reasoning."
*   **May 2025 (The Agentic Shift):** The release of tools like **Claude Code** and **Cursor** moved AI from a chat interface to the command line. This allowed AI to not just answer questions, but to *do* work.

**Key Insight:** Handley emphasized that if researchers are relying solely on techniques from 2024 (chat interfaces), they are already significantly behind.

### 3. The "Three Layers" Framework

Handley introduced a taxonomy for current AI coding tools:

1.  **Layer 1: Autocomplete (e.g., GitHub Copilot).**
    *   *Status:* Essential. "If you don't have this in VS Code or Vim, do it today."
    *   *Function:* Tight feedback loop, predicts the next few lines of code based on open buffers.
2.  **Layer 2: Chat-based AI (e.g., ChatGPT, Claude Web Interface).**
    *   *Status:* The "Trap."
    *   *Critique:* Handley argued that this turns the researcher into a "debug monkey." You ask for code, copy-paste it, run it, it fails, you copy-paste the error back. The human becomes the slow interface between the AI and the terminal.
3.  **Layer 3: Agentic Systems (e.g., Claude Code, Gemini CLI).**
    *   *Status:* The focus of this workshop.
    *   *Function:* You give the AI a terminal. It edits files directly, runs compilers, reads error logs, and fixes its own mistakes without human intervention.

### 4. Live Demonstration: Compiling GAMBIT with Claude Code

To demonstrate Layer 3, Handley ran **Claude Code** live on the GAMBIT repository.

**The Task:** He asked the agent to "Initialize and compile GAMBIT" on an Arch Linux laptop—a task notoriously difficult due to dependency management.

**The Process:**
1.  **Initialization:** The agent scanned the directory, reading the `README`, `CMakeLists.txt`, and file structure to build a "mental map" of the project (costing tokens).
2.  **Attempt 1:** The agent attempted to run `cmake`. It failed immediately due to an incompatibility with the `Eigen` library (Eigen 3 vs. Eigen 5) on the Arch Linux system.
3.  **Autonomous Debugging:** Instead of asking Handley for help, the agent:
    *   Read the standard error output.
    *   Identified that `cmake/FindEigen3.cmake` was finding the wrong version.
    *   "Rummaged around" the system directories to find the correct include paths.
    *   Proposed a patch to the CMake module.
4.  **Execution:** The agent applied the edit (removing the old path, inserting the new one), re-ran CMake, and successfully generated the makefiles.
5.  **Git Integration:** Finally, Handley commanded: *"Commit this and push."* The agent wrote a detailed commit message ("Fix CMake message verbosity and Eigen paths"), created a new branch, and pushed it to GitHub.

**Anders Kvellestad** commented on the speed of the workflow compared to manual debugging:
> "You should have seen the questions yesterday... getting to a successful CMake is a big deal. It usually takes hours."

### 5. Beyond Coding: Other Applications

Handley emphasized that while coding is impressive, these tools excel at "boring" academic infrastructure tasks.

*   **Wiki Maintenance:** He demonstrated asking the agent to scan the local GAMBIT wiki, identify broken links, fix formatting inconsistencies between working groups, and look for "insults" in meeting minutes.
*   **Context Engineering:** Handley introduced the concept of **Context Files** (e.g., `claude.md`). Instead of "prompt engineering" (crafting a perfect question), researchers should focus on "context engineering"—placing a file in the repository root that tells the AI: *"This is a scientific repo. Do not catch exceptions silently. Use this specific citation format."*
*   **Grant Writing:** He described a workflow where he transcribes every meeting and whiteboard session, then asks the AI to synthesize months of transcripts into a LaTeX grant proposal.

### 6. MCP and Custom Tools

Handley demonstrated the **Model Context Protocol (MCP)**, a standard that allows the AI to use custom tools.

**The Demo:** He showed a custom Python script he wrote that connects Claude to the **ArXiv API**.
*   *Command:* "Download recent papers by Will Handley."
*   *Action:* The AI used the tool to search ArXiv, parsed the JSON response, identified the PDFs, and downloaded them to a local directory.
*   *Implication:* Researchers can build custom tools that give the AI access to proprietary databases or specific simulation codes, closing the feedback loop even further.

### 7. Economics and Practical Considerations

A significant portion of the discussion revolved to the cost and ethics of these models.

*   **The "Rinse the VCs" Strategy:** Handley noted that while he pays **$20/month** for the subscription, his heavy usage consumes roughly **$3,000 worth of tokens** per month if priced at API rates.
    > "We are currently in a subsidized bubble. You should rinse the Venture Capitalists for every penny they are worth while this pricing lasts."
*   **Energy and "AGI":** Handley pushed back on the Silicon Valley hype of Artificial General Intelligence (AGI). He noted that while these models are useful, biological brains have been trained with "geological timescales of energy". He views them as a utility (like electricity) rather than a sentient peer.
*   **Safety Rule #1:** "Everything must be in a Git repository." Handley warned that agentic tools can—and will—delete files by accident. "I once asked it to `git reset`, and it deleted two hours of work. It literally replied: *'Oh shit, you're right. Sorry.'*"

### 8. Discussion: Implications for GAMBIT and Scientific Computing

**Anders Kvellestad** raised a critical point regarding the "hollowing out" of expertise:
> "I’ve used this for prototyping, and it’s great. But my fear is that if we rely on this, the core team might lose the deep understanding of the code. We might end up only understanding our project at a qualitative level, not the implementation level."

**Anders’ Rebuttal:**
He argued that the core team is already too small and unsustainable (running on "50% hobby time"). Agents act as a force multiplier, allowing a shrinking team to maintain a massive codebase.
> "The goal isn't automation; it's acceleration. I think the optimal future is **50% Human, 50% Token**. Humans provide the high-level reasoning and the 'vibes,' and the AI handles the implementation details."

**Anders** also noted a useful middle-ground workflow: asking the AI to write comments suggesting changes (`# CLAUDE_SUGGESTION`), which the human then manually implements. This forces the human to read and understand the code while still benefiting from the AI's logic.

**Licensing Offer:**
The session concluded with a practical proposal. Recognizing that many universities (like Oslo) prohibit paying for individual AI subscriptions, Handley offered to use his Cambridge grant funding (allocated for "AI Compute") to buy **Claude Code licenses** for the GAMBIT core team for a month-long experiment.

### 9. Closing Remarks

Handley summarized the session by comparing the current AI moment to the arrival of electricity or the printing press.
> "This is a massive IP land grab, effectively the biggest in history. It’s like the printing press arriving and the monks being pissed off that everyone can read. But even if the models don't get any smarter than they are today, this changes everything about how we do science."

He urged the group to start using these tools immediately—specifically Layer 3 Agentic tools—to avoid being left behind in the "Chat" era. The meeting adjourned with plans to continue the discussion over pizza.
