Here is a summary of the "LLMBit" workshop presented by Will Handley at the GAMBIT collaboration meeting in Oslo, December 2025.

### Overview
Will Handley presented a practical workshop on the state of AI tools as of late 2025, arguing that the landscape has shifted fundamentally from "Chat-based" AI to "Agentic" systems (autonomous tools that can execute terminal commands). He demonstrated how these tools can autonomously navigate, debug, and contribute to complex scientific codebases like GAMBIT, urging researchers to adopt these workflows immediately to remain competitive.

### Key Topics Covered
*   **The 2025 Timeline:**
    *   **Feb 2025:** Reasoning models (like OpenAI's **o3**) achieved PhD-student capabilities. Handley shared an anecdote where o3 took a Cambridge Astrophysics exam "blind" and achieved the highest mark in 25 years.
    *   **May 2025:** The rise of **Agentic Systems** (Claude Code, Cursor) which moved AI from a chatbot interface to the command line.
*   **The Three Layers of AI:**
    1.  **Autocomplete:** GitHub Copilot (tight feedback loop).
    2.  **Chat-based:** ChatGPT/Claude (requires copy-pasting code; turning the user into a "debug monkey").
    3.  **Agentic:** Systems that have terminal access, can edit files directly, and run git commands (the focus of the talk).
*   **Context Engineering:** The shift from "prompt engineering" (crafting the perfect question) to managing the file context and environment the AI operates within.
*   **Economics of AI:** While token costs have dropped 300x in a year, heavy agentic usage generates thousands of dollars in value. Handley advises researchers to "rinse the VCs" by using the currently subsidized $20/month subscriptions.

### Main Demonstrations
Handley performed live demonstrations using **Claude Code** directly on the GAMBIT repository:
*   **Repository Analysis:** Claude Code initialized in the directory, read the README and CMake files, and mapped the project structure without prior knowledge.
*   **Autonomous Compilation & Debugging:**
    *   Handley asked the agent to compile GAMBIT on an Arch Linux system (a notoriously difficult task).
    *   The agent attempted `cmake`, failed due to an Eigen version mismatch (Eigen 3 vs 5), read the error logs, located the specific CMake module causing the issue, applied a patch, and successfully compiled.
*   **Git Integration:** The agent wrote a detailed commit message, created a new branch, and pushed the changes to GitHub autonomously.
*   **Non-Coding Tasks:** Demonstrations of using agents to fix broken links in the internal Wiki, reformat LaTeX presentations, and organize meeting transcripts.
*   **Model Context Protocol (MCP):** A demo of a custom tool allowing Claude to query the ArXiv API via a Python script to find and download recent papers by specific authors.

### Key Takeaways and Recommendations
*   **Upgrade Immediately:** If you are not using Agentic systems (Layer 3), you are behind. The productivity gap between users and non-users is widening.
*   **Stop Copy-Pasting:** Move away from the "Chat" loop. Give the AI access to the terminal so it can run, debug, and fix its own errors.
*   **Safety First:** **Always** run agentic tools inside a Git repository. If the AI hallucinates and deletes your work (which happens), you must be able to `git reset`.
*   **Acceleration vs. Automation:** The goal isn't just to generate papers automatically (which leads to low quality) but to accelerate the "boring" parts of research (tooling, installation, formatting) so humans can focus on physics.
*   **Local Models:** While cloud models are currently superior, the long-term goal should be running these models locally on university hardware for privacy and data security.

### Discussion Points
*   **Codebase Knowledge:** Anders (audience) noted that while AI is great for prototyping, there is a fear that the core team might lose deep understanding of the code if they rely too heavily on agents. Handley countered that agents allow small teams to maintain massive projects that would otherwise be unsustainable.
*   **Legacy Bugs:** Anders mentioned he used Claude to find bugs in files that hadn't been touched in 10 years, suggesting it acts as a powerful review tool.
*   **Data Privacy & IP:** There was significant discussion about the ethics and security of sending unpublished code/data to Anthropic/OpenAI. Handley argued the "IP land grab" has already happened and that local models are the only eventual solution to this.
*   **University Policy:** Most universities will not pay for individual agentic subscriptions (like Claude Pro) yet; researchers often pay out of pocket or via grant "consumables."
*   **Human-AI Synergy:** Handley estimates the current optimal workflow is "50% Human, 50% Token," emphasizing that humans are still irreducible for high-level reasoning and "vibes."