# Conceptual audit

I treated Will’s direct remarks as primary evidence. Material supplied by agents, historical summaries, or the assistant remains assistant-side unless Will explicitly accepted, restated, or built on it. Several attractive formulations never crossed that boundary.

## 1. The concepts

Ordered by how much of the emerging talk depends on them.

### 1. A third narrative: AI by scientists, for scientists

**Status:** Will’s central thesis.

The talk should displace the Silicon Valley story of autonomous machine scientists with an account of AI designed around the work scientists actually want done.

**Introduced by:** Will.

**First formulation:**

> “I want them to understand there is another narrative for AI in science that is not CMBAgent and the agentic world that silicon valley is driving us towards. AI by scientists, for scientists.”

This is not principally a product-sovereignty pitch, and Will rejected making the foil explicit because “Boris is a friend.” The aim is simply “planting the narrative.”

---

### 2. The synthesis rejects the shared premise; it is not a midpoint

**Status:** Will’s framing, sharpened after correcting the assistant.

The pro-agent and anti-agent camps both organise the debate around autonomous AI doing science. Will wants a Hegelian synthesis: neither side wins, and the synthesis does not split the difference; it changes what the argument is about.

**Introduced by:** Will, after the assistant warned that “synthesis” might sound like compromise.

**First formulation:**

> “I’m using synthesis in the sense of thesis, antithesis, and synthesis in the Hegelian sense. So I don’t need to necessarily use the word synthesis, and I take your point that people might imagine that I’m trying to say a midpoint, but I’m very much not.”

The visual thesis/antithesis became “Scientists versus agents”: Natalie Hogg and Hiranya Peiris on one side, the AI labs or systems on the other. The synthesis is to happen in Will’s speech, not as a third box on the slide:

> “bear in mind slides are for me to speak over”

---

### 3. What scientists actually delegate is bounded, practical work

**Status:** Will’s load-bearing empirical claim.

The useful domain is not end-to-end scientific authorship but scripting, routine implementation, scaling known analyses, and bounded judgement.

**Introduced by:** Will.

**First formulation:**

> “The thing we really use AI for [scripting, grunt work, scaling an established analysis, simple judgement] can be achieved with local models.”

Later examples broadened this beyond writing a reference routine:

> “it’s also being able to string things together, like installing a step, running established pipelines, and using it to help curate HPC, using it to run analyses during a conversation”

The case-study taxonomy ultimately became:

- scientific coding from existing specifications;
- well-posed examination problems;
- workshops and meetings as the beginning of ambient work.

The assistant proposed “grunt at scale, interpretation human” as a compact formulation. It fits Will’s examples, but Will did not explicitly adopt that exact phrase.

---

### 4. Human judgement remains the scarce and decisive input

**Status:** Strongly implied by Will’s selections and corrections; several exact formulations came through historical material rather than being newly coined in the live exchange.

Agents can implement, test, search, curate, and scale; scientists still decide what question matters, whether an output is adequate, and whether it should enter the scientific record.

The clearest Will-side evidence in this session is his account of the unmerged `emcee` port:

> “I didn’t merge it because I didn’t have time and I wasn’t confident that it was good enough to go into Blackjaxx as is, but it was certainly good enough to produce title slides.”

That separates three standards that assistant drafts repeatedly collapsed:

1. possible to produce;
2. adequate for a specific use;
3. adequate to merge as maintained scientific software.

A historical line recovered during the session expresses the same division more cleanly:

> “I wrote no code, but made the scientific decisions.”

The assistant’s formulation that “the bottleneck moved from writing to checking” is a persuasive synthesis of the JAX examples, but Will did not explicitly endorse that sentence.

---

### 5. Agentic work means closing the code–execution–feedback loop

**Status:** Initially supplied from Will’s earlier talks; actively selected and developed by Will here.

A chat model becomes an agent in the relevant sense when it can edit files, run code, inspect outputs and errors, and feed them back into the next attempt. The difference is access and feedback, not simply a cleverer chatbot.

**Introduced in this session by:** the assistant, using what it described as Will’s existing “copy-paste-run-debug” line. Will explicitly chose to retain it:

> “We should definitely be speaking about the copy-paste-run debug line.”

The characteristic vocabulary is:

> “copy-paste-run-debug monkey”

and, from the assistant’s first slide-two draft:

> “Agentic systems are what happens if you let ChatGPT edit files, run commands, and inspect the behaviour.”

Will’s own elaboration of the desired diagram was:

> “an agent producing code, generating errors … and then those errors piping back into the agent … as well as the output, something that gets an abstract, closed loop”

The assistant proposed the broader thesis that both sides of “Scientists versus agents” are arguing over autonomy when “the thing that actually changed is not autonomy but access.” Will continued building the slide around that distinction, but never repeated the phrase himself. It is best marked as an assistant interpretation that the design work appears to have provisionally accepted.

---

### 6. Ambient AI is the same closed loop extended from the computer into the room

**Status:** The core connective idea emerged through collaboration; Will supplied its most important operational form.

Once an agent can participate in the code loop, the next move is to give it access to the conversation, whiteboard, meeting, supervision, and shared scientific context—without requiring people to stop and operate a separate interface.

The assistant first made the exact scale analogy:

> “Access to your computer closed the loop around code. Access to the room closes it around the conversation, the whiteboard, the supervision.”

Will’s later formulation grounded it:

> “there being a threshold cross when these things happen in real-time so that it can happen as you are thinking.”

And he explicitly rejected a screen-centred representation:

> “The whole point is that I want to be encouraging ambience here, not screens.”

This is probably the most valuable new bridge generated by the design conversation: agentic coding and ambient AI are not separate topics. They are the same feedback-loop architecture at two scales.

---

### 7. Ambient means freedom from continual interface operation, not AI silence

**Status:** Will’s position, partly recovered through the session’s historical synthesis and reinforced by current comments.

Ambient AI is “just there”: scientists continue working normally while the system listens, retrieves, acts, and returns useful material without continual clicking or prompting.

Earlier words recovered in the session include:

> “The AI is just there, like electricity”

> “Not a separate activity — you do your work, the AI is ambient”

and:

> “what does ambient AI that you don’t need to be constantly touching or looking at look like?”

In the current slide discussion Will tied this to:

> “I wrote this talk without typing”

and:

> “my hands-free, Tony Stark-like system”

Important correction: ambient does **not** mean a mute or passive machine. In the historical material, Will corrected an assistant draft that had inferred this:

> “I very much want them to be active contributors. That’s why I use the word ambient AI.”

The preference for screens over speech was an efficiency choice, not a rejection of conversational participation.

---

### 8. “Multiplayer mode” must operate where scientific collaboration actually occurs

**Status:** Will insisted this was essential after the assistant initially omitted it; much of the detailed articulation was reconstructed by the assistant from Will’s earlier discussion with Toby.

Vendor “multiplayer” assumes that shared work lives in Slack or an equivalent digital workspace. Scientific work also happens at whiteboards, desks, coffee, conferences, meetings, and supervisions. Ambient AI is multiplayer mode transplanted onto those surfaces.

Will’s correction was blunt:

> “You’re not capturing ‘multiplayer mode’ or the nuance of the discussion with Toby yesterday.”

The recovered Will formulation was:

> “we want multiplayer mode that works in the surfaces where surface is very broadly interpreted. Where a surface could be a whiteboard, it could be us sat at a desk with two laptops in front of us. It could be a Slack channel or coffee”

And:

> “make multiplayer mode ambient at a [whiteboard], rather than it sitting in a Slack workspace”

A useful internal distinction also surfaced in that historical quotation:

- first, ambient single-user interaction that requires no continual touching or looking;
- later, connection to always-on back-end agents that act or begin work proactively.

That staging is Will’s prior formulation, not merely the assistant’s.

---

### 9. Real-time latency produces a qualitative capability threshold

**Status:** Will’s idea, although the assistant initially over-promoted it into “latency, not intelligence.”

The important threshold is not merely benchmark capability. When the whole pipeline becomes fast enough to run while a scientist is thinking and talking, it changes category: from work submitted for later return into participation in the live scientific process.

Will’s clearest formulation:

> “there being a threshold cross when these things happen in real-time so that it can happen as you are thinking.”

The exam case supplies a second kind of threshold:

> “That’s a useful case study as it explains the level of capability that’s been here for a good year now. So this is something where models reached a capability threshold.”

These are related but not identical:

- **competence threshold:** a model can solve well-posed problems at a useful level;
- **interaction threshold:** the system can act quickly enough to remain inside a conversation or train of thought.

The assistant’s opening proposal—“the threshold that matters is latency, not intelligence”—was too strong and was never adopted. Will continued to care about both capability and latency.

---

### 10. Local models are sufficient for the base load

**Status:** Will’s claim, but its precise strength remains unresolved.

Will argues that the routine work scientists actually delegate can be done with local models:

> “The thing we really use AI for … can be achieved with local models.”

He later sharpened his scepticism about frontier intelligence:

> “I don’t think we need frontier AI to do _any_ of that? What these provide is harness and cheap tokens. Gemma is good enough at quatnum field theory to help edit text just fine. Don’t think you’re actually any smarter than them.”

Then corrected the economics:

> “and by cheap tokens I mean subsidised tokens.”

The assistant challenged the strongest version, pointing out that several headline successes used frontier models and suggesting the defensible claim is “local for the base load; frontier for the leading edge.” Will did not answer that challenge directly. Consequently:

- “local models can do the practical base load” is Will’s position;
- “frontier models add no relevant intelligence at all” is asserted but not demonstrated here;
- “frontier for the leading edge, local for the base load” is the assistant’s proposed reconciliation, not Will’s adopted wording.

---

### 11. The vendors’ present advantage may be harness and subsidy rather than model intelligence

**Status:** Will’s argument; empirically open.

The proprietary offering may consist chiefly of a polished agent harness and temporarily subsidised inference, not an irreplaceable cognitive advantage.

**Introduced by:** Will.

**First formulation:**

> “What these provide is harness and cheap tokens.”

Corrected immediately to:

> “by cheap tokens I mean subsidised tokens.”

This matters because it makes local ownership economically and strategically plausible without requiring the local model to beat the frontier model on every benchmark. The assistant extended this into a “repricing” argument—the lab that moves base load locally survives when token prices become honest—but Will did not explicitly endorse that extension during this session.

---

### 12. Locality is justified by confidentiality, ownership and seamless operation

**Status:** Mixed: confidentiality is demonstrated and accepted; broader sovereignty language was mostly assistant-side.

The exam work is the cleanest proof that some scientific tasks cannot leave the building, so local execution is not merely ideological.

Will:

> “alan is not ready, but I can use the exam checking example of that in action”

The assistant then framed it as the case where the frontier route was unavailable because exam material was confidential. Will accepted reporting the case, though not displaying the material:

> “I can report on the case.”

The larger claims—“the record never leaves the room,” “capture must be owned by the recorded,” and a general “sovereignty” argument—came primarily from the assistant and recovered grant material. Will did not reject them, but neither did he make them the talk’s central argument. Indeed, the assistant itself later proposed “no sovereignty sermon.”

---

### 13. Scientific reasoning is an unrecorded process asset

**Status:** Initially assistant/historical language; concretely adopted by Will through the transcript case.

The lab produces not only papers but the conversations, corrections, judgements and reasoning that produce them. Recording those processes creates reusable context.

The assistant’s early formulation was:

> “A group produces papers and the reasoning that produced them; only the first leaves a trace.”

Will did not select that as the opening. However, he supplied a compelling concrete form:

> “This session is also a good example of the ambient AI — I’m using transcripts from across the year to remind me of good things I’ve said that I’d like to include. The transcripts are a definite case study.”

This makes the talk self-demonstrating: a talk about ambient capture is being constructed from the captured history. The assistant’s phrase “the talk about the corpus is being written from the corpus” is a good summary, but is assistant language.

---

### 14. Process data may be the lab’s distinctive advantage

**Status:** Assistant/historical synthesis; not explicitly adopted in this design session.

The frontier laboratories have public literature and general data, but not the situated process data of how a scientific group reasons, supervises, corrects, and decides. Transcripts, decision records, skills, code and agent traces could therefore become a compounding asset.

Recovered historical wording included:

> “frontier labs can’t compete because the process data of real science is unrecorded”

and the earlier aim:

> “isolate and capture the irreducible human input”

This fits the transcript case and the “AI by scientists” thesis, but Will did not explicitly return to the “moat” or competitive-advantage version. It should not be attributed as a settled closing claim without confirmation.

---

### 15. The relevant AI system is instrumentation, not an artificial colleague or scientist

**Status:** Assistant proposal based on Will’s older analogy; compatible with his position but not adopted as the opening.

The assistant proposed that ambient AI be described as scientific instrumentation built by scientists, analogous to earlier locally built instruments:

> “this is instrumentation, scientists have always built their own”

It later offered “Instrumentation” as one possible opening:

> “Cambridge astronomy has built its own instruments for a century; this is one, not a product you buy.”

Will instead chose “Scientists versus agents” for the opening. So instrumentation remains valuable connective tissue, but not a chosen top-level frame.

---

### 16. Well-posedness is a key boundary on demonstrated capability

**Status:** Will’s framing.

The exam case is not evidence that models can conduct unconstrained research. It shows that they crossed a striking threshold on defined, assessable problems.

Will:

> “the second slide, for exams, talks about the capability that these things can do if given well-posed problems.”

This makes the exam case conceptually important, not merely colourful. It establishes high capability while preserving the distinction between solving bounded problems and choosing or interpreting a research programme.

---

### 17. Existing specifications and checkable outputs define a productive coding regime

**Status:** Will’s correction to an overly narrow slide; partly articulated by the assistant.

Ports and reimplementations work well because a reference routine, tests and numerical comparisons supply a specification and verification path. But Will warned against reducing scientific coding to ports alone:

> “Pintific coding for a reference routine, it’s also being able to string things together…”

The assistant/Codex subtitle—“existing specifications, checkable results”—captures the first half but not the whole category. Will’s broader category also includes installing software, operating established pipelines, HPC curation, and conversational analysis.

---

### 18. Abundance of use may be stronger evidence than one benchmark result

**Status:** Assistant proposal; not explicitly adopted.

The assistant argued that the lab’s many ports, recipes and live loops should feel like an “overflowing” practice:

> “The abundance is itself the evidence”

This would distinguish an established working mode from a one-off demo. Will did want “jaxwavelets and everything else” grouped behind `emcee`, but did not explicitly endorse “abundance itself” as an argumentative principle.

---

### 19. The human should cease being the transport layer

**Status:** Assistant interpretation of Will’s visual brief.

The copy/paste image makes the scientist visibly responsible for carrying code and errors between chat and editor. Agentic work removes that clerical transport while retaining human direction.

The assistant described the generated picture as:

> “the human is visibly the transport layer.”

Will liked the image and asked for its closed-loop counterpart, so the visual distinction is accepted. The phrase itself is assistant-authored.

---

### 20. “Hands-free” is a mode of working, not a separate case result

**Status:** Assistant classification; Will had raised the example without knowing where it belonged.

Will proposed:

> “I wrote this talk without typing”

and:

> “my hands-free, Tony Stark-like system”

The assistant argued that this belongs beside Toby’s whiteboard demonstration because it demonstrates the ambient mode at the scale of one person, rather than constituting a fourth scientific case study. Will did not answer that placement decision before the session ended.

---

### 21. The opening and demonstration will carry most of the audience’s memory

**Status:** Will’s design principle.

The abstract argument must be present immediately; it cannot be deferred until after a catalogue of examples.

Will:

> “The opening is also the most important bit of the talk, since it’s what people will actually remember in practice. They won’t remember the rest of the talk, they’ll remember Toby’s demo and my start.”

This corrected the first generated deck, whose cases began before any thesis had been established.

---

### 22. Slide titles are the slide’s retained proposition

**Status:** Will’s explicit presentational principle.

A title should be sufficiently specific that someone remembering only the title still remembers the slide’s point.

Will:

> “many people will just read the title”

and:

> “if they remember one thing with the slide, the title can be it”

This caused “Access to the computer” to be rejected as too amorphous and “Scientists versus labs” to be corrected because “labs” could mean either frontier AI labs or the ambient laboratory Will wants to build.

---

### 23. Slides support speech; they should not contain the whole synthesis

**Status:** Will’s explicit principle.

The opening slide should display the opposition and leave Will to make the Hegelian turn orally.

Will:

> “bear in mind slides are for me to speak over”

This is why the third “synthesis” element was not added visually to slide one.

---

### 24. The talk’s purpose is cultural implantation, not recruitment or a call to action

**Status:** Will’s explicit decision.

Asked whether the conclusion should offer access, recruit collaborators, or propose a pilot, Will answered:

> “planting the narrative.”

Therefore assistant suggestions for discussion prompts, offers of access, supervision recruitment, or an ai@cam pitch should not be mistaken for settled content.

---

### 25. Failure modes are not part of this talk

**Status:** An explicit rejection, therefore important.

The assistant initially made failures a major section. Will rejected both its relevance and its allocation of time:

> “we don’t want to discuss what failed, all of that is totaly unecessary, and not appropriate.”

Failures may be useful background constraints, but they are not part of the selected abstract argument.

---

## 2. The through-lines

### Connections Will made himself

**Practical delegation → local models.**  
Will directly connects the bounded tasks scientists really use—“scripting, grunt work, scaling an established analysis, simple judgement”—to the claim that local models are sufficient.

**Closed-loop agents → hands-free work.**  
He connects error/output feedback to no longer transporting material manually:

> “errors piping back into the agent … something that gets an abstract, closed loop”

and then links that to “allowing you to go hands-free.”

**Real time → ambient participation.**  
His final reflection explicitly says that running pipelines and analyses during a conversation plants the ambient idea, because a threshold is crossed when the system acts “as you are thinking.”

**Coding → well-posed exams → workshop/room.**  
Will defines the three chosen cases not merely by topic but by conceptual escalation:

1. agents readily produce certain scientific code;
2. models exhibit high capability on well-posed problems;
3. the loop begins to leave the individual computer and become ambient.

**Transcripts → ambient AI → this talk.**  
He makes the session itself a case study: captured conversations from across the year recover ideas for a talk about captured conversations.

**Critiques of agentic science → Hegelian synthesis.**  
He explicitly wants Natalie/Hiranya versus agents as thesis and antithesis, with his account as a synthesis that is “very much not” a midpoint.

**The talk’s argument → the workflow producing the talk.**  
Not stated in quite these words by Will, but strongly enacted by him: Will supplies narrative and scientific judgement; assistants gather evidence and constraints; Codex makes slides; Will inspects and redirects. He repeatedly insists that narrative cannot be delegated wholesale.

### Connections made by the assistant

**Computer access and room access are the same move at different scales.**  
This is the strongest assistant-origin synthesis:

- access to the computer closes the coding loop;
- access to the room closes the scientific-conversation loop.

It fits Will’s final reflection exceptionally well, but the exact two-scale analogy remains assistant-authored.

**Both sides of the autonomy debate share the wrong axis.**  
The assistant argues that Natalie/Hiranya and the frontier labs are all debating whether AI can autonomously do science, whereas the material change is practical access to tools and context. Will built slide two around this, though he never restated the “wrong axis” formula.

**Ambient single-player → ambient multiplayer.**  
The assistant distinguishes hands-free assistance for one scientist from agents operating across shared surfaces and people. This division derives from Will’s prior Toby conversation and is likely faithful, but it was not newly formulated by him here.

**Human judgement → verification becomes the new work.**  
The assistant reads `emcee`, `jaxwavelets`, the archive, and LSST as instances where generation becomes cheap and scientific checking, acceptance and interpretation become the bottleneck. This is well supported by the examples, but not yet a Will-approved slogan.

**Case chronology → capability threshold.**  
The assistant suggests that ordering grants, exams, coding, archive and whiteboard work chronologically would let the audience watch capability cross thresholds. Will did not answer this proposal.

**Hands-free desk and ambient whiteboard are the same idea at two scales.**  
The assistant places “I wrote this talk without typing” next to Toby’s loop: one person freed from the keyboard, one room freed from operating a recorder. Again, a strong synthesis, not yet adopted.

**Locality links economics, confidentiality and ambient latency.**  
The assistant combines three reasons for local systems:

- confidential material cannot leave;
- subsidised cloud tokens may not remain cheap;
- live conversation requires low latency.

Will supplied each ingredient unevenly, but did not explicitly combine them into one argument.

### Ideas that recur at different scales

| Scale | Closed loop | Human role | Ambient implication |
|---|---|---|---|
| Code | edit → run → error/output → revise | specifies and verifies | no manual copy/paste |
| Analysis | install → run pipeline → inspect → redirect | chooses method and interpretation | work can proceed during thought |
| Meeting | listen → retrieve/compute → display → discuss | challenges and redirects | AI participates without a separate interface |
| Lab memory | capture → index → recover → reuse | decides what matters | the group’s past reasoning becomes live context |
| Collaboration | shared conversation → tasks → parallel agents → returned results | coordinates people and judges outputs | “multiplayer mode” moves beyond Slack |

The first two rows are substantially Will’s; the latter three are a synthesis across Will’s examples and assistant proposals.

---

## 3. The unresolved ideas and decisions

### Conceptual questions

1. **How strong is the local-model claim?**  
   Is the assertion that local models can perform the routine base load, or that frontier models are not meaningfully more capable for any relevant scientific task? The assistant challenged the stronger claim; Will did not resolve it.

2. **What exactly do frontier providers contribute?**  
   Will says “harness and subsidised tokens.” There was no final decision about whether to present that openly, qualify it, or support it with an evaluation.

3. **Does “human judgement remains central” become an explicit claim?**  
   It is strongly present in the cases, especially the unmerged port, but no final synthesis language was chosen.

4. **Is the key transition autonomy → access?**  
   The assistant made this the hinge between slides one and two. Will liked the resulting direction but did not explicitly declare it the talk’s thesis.

5. **Does the closing use the “instrumentation” frame?**  
   It aligns with “AI by scientists, for scientists,” but Will chose a different opening and never decided whether instrumentation should return at the end.

6. **Is process data a strategic advantage or merely useful memory?**  
   Will unquestionably values transcripts as a case study. The stronger claim that the lab has a unique process-data “moat” was not adopted.

7. **How do consent and control fit the abstract story?**  
   Historical material repeatedly says “natural, or consensual” and discusses local/revocable capture, but Will excluded a failure/GDPR discussion. The positive conceptual role of consent remains unintegrated.

8. **What is the precise boundary between agentic work and agentic research?**  
   Natalie’s critique and Will’s cases imply a distinction between tool-using agency and autonomous interpretation, but the talk has not yet named it cleanly.

9. **Should the case studies be organised by type or chronology?**  
   Will selected coding → exams → workshop/ambient. The assistant later proposed chronological escalation. Will did not answer.

10. **Is “abundance itself is evidence” part of the argument?**  
    The assistant proposed an overflow gallery; Will only clearly requested that multiple coding examples share one slide.

11. **Where does “I wrote this talk without typing” belong?**  
    Will asked whether it fits. The assistant placed it in the ambient section next to Toby’s demonstration. No confirmation followed.

12. **What is the final synthesis/close?**  
    The session ended precisely when Will requested collection of the abstract concepts. The conclusion was not designed.

### Deferred content choices

- Whether the opening’s right side shows corporate labs or named agentic-science systems.
- Whether Natalie and Hiranya’s arguments are quoted, represented only through article crops, or mostly spoken over.
- Whether the paper crops should use Nature Astronomy pages or arXiv.
- Hiranya Peiris’s headshot source.
- Whether the closed-loop counterpart to the copy/paste image is an abstract diagram, a generated image, or omitted.
- Final title and subtitle for the agent-definition slide.
- Whether the grounding/hallucination qualifier should remain. Will said:

  > “I think we can drop the agentic systems of partially grounded reality”

  so it appears rejected, despite the assistant later trying to restore it.

- Whether LSST should appear as a forward reference to Nikhil’s talk; his scheduled subject may not include it.
- Which optional cases, if any, join coding, exams and workshops: archive, reading/arXiv, grants, or administrative “rubbish.”
- Whether the transcripts case is a separate slide or part of the ambient synthesis.
- Whether Toby’s loop is live or video. Late discussion refers increasingly to “Toby’s video,” but logistics were not finally settled.
- Toby’s attendance/registration and the room’s display/network path.
- The actual talk duration. The early session establishes a 20-minute slot, while later design repeatedly speaks of a fifteen-minute talk. The report request also calls it fifteen minutes; the dialogue never explicitly reconciles the two.
- Whether there is a fallback recording for the demo.
- Whether the `emcee` numerical validation figure still exists.
- How prominently to show that the `emcee` code was useful but never merged.
- Whether there should be an overflow/cookbook frame.
- Whether the assistant’s proposed arXiv/reading case is needed given Nikhil’s preceding talk.
- How to present confidential exam results without displaying exam content.

### Questions asked but not answered

- Whether the eval harness has slide-worthy local-versus-frontier results.
- Whether the “subsidised tokens” claim should be stated openly.
- Whether supervisions should be mentioned as a future direction.
- Whether Natalie and David Yallup should be named or their arguments absorbed.
- Whether the Astro Legacy Archive belongs in the main case sequence.
- Whether the case-study sequence should become chronological.
- Whether the second closed-loop visual should be drawn in TikZ.
- What the final connective “wrapping” should be—the question that precipitated the present audit.

---

## 4. Vocabulary to preserve

These are Will’s own or repeatedly selected phrases. A slide should prefer them to polished paraphrases.

### Central argument

- **The Ambient AI Lab**
- **Scientists versus agents**
- **AI by scientists, for scientists**
- **another narrative for AI in science**
- **planting the narrative**
- **thesis, antithesis, and synthesis**
- **very much not [a midpoint]**
- **the agentic world that Silicon Valley is driving us towards**

### What AI is for

- **scripting**
- **grunt work**
- **scaling an established analysis**
- **simple judgement**
- **well-posed problems**
- **string things together**
- **installing a step**
- **running established pipelines**
- **curate HPC**
- **running analyses during a conversation**
- **the rest of the rubbish a scientist does**
- **GPUs for science, LLMs for development** — historically recovered Will phrase; useful if accurate to the intended claim.

### Agentic work

- **copy-paste-run-debug monkey**
- **edit files, run commands**
- **errors piping back into the agent**
- **the output**
- **an abstract, closed loop**
- **This closes the scientific loop** — assistant/Codex wording, but used throughout the slide design.
- **access to the computer** — conceptually relevant, although rejected as too amorphous for a title.
- **hands-free**
- **Tony Stark-like system**
- **I wrote this talk without typing**

### Ambient work

- **ambient AI**
- **multiplayer mode**
- **surfaces**, “very broadly interpreted”
- **whiteboard**
- **two laptops in front of us**
- **Slack channel or coffee**
- **you don’t need to be constantly touching or looking at**
- **always on and listening for the next thing to do**
- **as you are thinking**
- **encouraging ambience, not screens**
- **The AI is just there, like electricity**
- **you do your work, the AI is ambient**
- **active contributors**
- **natural, or consensual**

### Locality and economics

- **local models**
- **harness**
- **subsidised tokens**
- **the record never leaves the room** — principally assistant/historical wording; use only if Will wants the sovereignty claim.
- **local hardware**
- **base load** — assistant terminology, not Will’s.

### Human contribution and verification

- **I wrote no code, but made the scientific decisions**
- **good enough to get on with my results**
- **I didn’t have the time, or the confidence**
- **still open, never merged**
- **do not normalise**
- **checkable results**
- **reference routine**
- **because that’s how we do science** — from James, quoted during the session; not Will’s coinage.

### Records and memory

- **the transcripts**
- **a definite case study**
- **using transcripts from across the year**
- **remind me of good things I’ve said**
- **the unrecorded half of science** — historical/project language, not newly endorsed as the opening.
- **isolate and capture the irreducible human input** — recovered historical Will formulation.

### Presentation method

- **slides are for me to speak over**
- **many people will just read the title**
- **if they remember one thing with the slide, the title can be it**
- **the opening is the most important bit**
- **they’ll remember Toby’s demo and my start**

---

## 5. Candidates for a synthesis slide

The strongest synthesis is not “AI is good but humans matter.” That is generic and does not explain the ambient lab. The material now supports a more specific four-part argument.

### Recommended four-concept synthesis

1. **Agents close a practical loop.**  
   They edit, run, inspect errors and outputs, and revise—removing the scientist as the copy/paste transport layer.

2. **Scientists supply the irreducible judgement.**  
   The high-value human act is choosing the problem, specifying standards, interpreting results, and deciding whether something is good enough to use or merge.

3. **Real-time operation moves the loop into the scientific process.**  
   Once code, retrieval and analysis happen “as you are thinking,” the system no longer sits beside the work as a chat interface. It becomes ambient at the desk, whiteboard, meeting and supervision.

4. **That system can be built locally, by scientists, for scientists.**  
   Much of the useful base load is bounded work within local-model capability; local infrastructure also makes confidentiality, persistent context and room-scale interaction possible. The frontier providers’ durable advantage may be harness rather than autonomous scientific intelligence.

### The connective tissue

A possible spoken argument—deliberately not polished into slide copy—is:

> The change was not that a model became a scientist. The change was that it gained access to the scientific loop. On the computer, that means writing code, running it and reading the error. In the lab, it means hearing the discussion, retrieving or running what is needed, and returning the result while the thought is still alive. Scientists continue to choose, test and interpret; the agent makes the surrounding implementation and recall cheap enough to happen continuously. That is the Ambient AI Lab: not autonomous science, but scientific work with the machinery present everywhere it is useful.

Attribution matters here:

- “The model did not become a scientist” is my synthesis of Will’s opposition to the autonomous-science narrative.
- “Access to the scientific loop” is the assistant’s strongest generated bridge.
- “while the thought is still alive” is my paraphrase of Will’s “as you are thinking”; the slide should use Will’s phrase.
- The final definition follows Will’s positions but was not spoken by him verbatim.

### A tighter three-concept alternative

If four ideas are too many for one closing frame:

- **Closed loop:** agents run and inspect the work.
- **Human judgement:** scientists decide what the work means and whether it is adequate.
- **Ambient scale:** put that loop at every surface where science happens.

Then “local, by scientists, for scientists” becomes the conclusion underneath rather than a fourth conceptual box.

### What should not anchor the synthesis

- **“Latency, not intelligence.”** Too absolute; Will uses both competence and latency thresholds.
- **“Humans versus agents.”** That is the opening opposition, not the answer.
- **“Agentic research is impossible.”** Will uses the failure to see good autonomous papers as evidence, but his own cases include powerful agentic work.
- **“Local models are as smart as frontier models.”** Will voiced this scepticism, but the session did not establish it.
- **“The process-data moat.”** Interesting, but not adopted strongly enough to carry the close.
- **“Instrumentation” alone.** Useful language, but too broad unless connected to the closed loop and the room.
- **A catalogue of case studies.** The first generated deck demonstrated that examples without the abstraction produce coverage, not narrative.

The abstract argument now available is therefore:

> **Agentic access closes the scientific work loop; scientific judgement remains human; real-time access extends that loop into the room; and local scientist-built infrastructure makes the resulting ambient lab possible.**

That is the clearest common structure across Will’s coding, exams, transcripts, workshops, hands-free system and Toby’s whiteboard loop.