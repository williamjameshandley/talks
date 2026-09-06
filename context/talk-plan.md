# KAS2026 talk plan: the ambient AI lab

Slot: Monday 7 September 2026, 11:50–12:10, Hoyle Lecture Theatre, AI for
Scientific Practice workshop (Kavli Astrophysics Symposium satellite day).
Billed in the programme as "Local transcription / ambient AI lab".
Preceded by Nikhil Sarin (11:30, arXiv / structured summaries); followed by
20 minutes of joint discussion. Morning session: Miles Cranmer (foundation
models), Inigo Zubeldia (agentic systems / reviewing). Organisers: Debora
Sijacki, Sandro Tacchella, Nikhil Sarin. Slides go to the in-house system
(PDF or PPTX) by 20:00 the day before; a LOC member tests them in the break.
Speakers were asked to bring discussion points.

Organisers' steer (Nikhil, 24 June): real case studies with a demonstration
element rather than lists of what one can do with AI; discussion of limits,
biases and failure modes across the sessions. Will's acceptance (14 Aug):
speak "on the ambient AI lab we are all building, of which the transcription
and note-taking is one component".

## Purpose

Plant one narrative in the Cambridge/Kavli community. There is a way to do AI
in science that is not the autonomous-scientist story: AI by scientists, for
scientists. No ask, no offer; the talk exists to make the narrative visible.

## Thesis

1. What we actually delegate to AI is scripting, grunt work, scaling an
   established analysis, and simple judgement. Every interpretive act stays
   human.
2. The model was never the scarce thing. What the frontier vendors supply is
   a harness and tokens priced below cost. Both are replaceable: the tokens
   by our own GPUs, the harness by what we build. Gemma 4 is good enough at
   quantum field theory to edit a paper.
3. The record of how science is actually done (whiteboards, supervisions,
   meetings) exists nowhere but in the lab. Capturing it, locally, is what
   ambient AI is for, and it is what scientists would design that a vendor
   would not.

## Decisions taken in the planning interview

- Open with case studies driven by ordinary agentic work; then ambient AI;
  then the move to local models.
- All case studies one slide each.
- No failure-modes section. Constraints (local, consensual, no change to
  participants) are stated as design principles, not as failures.
- Do not name CMBAgent or Denario. The narrative stands on its own and the
  room draws the comparison.
- Exam checking is reported, not shown.
- Alan is not ready to demonstrate; the exam case is Alan in action, told.
- Toby's whiteboard loop is the live demonstration.
- LSST decision agent is a forward reference to Nikhil's talk (confirm with
  Nikhil that LSST is in his 11:30 slot; the programme bills it as arXiv
  summaries).
- Not Dily's evidence grid.
- Astro Legacy Archive is a case study.
- SBI4GALEV and the arXiv pipeline are flagged explicitly.
- emcee-to-blackjax is the starter, with the other JAX ports as the pattern.
- Open: whether Natalie's Comment and David's argument are attributed by
  name or absorbed as framing. Natalie is at the Oxford workshop on the day.

## Arc

| Minutes | Section | Content |
|---|---|---|
| 0–2 | What we use agents for | One sentence on the lab's practice against the AI-scientist story the room has read about. |
| 2–9 | Case studies, one slide each | JAX ports as standard practice (emcee first); Astro Legacy Archive; exam checking (told); SBI4GALEV; arXiv pipeline; LSST decision agent (forward reference). Plus the cookbook grid as one slide. |
| 9–12 | The pattern | Grunt at scale, interpretation human. Frame from Natalie's Comment and David's observation (attribution TBD). |
| 12–16 | Ambient AI | What the tool looks like when scientists design it: multiplayer mode at the surfaces where research happens. Inventory of what the lab has already done live. Toby's loop, live. |
| 16–19 | Why local | Harness plus subsidised tokens is all that was sold. The threshold graph. The record never leaves the room. |
| 19–20 | Where it goes | Supervisions in Lent; the Cosmos build. Two discussion prompts. |

## Case studies (one slide each)

### JAX ports as standard practice
- emcee ensemble sampler implemented in BlackJAX under supervised execution:
  15 minutes; numerical agreement with reference emcee. (Cookbook card;
  URF progress report: "took 15 minutes instead of months".)
- The pattern behind it: `jaxwavelets` (Apr 2026; PyWavelets reproduced to
  1e-14 across 1,177 tests in 4.7 hours; 65 external reviews, 60% rejected
  before merge; on PyPI), `jaxPOSEIDON` (May 2026, exoplanet transmission
  spectroscopy), `jax-lens` (Dec 2025, pixel-perfect PyAutoLens),
  `jim-emulators` (GW waveform emulators, "from an empty repository in
  eight hours"), `blackjax-gw`, Charlotte Priestley's `jaxsgp4` (1500×),
  `k2-18b-blackjax-retrieval` (built live in the room with Madhusudhan,
  13 May 2026).
- Method sentence (jaxwavelets write-up): "I wrote no code, but made the
  scientific decisions."
- Sources: handley-lab GitHub; handley-lab.co.uk cookbook (`_data/recipes.yml`).

### Astro Legacy Archive
- Measurement-level astronomy data (sky maps, timestreams, event lists,
  visibilities, chains) is absent from the ML data ecosystem: 2,130 Hugging
  Face datasets audited, 449 astronomy, none of the above beyond one broken
  strain fragment.
- Each dataset a rebuildable projection of its canonical archive: Parquet,
  HATS-style conventions, provenance, checksums, licence and the regeneration
  command in every card. 130 cards written by agents; ~2.5 TB uploaded.
- Access case measured: LAMBDA 1.1 MB/s vs Hugging Face CDN 103.6 MB/s.
- Pilot: Haslam, WMAP, COBE, IRAS, ROSAT maps (~8 GB) plus the complete EHT
  visibility releases. Governance home: EuCAIF SBI working group (James
  Alvey). Pre-registered success criteria.
- The human part on show is the judgement: "do not normalise; you'll only be
  normalising for this set, not for the sets that come after."
- Sources: personal deck `reference/astro-legacy-archive/proposal-v0.3.md`,
  `eucaif-onepager-draft.md`; records 2026-09-03 14:36.

### Exam checking (reported)
- Exam papers cannot leave the building. Qwen 3.5 and Gemma 4 on the office
  box (2× RTX PRO 6000) checked Part III/CATAM material and found errors a
  human examiner had missed; "toe to toe with Qwen 3.5" against Anthony
  Challinor on the same paper.
- The instance where the frontier route was not merely unnecessary but
  impossible. Alan in action, told.
- Sources: records 2026-04-27, 2026-05-07, 2026-05-14, 2026-05-19.

### SBI4GALEV
- Chris Lovell's conference week (27–31 July): local live transcription,
  per-talk and per-day summaries, a white paper by the end of the meeting.
  Just over two thirds of speakers opted in; one declined. Public code at
  sbi-galev/ambient-ai. Write-up targeted at BAAS.
- Sources: record 2026-07-31 09:12; mail 2026-07-17 "Ambient AI write up".

### arXiv pipeline
- Toby Lovick: nightly Gemma-on-turing summaries, tools and keywords for
  ~800 papers; the scrolling-club site.
- James Alvey and Thomas Spieksma: structured summaries that return only
  verbatim quotes, equations and figures with IDs into a database, no
  generated prose. "I don't want to read AI's overconfident summary. I want
  original science."
- Sources: records 2026-08-06, 2026-08-07, 2026-08-13.

### LSST decision agent (forward reference to Nikhil)
- Redback plus blackjax bring transient inference to 85 minutes on one
  consumer GPU. Gemma then writes the "does this need reanalysis, and why"
  explanation across 50–70 thousand events. Classification was tried first
  and abandoned: "that isn't the question". Runs on a 16 GB card.
- Sources: record 2026-08-14 (James Alvey, Nikhil Sarin).

### The transcripts (this talk as its own case study)
- Roughly 1,000 recorded meetings over 18 months, Otter to mdrecord cards
  nightly, and the retrospective loop: this talk was planned by reading the
  year's transcripts back. Thirteen conversations from one Thursday read in
  full by agents; 394 records swept for every live-loop moment; the April
  internal talk, the Cosmos proposal, and remarks made in supervisions and
  lunches recovered verbatim and reused ("I'm going to try and do both at
  the same time when I write the report and the talk"; "the moat would be
  this is how scientists want ambient AI"; David's "if it were possible, it
  was possible last year").
- The point for the room: the record is the asset. Things said well once,
  in a meeting nobody wrote up, become retrievable and reusable months later
  without anyone having taken notes. The same loop that answers "what did
  we decide with James in August" builds a talk.
- Also the honest limit: the retrieval works because the recordings exist;
  the corpus is only as complete as the recording habit (the 17:59 chat on
  3 Sep was caught only because the recorder was already running).
- Sources: personal deck project "Ambient talk and grants" (fa0d11d4) with
  the research provenance; this planning session's transcript.

### Cookbook grid (one slide)
Complete recipes from handley-lab.co.uk/cookbook: jaxwavelets; emcee in
BlackJAX (15 min); GW emulator in eight hours; covariance computation from
24 hours to 1–2 minutes; CAMB torsion (N_eff double count, missing massive
neutrino pathway); CLASS discrete k-modes (FCB sign inconsistency); six-slide
deck live in five minutes; CosmoSIS→Cobaya port under adversarial review.
Seven further recipes proposed by group members at the 5 May workshop show
the pattern spreading: slide deck from a paper (Natalie), voice-note paper
debate and talk-rehearsal loop (Toby), interview-mode prompting, Matplotlib
infographics (Matt), LLM-managed HPC hyperparameter search (Namu), writer's
voice perspective (Charlotte), ambient orchestration of the workshop itself.

## The pattern (frame)

- Natalie Hogg, "Agentic research is oxymoronic", Nature Astronomy, 31 Aug
  2026 (arXiv 2608.31161). Research requires human interpretation; agentic
  systems that write the paper remove it; "deliteration" of the literature;
  LLMs legitimately for "grunt work" which they "usually get done faster than
  I can do it myself"; asks for disclosure of human interpretation, open
  data, longer early-career contracts. Says nothing about local models.
- David Yallup (18 Aug): "There are no good AI-generated papers. There still
  isn't, and we've been diverting a huge amount of resources to that. If it
  doesn't happen, it's because it systematically won't happen. It's not
  'what if we just prompted a bit harder'." Will's gloss (1 Sep): the
  agentic-army worry "is a very 2025 kind of worry; if it were possible, it
  was possible last year"; the real danger "is the corruption it does to
  our people, because they're so good at grunt work, and we should be using
  them for grunt work."
- James Alvey (14 Aug): "If that's what most people think and see AI is,
  it's up to us to make a better narrative"; "pick some concrete problems
  and really structure it… much more persuasive to me than talking about
  the personality of Fable Five"; "most people are actually compute limited
  rather than model limited."

## Ambient AI: what the lab has already done

Corpus: roughly 1,000 recorded meetings over 18 months; Otter to mdrecord
cards nightly; Parakeet transcription on newton; Gemma 4 serving on turing
at 64 concurrent behind intranet API keys; speaker-identification work on
the group's own recordings.

Live loops in meetings, from a sweep of all 394 meeting records April to
September 2026 (AI listening to the room and acting during the meeting):

| Date | Meeting | What happened live | Mechanism |
|---|---|---|---|
| 2026-04-24 | AInstein (Kwan, Hobson, Bolliet) | Claude, fed the running transcript and the DeepSeek paper, produced a pre-1905 reasoning-model training recipe read aloud to the room | Claude reading the live transcript |
| 2026-04-27 | David Yallup, two sessions | Read the flow-matching paper under discussion and explained the simulation-free step; later investigated JAX JIT recompilation costs; "it's read everything we've said" | Claude Code with transcript access |
| 2026-04-28 | Group meeting | Toby's local Whisper recorder injecting the room into Claude Code via hooks; "we realised we'd left it recording, so we didn't need to type those things in" | Whisper on CPU, ~5 s per 20 s of speech |
| 2026-04-28 | Xuelei Chen visit | otter-live skill drafting notes mid-meeting, answering the guest's spoken question, emailing the transcript; in parallel cloning CAMB and implementing an EDE module | Otter live skill, Claude Code |
| 2026-04-30 | Lawrence Berry | Suggested other many-uncoupled-ODE systems; LaTeX notes emailed | Otter to Claude, ~2 minutes to digest |
| 2026-04-30 | Sinah Legner, Will Barker | Cloned her CAMB fork and assessed PPF energy conservation against K-screening; emailed a summary; got OpenAI's second opinion | Otter to Claude, 30 s delay |
| 2026-05-01 | Namu Kroupa | Headless Claude Code implemented a radial custom kernel in blackjax and requested review; texted Namu from its own number | Headless Claude Code with MCP tools |
| 2026-05-05 | Website workshop | Claude Code as ambient orchestrator over the Otter feed, coordinating parallel PRs from about ten attendees | Otter polling, GitHub issue #3 |
| 2026-05-07 | Charlotte Priestley | Whiteboard supervision synthesised into a document; "did I make any mistakes?" caught a stray β; sent to Charlotte | Otter to Claude |
| 2026-05-13 | Madhusudhan | K2-18b retrieval repository built live in the room | Voice-driven Claude Code |
| 2026-05-15 | AInstein (Kwan) | Summary for absent members composed from the transcript with a specified emphasis order | Claude with transcript |
| 2026-05-18 | Jody Fletcher, Caius | GitHub issue filed on the wine-order bug, questions answered from the conversation, a red/white filter demo built and shown, minutes produced | Otter to Claude Code |
| 2026-06-04 | Toby Lovick | e-whiteboard screenshot plus audio to Gemini to fill in a half-remembered Klein–Gordon term; PDF with highlighted edits mid-meeting | Local recorder, Gemini |
| 2026-07-08 | Sienna (summer student) | Checked the nested-sampling workshop against upstream blackjax and proposed an update, approved verbally | Claude Code listening |
| 2026-07-23 | Summer students | Tursa onboarding documentation assembled live, including a default sbatch script, values looked up not guessed | Otter plus local Parakeet and pyannote; Claude Code |
| 2026-08-07 | James Alvey, Thomas Spieksma | ET Bluebook plot and verbatim quotes on screen 15 s after the words; only database IDs returned, no generated prose | Parakeet on newton, Gemma on turing |
| 2026-08-13 | Matt Grayling | State of the art on sparse GP representations answered from the transcript-fed session | Claude/Codex fed the transcript |
| 2026-08-27 | David Yallup | Charitable-but-firm text on the Liddle Bayes-factor paper drafted, sub-agents building a citation tree, git-latexdiff rendered on boltzmann | Claude Code and Codex |
| 2026-09-01 | Group meeting | Toby's Pi whiteboard camera: ink segmented from arm, blocks to turing for LaTeX; "what's missing from this equation" answered and projected in place (video) | Pi Zero camera, Gemma on turing, projector |
| 2026-09-03 | Toby Lovick | Pi Zero audio box transcribing live, turing detecting actionable items and writing tickets for Claude to claim; "check the thermals" answered two minutes later | Pi Zero and audio hat, Whisper, turing |
| 2026-09-04 | PolyChord team | Iman's inconsistent-evidence findings summarised precisely on request during the call | Claude listening |

Mechanism generations: April to June, Otter polling through a
reverse-engineered API into Claude Code, about 30 s round trip; from
mid-July a second generation on local Parakeet with pyannote diarisation;
August to September, Gemma on turing as the local model and Toby's Pi camera
and audio devices as ambient inputs with a projector for in-place output.

Misfires worth knowing (not for the slides): a recipe produced without the
project's actual state (24 Apr); Opus 4.7 implementing the wrong kernel
(1 May); a summary fixating on an aside (15 May); the Pi dying mid-demo
after a shorted screen hat (1 Sep); turing not triggering a ticket (3 Sep);
and the 30 s Otter latency throughout the spring.

## Multiplayer mode (the design argument)

- Anthropic's @Claude in internal Slack and OpenClaw 2.0 sell multiplayer
  mode: an agent with the team's shared context is "a real multiplier", not
  one more participant.
- For most scientists the job is not in Slack; Slack is an anti-pattern.
  You work alone or you meet, and the meeting is a specific event. "How do
  you bring AI into that?"
- Answer: make multiplayer mode ambient at the surfaces where research
  happens: whiteboard, two laptops at a desk, coffee, supervision. Stage one
  drops the "multi": an AI you never have to touch or look at. Stage two
  plugs it into always-on agents that pick up the next thing.
- Design rules from the room: there when I want it, never thrown at me
  (James); the device sits there, you behave as normal, and the AI does
  something that feels natural, or consensual (Will); "give me the numbers I
  need"; capture never leaves the room.
- Academia is defined by autonomy; collaboration is voluntary. With agents
  under each person, cooperation scale contracts to what is necessary.
  Multiplayer mode for scientists is small, organic and voluntary.

## Toby's loop (live demonstration)

- Pi Zero W with Codec Zero hat, on a phone hotspot, streaming to Whisper
  (Parakeet with diarisation next); turing decides what is actionable and
  writes tickets; each person's Claude claims its own tickets, which is also
  the injection defence found in the first 20 minutes when Alex's agent
  tried to task Toby's.
- Pi camera on the whiteboard: "where is this wrong?" round-trips to turing,
  answer projected back. Wide-angle lens; the equation occupied 5% of the
  pixels and was still read correctly.
- Logistics: own laptop and display feed on the HLT desk agreed with Debora
  and Sandro; test in the 11:00 coffee break; turing reachable from the IoA
  network; a recorded video as fallback. Confirm Toby is registered for the
  Monday.

## Why local

- The scarce thing was never the model. Vendors supply a harness and
  subsidised tokens. Will's own usage ran at "$1,000 a day, about the same
  as my ERC grant"; "the reckoning is coming where the price tag becomes
  honest."
- The threshold: the whole local chain (transcription, model, display)
  became fast enough to join a conversation rather than be set running.
  "A lot of latency can appear by accident." One graph: local STT and LLM
  latency against year, crossing conversational timescale around 2022 and
  closing by 2026. Lawrence's "isn't CLASS fast enough?" is the same
  question asked of physics codes.
- Instrumentation framing, not ideology: Cambridge astronomy has built its
  own instruments for a century; if you don't build it you get sold a puck.
- The exam case is the existence proof; the eval harness (with James and
  Nikhil: transcript-to-report, catalogue analysis, technical recall, paper
  reproduction) is how the local-sufficiency claim gets measured rather than
  asserted.

## Where it goes

- Toby's supervision pilot: whiteboard captured, the good extension from
  supervision three written down, cleaned per-group transcripts or a merged
  one, four transcripts plus whiteboards plus textbook giving "everything you
  learned"; zero change to the participants; more supervisors in Lent.
- The Cosmos Institute build (awarded $5,000, "Instrumenting the Unrecorded
  Half of Science"): wearable ring-buffer recorders, whiteboard imaging,
  display fabric, self-hosted storage, iterated over 90 days.
- Discussion prompts: what would you record if the recording never left the
  building; which part of your week would you hand to an ambient agent first.

## Out of scope for this talk

Alan as a system; the GPU-inference programme; the grants; failure modes as
a section; Dily's evidence grid; naming CMBAgent or Denario;
supervision-as-training.

## Source pointers

- Personal deck project "Ambient talk and grants" (fa0d11d4) and task
  "Prepare KAS2026 talk" (4de1e197), with mail evidence linked.
- Internal talk, April 2026: handley-lab/internal-talks branch
  `2026_ambient_ai` (handley-lab.tex, feedback.md).
- Master report: ~/reports/ambient-ai-ambitions.md (26 Jul 2026).
- Cosmos proposal text: ~/reports/cosmos-text-for-review.md.
- Cookbook: handley-lab/handley-lab.github.io `_data/recipes.yml`,
  `_recipes/jaxwavelets.md`.
- Records: personal deck `reference/record/2026-0[4-9]-*.md`.
- Slide tooling: codex skill `create-beamer-presentations` (bundles the
  house style from this repo's `ioa_2026` branch); build with
  `latexmk -pdf will_handley.tex`.
