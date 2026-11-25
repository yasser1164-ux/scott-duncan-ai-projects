# Scott Duncan's AI projects for process plants (copy)

A copy of the 7 open-source projects by **Scott Duncan**, a chemical engineer who builds AI tools for the plant floor.
All credit goes to him: [scottduncanai.com](https://scottduncanai.com/) · [GitHub: ScottDuncanAI](https://github.com/ScottDuncanAI) · [LinkedIn](https://www.linkedin.com/in/scottduncanai/)

Each folder is copied **unchanged** from his repository on 25 Nov 2026, with its own licence file. Only this README was added. For updates, go to the original repositories.

| Folder | What it does | Original | Copied at | Licence |
|---|---|---|---|---|
| [PID-to-Knowledge-Graph](PID-to-Knowledge-Graph) | Claude reads a P&ID PDF tile by tile; Python builds a knowledge graph of flow, instruments and control loops you can ask questions | [link](https://github.com/ScottDuncanAI/PID-to-Knowledge-Graph) | 7ee3a82 | MIT |
| [claude-manufacturing-skills](claude-manufacturing-skills) | PFD Generator: describe a process in plain English, get an editable PFD (SVG) | [link](https://github.com/ScottDuncanAI/claude-manufacturing-skills) | e8f66ff | MIT |
| [industrial-ai-troubleshooting-agent](industrial-ai-troubleshooting-agent) | Claude troubleshooting assistant for a boiler, connected to historian data, alarm log, knowledge graph and plant documents | [link](https://github.com/ScottDuncanAI/industrial-ai-troubleshooting-agent) | ef8672f | Apache 2.0 |
| [HistorianSim](HistorianSim) | `/simulate` a process in Claude Code and get 90 days of historian data with injected faults and an answer key | [link](https://github.com/ScottDuncanAI/HistorianSim) | 14e616c | MIT |
| [HistorianRCA](HistorianRCA) | Root-cause analysis challenge: 90 days of data from a steam-heated tank (TK-101) with hidden faults | [link](https://github.com/ScottDuncanAI/HistorianRCA) | 8d12a82 | CC BY 4.0 |
| [TEP-Claude-Code](TEP-Claude-Code) | Fault detection on the Tennessee Eastman Process: statistical monitoring (PCA T²/SPE) vs machine learning | [link](https://github.com/ScottDuncanAI/TEP-Claude-Code) | c11a515 | MIT |
| [public-chemical-manufacturing-datasets](public-chemical-manufacturing-datasets) | List of public P&ID sets and process time-series datasets for testing | [link](https://github.com/ScottDuncanAI/public-chemical-manufacturing-datasets) | 3f21a85 | CC BY 4.0 |

## Where to start

1. **PID-to-Knowledge-Graph**: copy `skills/pid-to-kg` into your `.claude/skills/` folder, open a folder with a P&ID PDF in Claude Code, and ask Claude to convert it. Public P&IDs to test on are listed in `public-chemical-manufacturing-datasets`.
2. **claude-manufacturing-skills**: same install for `skills/pfd-generator`, then describe a process.
3. **HistorianSim** and **industrial-ai-troubleshooting-agent**: open the folder in Claude Code and type `/simulate` or `/setup`.

Each folder's own README has the full instructions.
