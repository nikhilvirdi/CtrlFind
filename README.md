# Stenod

<p align="center">
  <img width="480" alt="Stenod — a black box recorder for your AI coding sessions" src="https://github.com/user-attachments/assets/de2bdd63-0cb7-447a-a2cb-f1ac3525748d" />
</p>

<p align="center">

[![CI](https://github.com/nikhilvirdi/stenod/actions/workflows/ci.yml/badge.svg)](https://github.com/nikhilvirdi/stenod/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/steno-daemon.svg)](https://www.npmjs.com/package/steno-daemon)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Node.js](https://img.shields.io/badge/Node.js-20%2B-339933?logo=node.js&logoColor=white)](https://nodejs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![SQLite](https://img.shields.io/badge/SQLite-WAL%20mode-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![better-sqlite3](https://img.shields.io/badge/better--sqlite3-driver-003B57)](https://github.com/WiseLibs/better-sqlite3)
[![Commander.js](https://img.shields.io/badge/CLI-commander.js-black)](https://github.com/tj/commander.js)
[![chokidar](https://img.shields.io/badge/fs%20watching-chokidar-blue)](https://github.com/paulmillr/chokidar)
[![node-pty](https://img.shields.io/badge/terminal-node--pty-blue)](https://github.com/microsoft/node-pty)
[![web-tree-sitter](https://img.shields.io/badge/AST%20parsing-web--tree--sitter-orange)](https://github.com/tree-sitter/tree-sitter)
[![wink-nlp](https://img.shields.io/badge/text%20analysis-wink--nlp-teal)](https://github.com/winkjs/wink-nlp)
[![mockttp](https://img.shields.io/badge/HTTPS%20interception-mockttp-purple)](https://github.com/httptoolkit/mockttp)
[![node-forge](https://img.shields.io/badge/CA%20generation-node--forge-purple)](https://github.com/digitalbazaar/forge)
[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-5A67D8)](https://modelcontextprotocol.io/)
[![Vitest](https://img.shields.io/badge/tested%20with-Vitest-6E9F18?logo=vitest&logoColor=white)](https://vitest.dev/)
[![ESLint](https://img.shields.io/badge/lint-ESLint-4B32C3?logo=eslint&logoColor=white)](https://eslint.org/)
[![Prettier](https://img.shields.io/badge/code%20style-Prettier-F7B93E?logo=prettier&logoColor=black)](https://prettier.io/)

</p>

A black box recorder for your AI coding sessions — now watching more than one at a time.

AI-assisted coding sessions collapse at boundaries: rate limits, provider outages, context-window exhaustion, a developer switching tools mid-task. What's lost at that moment isn't just chat history. It's the *reasoning* — which decisions were made and why, which approaches were tried and rejected, and what you were in the middle of doing.

Stenod is a local, deterministic daemon that watches your files, your terminal, and — where a tool allows it — the reasoning behind what an AI agent just did, and compiles all of it into a Handoff Manifest the moment you need to resume, in any AI tool, cold. It's out-of-band by design: it never depends on an AI being alive or reachable, because it was never inside that AI to begin with. That also means it doesn't care whether one agent is working your project or three of them are, at once, in different windows — it watches the shared ground under all of them.

**Stenod 1.0** was filesystem and terminal capture for a single project. **Stenod 2.0**, described below, adds structured capture from the coding agents themselves — Claude Code, Codex, Kiro, Google Antigravity, Cursor — merged into one shared graph and interpreted as a current-truth scoreboard rather than a raw log. Full reasoning for both: [ARCHITECTURE.md](./ARCHITECTURE.md). Exactly what's captured and how: [SECURITY.md](./SECURITY.md).

---

## Install

```
npm install -g steno-daemon
```

The package on npm is named `steno-daemon` (a naming-availability adjustment); the command you run is `stenod`.

## Quick Start

```
stenod init
stenod start
```

> [!IMPORTANT]
> `stenod start` alone only captures **filesystem** events — it never spawns a shell itself, since the daemon runs fully detached. To also capture a **terminal** session, run `stenod attach` in each interactive shell you want captured, once per session. To capture an AI tool's own reasoning, see [Supported Tools](#supported-tools) below.

Work normally: save files, run commands, let your AI agent do its thing. When you need to hand off:

```
stenod handoff
```

The compiled Handoff Manifest — current decisions, plus everything tried and rejected along the way, with the reasons — is copied to your clipboard. Paste it into any AI tool to resume where you left off.

## Supported Tools

Stenod's filesystem and terminal capture works underneath any tool, automatically, with no setup — that's the baseline every project gets. On top of that, some tools expose enough of their own internals for Stenod to also capture *why* a change was made, not just that it happened. Where that's available, turn it on with:

```
stenod integrate <tool>
```

| Tool | What gets captured | Setup |
|---|---|---|
| Claude Code | Full reasoning, via the tool's own hook system | `stenod integrate claude-code` |
| Codex | Full reasoning, via hooks | `stenod integrate codex` |
| Kiro | Full reasoning, via hooks, plus its own spec files | `stenod integrate kiro` |
| Google Antigravity | Full reasoning, read from its session files directly | `stenod integrate antigravity` |
| Cursor | Tool-call and file-edit reasoning, via its agent hooks | `stenod integrate cursor` |
| VS Code + GitHub Copilot, Windsurf | Filesystem and terminal only — neither exposes anything more stable to build against yet | none needed, works by default |

`stenod integrate` only ever adds to a tool's own configuration, never touches what's already there, and prints exactly what it changed. `stenod detach <tool>` undoes it and confirms the removal.

## Command Reference

| Command | Description |
|---|---|
| `stenod init [--reset]` | Set up the daemon + database for the current project. `--reset` rotates the local auth token. |
| `stenod start [--project-root <path>] [--foreground]` | Start the ingestion daemon (filesystem capture by default). `--foreground` runs it attached to your terminal instead of detached. |
| `stenod stop` | Stop the daemon cleanly. |
| `stenod status` | Show daemon health, node count, last event timestamp, and any stale rule files or flagged conflicts. |
| `stenod attach` | Bridge the current interactive shell's terminal activity to the daemon. Run once per terminal session you want captured. |
| `stenod integrate <tool>` | Turn on structured capture for a supported tool. See [Supported Tools](#supported-tools). |
| `stenod detach <tool>` | Remove exactly what `integrate` added for that tool. |
| `stenod handoff [--full \| --new] [--worked \| --failed] [--token-budget <n>]` | Compile and copy the Handoff Manifest to your clipboard. `--full` (the default) gives the complete current picture; `--new` gives just what's changed since your last handoff. `--worked`/`--failed` tags the outcome of the most recent manifest. `--token-budget` overrides the default packing budget. |
| `stenod anchor "<text>"` | Create a `CONSTRAINT` node directly — a decision or rule you want the compiler to always include. |
| `stenod reject --since <duration>` | Mark nodes from a time window (e.g. `15m`) as rejected, excluded from all future manifests. |
| `stenod enable-network-capture` | Opt in to capturing AI-provider network traffic (installs a local CA, starts a local proxy). Unix/Mac only. |
| `stenod disable-network-capture` | Fully revert the network-capture tier: removes the CA from the OS trust store, reverts proxy settings. |
| `stenod mcp` | Run as an MCP server over stdio, exposing the handoff manifest as a resource for MCP-connected clients. |

## How It Works

`chokidar` watches your project's files, `stenod attach` bridges in terminal activity, and — for tools you've integrated — hooks or session files bring in the reasoning behind what an agent did. All of it feeds one causal graph in local SQLite, using typed edges instead of a vector store, so the system tracks which decisions are still true rather than which ones merely sound similar.

Every decision lands on a scoreboard with one of three states: **settled** (the current answer), **rejected** (tried and abandoned, with the reason kept permanently next to it), or **open** (raised, not yet resolved). Nothing is ever deleted — a rejected idea stays visible so a fresh agent can't quietly re-propose it. Turning raw reasoning text into scoreboard entries runs through deterministic rules first, and only reaches for an optional AI assist — your own API key, never Stenod's, and only when you explicitly run a handoff — for genuinely ambiguous cases. Without that key configured, Stenod still works in full; the rare ambiguous item just gets flagged instead of auto-sorted.

A deterministic compiler then packs the highest-value entries into a token budget with greedy-by-ratio knapsack packing, structures the output U-shaped (settled decisions and constraints first, causal history in the middle, next steps last) to work with how transformers attend to context, and delivers it via clipboard or, optionally, as an MCP resource or a local dashboard.

Full detail: [ARCHITECTURE.md](./ARCHITECTURE.md).

## Companion Site

A small documentation site covers the architecture, install steps, and command reference in one place, alongside this README. There's also an optional local dashboard — run from your own machine, showing your own handoffs update in real time — that never leaves your machine and requires no login, because it's never reachable from anywhere but `localhost`. Neither replaces the CLI; both are read-only companions to it. Details in [ARCHITECTURE.md](./ARCHITECTURE.md#12-companion-website).

## Security & Privacy

Zero telemetry, always. Local-only storage, including anything captured from an integrated AI tool. Stenod reads a tool's own rule and instruction files to flag when they've gone stale against current decisions, but never edits them. The optional AI tie-breaker spends nothing on Stenod's behalf — it runs on your own key, only when you ask for a handoff. The network-capture tier is opt-in, allowlists exactly three known AI-provider domains, and ships with a full, confirmed-clean uninstall path — the same pattern `stenod integrate`/`detach` follow for AI-tool hooks.

Full detail: [SECURITY.md](./SECURITY.md).

## Platform Support

Filesystem capture and the core CLI work on Windows, Linux, and Mac. Terminal capture (`stenod attach`) and the network-capture tier depend on Unix/Mac-only mechanisms (`node-pty`, OS trust-store APIs) and aren't available on Windows. Tool integrations follow the same split: Claude Code, Codex, Kiro, and Antigravity integrations work wherever those tools run; Cursor's hook-based capture is Unix/Mac only, matching the rest of the terminal-dependent tier.

## License & Contributing

MIT © Nikhil Virdi — see [LICENSE](./LICENSE) for details.

This project isn't yet accepting external contributions; no `CONTRIBUTING.md` exists yet. That's a deliberate early-stage choice, not an oversight. Contribution infrastructure may be added later.